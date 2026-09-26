// SPDX-License-Identifier: GPL-3.0-or-later

//! The RAW-to-image pipeline: sensor decode, mono reconstruction, downscale,
//! tone shaping, and geometry (crop/rotate/orient). Every CPU bake — grid
//! thumbnails, roll covers, detail overviews, and exports — funnels through the
//! functions here, so grid == detail == export hold structurally; the GPU
//! detail shader shares the tone math via `shader::tone_model`.

use std::path::PathBuf;

use cosmic::widget::image::Handle;

use crate::edit_manifest::{CropMargins, ToneEdit};
use crate::error::FrameError;
use crate::film::{
    self, ACTIVE_STOCK, BaseConfig, FilmPreset, MIN_PLAUSIBLE_BASE, MonoStock, measure_base,
};
use crate::shader;
fn normalize_samples(image: &rawloader::RawImage) -> Vec<f32> {
    let width = usize::max(image.width, 1);

    match &image.data {
        rawloader::RawImageData::Integer(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                // Black/white levels are per-channel, so look them up by CFA position.
                let color = image.cfa.color_at(index / width, index % width);
                let black = f32::from(image.blacklevels[color]);
                let span = (f32::from(image.whitelevels[color]) - black).max(f32::EPSILON);

                (f32::from(*value) - black).clamp(0.0_f32, span) / span
            })
            .collect(),
        rawloader::RawImageData::Float(values) => {
            let max = values.iter().copied().fold(0.0_f32, f32::max);
            let gain = if max > f32::EPSILON { 1.0 / max } else { 1.0 };

            values
                .iter()
                .map(|value| (*value * gain).clamp(0.0, 1.0))
                .collect()
        }
    }
}

/// Slices a sample buffer down to the usable area described by rawloader's
/// `[top, right, bottom, left]` crops, discarding masked sensor borders.
pub(crate) fn crop_samples(
    samples: &[f32],
    width: usize,
    height: usize,
    crops: [usize; 4],
) -> Option<(Vec<f32>, usize, usize)> {
    let [top, right, bottom, left] = crops;
    if right + left >= width || top + bottom >= height || samples.len() < width * height {
        return None;
    }

    let (out_width, out_height) = (width - right - left, height - top - bottom);
    let mut cropped = Vec::with_capacity(out_width * out_height);
    for y in top..top + out_height {
        let row = y * width + left;
        cropped.extend_from_slice(&samples[row..row + out_width]);
    }

    Some((cropped, out_width, out_height))
}

/// Approximate number of photosites sampled per CFA class when measuring
/// clear-film bases in [`flatten_bayer`].
pub(crate) const BASE_SAMPLE_TARGET: usize = 250_000;

/// Reconstructs a monochrome negative from bayer samples, one output pixel per
/// 2×2 CFA cell.
///
/// Monochrome film carries no color signal, so each CFA class is treated as an
/// independent density measurement: every class's clear-film transmission is
/// measured from a strided subsample and the classes are rescaled onto one
/// common base (neutralizing the light-table/sensor cast). Because the four
/// sites of a cell then measure the *same* luminance on one common scale, they
/// are averaged — gain first, then mean — halving the read noise versus
/// emitting a single photosite per pixel. The output is half-resolution in each
/// axis (a final partial row/column for odd dimensions is edge-replicated).
///
/// This is a luminance reconstruction for a monochrome scene, not a color
/// demosaic; color negatives will need a real demosaic pass later.
///
/// Returns the mono samples plus their (halved) dimensions.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn flatten_bayer(
    samples: &[f32],
    width: usize,
    height: usize,
    cfa: &rawloader::CFA,
) -> (Vec<f32>, u32, u32) {
    let pixels = width * height;
    // An odd step cannot alias with the period-2 CFA grid.
    let step = usize::max(pixels / BASE_SAMPLE_TARGET, 1) | 1;

    let mut class_samples: [Vec<f32>; 4] = Default::default();
    for idx in (0..pixels).step_by(step) {
        let (y, x) = (idx / width, idx % width);
        class_samples[cfa.color_at(y, x)].push(samples[idx]);
    }

    let mut anchored = [ACTIVE_STOCK.base; 4];
    for (base, class) in anchored.iter_mut().zip(&class_samples) {
        if let Some(measured) =
            measure_base(class).filter(|measured| *measured >= MIN_PLAUSIBLE_BASE)
        {
            *base = measured;
        }
    }
    // Anchor onto the green class when present (best SNR, keeps magnitudes
    // close to true transmissions); otherwise the dimmest measurable class.
    let empty = std::array::from_fn(|class| class_samples[class].is_empty());
    let gains = class_gains(anchored, empty);

    // Gain-normalize every site onto the common base BEFORE averaging, so the
    // four classes of a cell are comparable measurements of one luminance.
    let gain_at = |y: usize, x: usize| gains[cfa.color_at(y, x)];

    let out_w = width.div_ceil(2);
    let out_h = height.div_ceil(2);
    let last_x = width.saturating_sub(1);
    let last_y = height.saturating_sub(1);
    let mut mono = Vec::with_capacity(out_w * out_h);
    for oy in 0..out_h {
        let y0 = oy * 2;
        // Edge-replicate the final partial cell's row/column.
        let y1 = usize::min(y0 + 1, last_y);
        for ox in 0..out_w {
            let x0 = ox * 2;
            let x1 = usize::min(x0 + 1, last_x);
            let mut sum = 0.0_f32;
            let mut count = 0.0_f32;
            for y in [y0, y1] {
                for x in [x0, x1] {
                    sum += samples[y * width + x] * gain_at(y, x);
                    count += 1.0;
                }
            }
            mono.push(sum / count);
        }
    }

    (mono, out_w as u32, out_h as u32)
}

/// Gains that rescale the four CFA classes onto one common base.
///
/// Anchors onto the green class when present (best SNR, keeps magnitudes close
/// to true transmissions); otherwise the dimmest measurable class. A class
/// with no samples at its measurement resolution is rescaled by its anchored
/// base like any other. Mirrored by the fused thumbnail downscaler so the
/// full-res detail path and the collapsed thumbnail path share one rule.
///
/// `anchored` and `empty` are indexed by CFA color (0=R, 1=G, 2=B, 3=fourth).
pub(crate) fn class_gains(anchored: [f32; 4], empty: [bool; 4]) -> [f32; 4] {
    let reference = if empty[1] {
        let Some(dimmest) = anchored
            .iter()
            .zip(empty)
            .filter(|(_, class_empty)| !*class_empty)
            .map(|(base, _)| *base)
            .reduce(f32::min)
        else {
            // No measurable class at all; keep every class unscaled.
            return [1.0; 4];
        };
        dimmest
    } else {
        anchored[1]
    };

    std::array::from_fn(|class| reference / anchored[class])
}

/// Multiplies linear samples by `2^EV` in place, mirroring the detail
/// shader's gain so CPU and GPU rendering stay bit-consistent.
pub(crate) fn apply_exposure(mono: &mut [f32], exposure_ev: f32) {
    let gain = f32::exp2(exposure_ev);
    for value in mono {
        *value *= gain;
    }
}

/// Encodes a linear intensity into the sRGB transfer function.
pub(crate) fn srgb_encode(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Collapses interleaved linear RGB to Rec.709 luminance.
///
/// Only meaningful in linear light, where luminance coefficients are defined;
/// stripping the channels this way removes any capture cast from monochrome
/// film scans outright.
pub(crate) fn luma(rgb: &[f32]) -> Vec<f32> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|&[r, g, b]| 0.212_6 * r + 0.715_2 * g + 0.072_2 * b)
        .collect()
}

/// Start and length of the source range that output coordinate `out` covers,
/// offset by `origin` (the cropped edge on that axis).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn block_span(out: u32, source: u32, out_total: u32, origin: usize) -> (usize, usize) {
    let start = range(out, source, out_total) + origin;
    let len = range_len(start, range(out + 1, source, out_total) + origin);
    (start, len)
}

/// Rec.709 luminance of three per-block channel means.
fn luma_of_means(r: f32, g: f32, b: f32) -> f32 {
    0.212_6 * r + 0.715_2 * g + 0.072_2 * b
}

/// Fused normalization, cropping, and phase-preserving downscale.
///
/// Reads the raw sensor samples exactly once into per-output-pixel
/// accumulators, bypassing the full-resolution linear buffer the separate
/// [`normalize_samples`]/[`crop_samples`]/[`flatten_bayer`]/[`resize_area`]
/// steps build. The size math and block mapping match [`resize_area`] (never
/// upscales). Returns a small linear negative-space mono; inversion happens in
/// the caller.
///
/// RGB sources collapse through the Rec.709 luminance of each block's channel
/// means — exact, because luma is linear. Bayer mosaics keep each CFA class
/// separate per block and rescale the class means onto one common base with
/// the same [`class_gains`] rule the detail path uses, so a heavy downscale
/// averages each site's cast independently (flatten-before-average tone
/// accepted over the full-res invert-then-average ordering).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn downsample_thumbnail(
    image: &rawloader::RawImage,
    max_size: u32,
) -> Option<(Vec<f32>, u32, u32)> {
    let width = usize::max(image.width, 1);
    let height = usize::max(image.height, 1);
    let [top, right, bottom, left] = image.crops;
    let cw = width.checked_sub(right.saturating_add(left))?;
    let ch = height.checked_sub(top.saturating_add(bottom))?;
    if cw == 0 || ch == 0 {
        return None;
    }

    let scale = f32::min(
        1.0,
        f32::min(max_size as f32 / cw as f32, max_size as f32 / ch as f32),
    );
    let out_w = u32::max((cw as f32 * scale) as u32, 1);
    let out_h = u32::max((ch as f32 * scale) as u32, 1);

    let mono = if image.cpp >= 3 {
        downsample_rgb(image, out_w as usize, out_h as usize)?
    } else {
        downsample_bayer(image, out_w as usize, out_h as usize)?
    };

    Some((mono, out_w, out_h))
}

/// Fused RGB-source branch of [`downsample_thumbnail`]: block channel means
/// collapsed through Rec.709 luminance.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn downsample_rgb(image: &rawloader::RawImage, out_w: usize, out_h: usize) -> Option<Vec<f32>> {
    let width = usize::max(image.width, 1);
    let height = usize::max(image.height, 1);
    let cpp = usize::max(image.cpp, 1);
    let [top, right, bottom, left] = image.crops;
    let (cw, ch) = (width - right - left, height - top - bottom);

    let mut mono = Vec::with_capacity(out_w * out_h);

    match &image.data {
        rawloader::RawImageData::Integer(values) => {
            if values.len() < width * height * cpp {
                return None;
            }
            for out_y in 0..out_h as u32 {
                let (row_start, rows) = block_span(out_y, ch as u32, out_h as u32, top);
                for out_x in 0..out_w as u32 {
                    let (col_start, cols) = block_span(out_x, cw as u32, out_w as u32, left);
                    let mut sums = [0.0_f32; 3];
                    for y in row_start..row_start + rows {
                        for x in col_start..col_start + cols {
                            // Black/white levels are per-channel, looked up by
                            // CFA position, matching normalize_samples.
                            let color = image.cfa.color_at(y, x);
                            let black = f32::from(image.blacklevels[color]);
                            let span =
                                (f32::from(image.whitelevels[color]) - black).max(f32::EPSILON);
                            let offset = (y * width + x) * cpp;
                            for (channel, sum) in sums.iter_mut().enumerate() {
                                let value = f32::from(values[offset + channel]);
                                *sum += (value - black).clamp(0.0, span) / span;
                            }
                        }
                    }
                    let count = (rows * cols) as f32;
                    mono.push(luma_of_means(
                        sums[0] / count,
                        sums[1] / count,
                        sums[2] / count,
                    ));
                }
            }
        }
        rawloader::RawImageData::Float(values) => {
            let max = values.iter().copied().fold(0.0_f32, f32::max);
            let gain = if max > f32::EPSILON { 1.0 / max } else { 1.0 };
            for out_y in 0..out_h as u32 {
                let (row_start, rows) = block_span(out_y, ch as u32, out_h as u32, top);
                for out_x in 0..out_w as u32 {
                    let (col_start, cols) = block_span(out_x, cw as u32, out_w as u32, left);
                    let mut sums = [0.0_f32; 3];
                    for y in row_start..row_start + rows {
                        for x in col_start..col_start + cols {
                            let offset = (y * width + x) * cpp;
                            for (channel, sum) in sums.iter_mut().enumerate() {
                                *sum += (values[offset + channel] * gain).clamp(0.0, 1.0);
                            }
                        }
                    }
                    let count = (rows * cols) as f32;
                    mono.push(luma_of_means(
                        sums[0] / count,
                        sums[1] / count,
                        sums[2] / count,
                    ));
                }
            }
        }
    }

    Some(mono)
}

/// Fused bayer branch of [`downsample_thumbnail`]: per-CFA-class block means
/// rescaled onto one common base.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn downsample_bayer(image: &rawloader::RawImage, out_w: usize, out_h: usize) -> Option<Vec<f32>> {
    let width = usize::max(image.width, 1);
    let height = usize::max(image.height, 1);
    let [top, right, bottom, left] = image.crops;
    let (cw, ch) = (width - right - left, height - top - bottom);
    let out_pixels = out_w * out_h;

    let mut sums = vec![[0.0_f32; 4]; out_pixels];
    let mut counts = vec![[0_u32; 4]; out_pixels];

    match &image.data {
        rawloader::RawImageData::Integer(values) => {
            if values.len() < width * height {
                return None;
            }
            for out_y in 0..out_h as u32 {
                let (row_start, rows) = block_span(out_y, ch as u32, out_h as u32, top);
                for out_x in 0..out_w as u32 {
                    let (col_start, cols) = block_span(out_x, cw as u32, out_w as u32, left);
                    let slot = out_y as usize * out_w + out_x as usize;
                    for y in row_start..row_start + rows {
                        for x in col_start..col_start + cols {
                            let color = image.cfa.color_at(y, x);
                            let black = f32::from(image.blacklevels[color]);
                            let span =
                                (f32::from(image.whitelevels[color]) - black).max(f32::EPSILON);
                            let value = f32::from(values[y * width + x]);
                            sums[slot][color] += (value - black).clamp(0.0, span) / span;
                            counts[slot][color] += 1;
                        }
                    }
                }
            }
        }
        rawloader::RawImageData::Float(values) => {
            let max = values.iter().copied().fold(0.0_f32, f32::max);
            let gain = if max > f32::EPSILON { 1.0 / max } else { 1.0 };
            for out_y in 0..out_h as u32 {
                let (row_start, rows) = block_span(out_y, ch as u32, out_h as u32, top);
                for out_x in 0..out_w as u32 {
                    let (col_start, cols) = block_span(out_x, cw as u32, out_w as u32, left);
                    let slot = out_y as usize * out_w + out_x as usize;
                    for y in row_start..row_start + rows {
                        for x in col_start..col_start + cols {
                            let color = image.cfa.color_at(y, x);
                            let value = values[y * width + x];
                            sums[slot][color] += (value * gain).clamp(0.0, 1.0);
                            counts[slot][color] += 1;
                        }
                    }
                }
            }
        }
    }

    // Collapse each class to its block mean, measure a per-class base, and
    // rescale onto one common base with the same anchoring the detail path
    // uses. A class absent from a block simply does not contribute to that
    // output pixel.
    let mut class_values: [Vec<f32>; 4] = Default::default();
    for (slot, sums) in sums.iter().enumerate() {
        for class in 0..4 {
            if counts[slot][class] != 0 {
                class_values[class].push(sums[class] / counts[slot][class] as f32);
            }
        }
    }

    let mut anchored = [ACTIVE_STOCK.base; 4];
    for (base, class) in anchored.iter_mut().zip(&class_values) {
        if let Some(measured) =
            measure_base(class).filter(|measured| *measured >= MIN_PLAUSIBLE_BASE)
        {
            *base = measured;
        }
    }
    let empty = std::array::from_fn(|class| class_values[class].is_empty());
    let gains = class_gains(anchored, empty);

    let mut mono = Vec::with_capacity(out_pixels);
    for (slot, sums) in sums.iter().enumerate() {
        let mut sum = 0.0_f32;
        let mut non_empty = 0_u32;
        for class in 0..4 {
            if counts[slot][class] != 0 {
                sum += (sums[class] / counts[slot][class] as f32) * gains[class];
                non_empty += 1;
            }
        }
        let level = if non_empty == 0 {
            ACTIVE_STOCK.base
        } else {
            sum / non_empty as f32
        };
        mono.push(level);
    }

    Some(mono)
}

/// Scales source-pixel crop margins onto a print of target dimensions.
///
/// Both the source dims and the print must be in the SAME (display-oriented)
/// frame: the caller resolves the full-resolution display dimensions via
/// [`display_source_dims`] (the sensor's post-masked-border dims, rotated to
/// upright), so the print's horizontal axis always maps back to the source's
/// horizontal and no axis-swap detection is needed here. A rotation merely
/// swaps which of `src_w`/`src_h` the caller passes.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn scale_crop(
    crop: CropMargins,
    src_w: u32,
    src_h: u32,
    out_width: u32,
    out_height: u32,
) -> CropMargins {
    let sx = f64::from(out_width) / f64::from(src_w.max(1));
    let sy = f64::from(out_height) / f64::from(src_h.max(1));
    CropMargins {
        top: (f64::from(crop.top) * sy).round() as u32,
        right: (f64::from(crop.right) * sx).round() as u32,
        bottom: (f64::from(crop.bottom) * sy).round() as u32,
        left: (f64::from(crop.left) * sx).round() as u32,
    }
}

/// The display-upright dimensions of a sensor whose post-masked-border dims
/// are `(cw, ch)`: any orientation that swaps the print axes (90°/270°
/// rotation, transpose) maps the display horizontal onto the sensor vertical.
pub(crate) fn display_source_dims(cw: u32, ch: u32, orientation: rawloader::Orientation) -> (u32, u32) {
    use rawloader::Orientation;
    match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transpose
        | Orientation::Transverse => (ch, cw),
        _ => (cw, ch),
    }
}

/// Removes the given margins from an RGBA frame into a new, smaller buffer.
/// Returns the frame unchanged when the margins overrun the extent so a
/// degenerate crop can never collapse a thumbnail.
pub(crate) fn crop_rgba(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    crop: CropMargins,
) -> (Vec<u8>, u32, u32) {
    let cw = crop.cropped_width(width);
    let ch = crop.cropped_height(height);
    if cw == 0 || ch == 0 {
        return (rgba, width, height);
    }
    let mut out = Vec::with_capacity((cw * ch) as usize * 4);
    let mut rows = 0u32;
    let top = crop.top.min(height.saturating_sub(ch));
    for y in top..top.saturating_add(ch) {
        let row = y as usize * width as usize + crop.left as usize;
        let start = row * 4;
        if start >= rgba.len() {
            break;
        }
        let end = (start + cw as usize * 4).min(rgba.len());
        out.extend_from_slice(&rgba[start..end]);
        rows += 1;
    }
    (out, cw, rows)
}

/// Rotates an RGBA buffer counter-clockwise by `turns` 90° quarter turns
/// (`0`…`3`, masked to `& 3`), swapping the print dims on odd turns. The
/// output naturally has the swapped dimensions (no re-fit is attempted):
/// [`convert_thumbnail`] hands the already-downscaled thumbnail to the GPU at
/// the same aspect, and the detail shader's rotated-fraction UV layout matches
/// this exact pixel mapping, so the grid tile and the detail view stay in
/// agreement. `0` (and any multiple of 4) is the identity — the untouched bake
/// stays byte-identical.
pub(crate) fn rotate_quarters(rgba: Vec<u8>, width: u32, height: u32, turns: u8) -> (Vec<u8>, u32, u32) {
    let out_w = height;
    let out_h = width;
    match turns & 3 {
        0 => (rgba, width, height),
        1 => {
            let mut out = vec![0_u8; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let src = ((y * width + x) as usize) * 4;
                    let ox = y;
                    let oy = width - 1 - x;
                    let dst = ((oy * out_w + ox) as usize) * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                }
            }
            (out, out_w, out_h)
        }
        2 => {
            let mut out = vec![0_u8; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let src = ((y * width + x) as usize) * 4;
                    let oy = height - 1 - y;
                    let ox = width - 1 - x;
                    let dst = ((oy * width + ox) as usize) * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                }
            }
            (out, width, height)
        }
        _ => {
            let mut out = vec![0_u8; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let src = ((y * width + x) as usize) * 4;
                    let ox = height - 1 - y;
                    let oy = x;
                    let dst = ((oy * out_w + ox) as usize) * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                }
            }
            (out, out_w, out_h)
        }
    }
}

/// Applies the export geometry to an RGBA print: crops to `crop` (scaled by the
/// caller to the print's native dims) then quarter-turns to `rotation`, matching
/// the thumbnail/detail ordering (crop-then-rotate). Shared by the JPEG and PNG
/// export encoders so both bake the same framing.
pub(crate) fn bake_geometry(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    crop: CropMargins,
    rotation: u8,
) -> (Vec<u8>, u32, u32) {
    let (rgba, width, height) = if crop == CropMargins::default() {
        (rgba, width, height)
    } else {
        crop_rgba(rgba, width, height, crop)
    };
    rotate_quarters(rgba, width, height, rotation)
}

/// The 16-bit analogue of [`rotate_quarters`]: rotates an RGBA buffer of `u16`
/// samples counter-clockwise by `turns` 90° quarter turns, swapping the print
/// dims on odd turns. Used by the 16-bit PNG export path.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rotate_quarters16(rgba: Vec<u16>, width: u32, height: u32, turns: u8) -> (Vec<u16>, u32, u32) {
    let out_w = height;
    let out_h = width;
    match turns & 3 {
        0 => (rgba, width, height),
        1 => {
            let mut out = vec![0_u16; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let src = ((y * width + x) as usize) * 4;
                    let ox = y;
                    let oy = width - 1 - x;
                    let dst = ((oy * out_w + ox) as usize) * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                }
            }
            (out, out_w, out_h)
        }
        2 => {
            let mut out = vec![0_u16; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let src = ((y * width + x) as usize) * 4;
                    let oy = height - 1 - y;
                    let ox = width - 1 - x;
                    let dst = ((oy * width + ox) as usize) * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                }
            }
            (out, width, height)
        }
        _ => {
            let mut out = vec![0_u16; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let src = ((y * width + x) as usize) * 4;
                    let ox = height - 1 - y;
                    let oy = x;
                    let dst = ((oy * out_w + ox) as usize) * 4;
                    out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
                }
            }
            (out, out_w, out_h)
        }
    }
}

/// Convenience for [`rotate_quarters`] vs [`rotate_quarters16`]: the export
/// path picks the deeper buffer's rotation directly.
pub(crate) fn bake_geometry16(
    rgba: Vec<u16>,
    width: u32,
    height: u32,
    crop: CropMargins,
    rotation: u8,
) -> (Vec<u16>, u32, u32) {
    let (rgba, width, height) = if crop == CropMargins::default() {
        (rgba, width, height)
    } else {
        crop_rgba16(rgba, width, height, crop)
    };
    rotate_quarters16(rgba, width, height, rotation)
}

/// The 16-bit analogue of [`crop_rgba`]: crops an RGBA buffer of `u16` samples
/// to the given margins (which the caller scaled to the print's dims), letting
/// a degenerate crop leave the frame unchanged.
fn crop_rgba16(
    rgba: Vec<u16>,
    width: u32,
    height: u32,
    crop: CropMargins,
) -> (Vec<u16>, u32, u32) {
    let cw = crop.cropped_width(width);
    let ch = crop.cropped_height(height);
    if cw == 0 || ch == 0 {
        return (rgba, width, height);
    }
    let mut out = Vec::with_capacity((cw * ch) as usize * 4);
    let mut rows = 0u32;
    let top = crop.top.min(height.saturating_sub(ch));
    for y in top..top.saturating_add(ch) {
        let row = y as usize * width as usize + crop.left as usize;
        let start = row * 4;
        if start >= rgba.len() {
            break;
        }
        let end = (start + cw as usize * 4).min(rgba.len());
        out.extend_from_slice(&rgba[start..end]);
        rows += 1;
    }
    (out, cw, rows)
}

/// Measures the tone pivots (shadow/mid-gray/white) a frame needs — WITHOUT
/// mutating `mono` — plus, for a film negative, the resolved clear-film base and
/// its stock (needed by the density inversion).
///
/// The inverted (film) path uses the shader's EV-exact mechanic: raw sensor
/// fractiles measured on the intact pre-gain buffer, mapped through
/// `invert_value(fractile · 2^-EV)` so the pivots describe the ACTUAL render at
/// the current exposure. The positive path measures the regular anchors on the
/// positive. Returning the pair separately lets the whole-frame parity test feed
/// `render_tail` and the WGSL reference the identical inputs.
pub(crate) fn pivots_for(
    mono: &[f32],
    tone: ToneEdit,
    preset: FilmPreset,
    base_config: BaseConfig,
) -> (Option<(MonoStock, f32)>, (f32, f32, f32)) {
    if let Some(stock) = preset.stock() {
        // Measured BEFORE any gain so the ranks stay in the shader's upload
        // domain (the gain would shift them).
        let fractiles = shader::film_anchor_fractiles(mono);
        let measured = if base_config.auto {
            measure_base(mono)
        } else {
            None
        };
        let base = base_config.resolve(measured, &stock);
        let pivots = shader::film_pivots_at_gain(
            fractiles,
            base,
            shader::sensor_gain(tone.exposure_ev, true),
            stock,
        );
        (Some((stock, base)), pivots)
    } else {
        (None, shader::tone_anchors(mono))
    }
}

/// Applies the detail shader's exact per-pixel tone ordering to a mono buffer in
/// place, then sRGB-encodes: for a film negative the EV gain hits the true
/// sensor data first (`2^-EV`), then the density inversion, then the pivoted
/// curve; for an already-positive scan the curve comes first, then the `2^EV`
/// gain. `stock_and_base` is `Some((stock, base))` exactly when the buffer is a
/// negative to invert. Shared by the grid thumbnail bake, the export bake, and
/// the WGSL-parity tests.
fn render_tail(
    mono: &mut [f32],
    tone: ToneEdit,
    stock_and_base: Option<(MonoStock, f32)>,
    pivots: (f32, f32, f32),
) {
    if let Some((stock, base)) = stock_and_base {
        // Film path: EV gain on the true sensor data, density inversion with a
        // Contrast-scaled window, then the density-domain toe/shoulder shape —
        // the exact ordering and math the WGSL film branch applies. Contrast
        // and the region controls are read as STOP LIFTS (their film meaning),
        // recovered from the decoded powers by the same maps the UI uses:
        // contrast `+log2`, highlights `+log2`, shadows `-log2`.
        apply_exposure(mono, -tone.exposure_ev);
        let d_max_eff = film::effective_d_max(stock.d_max, tone.curve_contrast.log2());
        let shadows = film::region_strength(-tone.curve_shadows.log2());
        let highlights = film::region_strength(tone.curve_highlights.log2());
        for value in mono.iter_mut() {
            let positive =
                film::invert_value_graded(*value, base, d_max_eff, stock.gamma);
            *value = film::region_shape(positive, shadows, highlights);
        }
    } else {
        // Already-positive path keeps the pivoted-power tone curve.
        let (shadow, mid, white) = pivots;
        shader::apply_curve(
            mono,
            tone.curve_contrast,
            tone.curve_highlights,
            tone.curve_shadows,
            shadow,
            mid,
            white,
        );
        apply_exposure(mono, tone.exposure_ev);
    }
    for value in mono {
        *value = srgb_encode(*value);
    }
}

/// The one shared tone tail for every CPU bake (grid thumbnails and exports):
/// measure the pivots from `mono`, then apply the shader's exact ordering and
/// sRGB-encode. Grid == detail == export hold structurally because this single
/// body (plus its WGSL-parity tests) is the only place the tone math lives for
/// the non-shader paths.
pub(crate) fn bake_tone(
    mono: &mut [f32],
    tone: ToneEdit,
    preset: FilmPreset,
    base_config: BaseConfig,
) {
    let (stock_and_base, pivots) = pivots_for(mono, tone, preset, base_config);
    render_tail(mono, tone, stock_and_base, pivots);
}

/// Converts a decoded RAW image into a small oriented RGBA image, scaled so no
/// dimension exceeds `max_size`, baking the tone edit (`ToneEdit`: exposure,
/// curve powers) and the display rotation into the pixels.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn convert_thumbnail(
    image: &rawloader::RawImage,
    max_size: f32,
    tone: ToneEdit,
    crop: CropMargins,
    rotation: u8,
    preset: FilmPreset,
    base_config: BaseConfig,
) -> Result<Handle, crate::error::FrameError> {
    // One fused pass: normalize, discard masked borders, and phase-preserve
    // downscale straight from the sensor samples into a small TRUE sensor-linear
    // mono (the same domain the detail decode produces, for every preset).
    let (mut mono, width, height) =
        downsample_thumbnail(image, max_size as u32).ok_or(crate::error::FrameError::ShortSamples)?;

    // Restore edge punch lost to the heavy downscale — in true sensor space for
    // every preset, matching the detail decode's unsharp so a film negative
    // carries its sharpening INTO the density inversion instead of leaving it
    // on the positive.
    unsharp_mask(&mut mono, width as usize, height as usize);

    // The one shared tone tail: measure the EV-exact (film) or histogram
    // (positive) pivots, apply the shader's exact ordering per preset (gain
    // before the density inversion for a negative, curve then gain for a
    // positive scan), then sRGB-encode — the same `bake_tone` every CPU bake
    // uses, so grid == detail == export hold structurally.
    bake_tone(&mut mono, tone, preset, base_config);

    let mut rgba = Vec::with_capacity(mono.len() * 4);
    for &value in &mono {
        let level = (value * 255.0).round() as u8;
        rgba.extend_from_slice(&[level, level, level, 255]);
    }

    let (rgba, width, height) = orient(&rgba, width, height, image.orientation);

    // Bake the live crop into the print. The persisted margins are authored in
    // full-resolution DISPLAY source pixels (matching the detail shader's
    // source frame); resolve the oriented source dims and scale them onto the
    // oriented print. A zero crop is a no-op.
    let (rgba, width, height) = if crop == CropMargins::default() {
        (rgba, width, height)
    } else {
        let src_w = usize::max(image.width, 1);
        let src_h = usize::max(image.height, 1);
        let [ct, cr, cb, cl] = image.crops;
        let cw = src_w.saturating_sub(cr.saturating_add(cl)).max(1) as u32;
        let ch = src_h.saturating_sub(ct.saturating_add(cb)).max(1) as u32;
        let (disp_w, disp_h) = display_source_dims(cw, ch, image.orientation);
        let scaled = scale_crop(crop, disp_w, disp_h, width, height);
        crop_rgba(rgba, width, height, scaled)
    };

    // Bake the user's display rotation: quarter-turn the already EXIF-oriented
    // (and cropped) print so the grid tile matches the rotated detail view.
    // The crop margins are authored in the EXIF-upright source frame, so the
    // rotation composes AFTER the crop sub-rect was taken — the same ordering
    // as the detail shader's uniform rotate (`rotate_ccw(crop(orient(EXIF)))`).
    let (rgba, width, height) = rotate_quarters(rgba, width, height, rotation);

    Ok(Handle::from_rgba(width, height, rgba))
}

/// Output dimensions a downscale of `src_w` × `src_h` to a `max_edge` long-edge
/// cap would produce: both axes scale by the same factor `max_edge / long_edge`
/// (≤ 1), preserving aspect, with the float truncation [`resize_area`] applies.
/// Single-sourcing the size math lets the detail view project the dimensions a
/// native (level-up) decode will produce before it lands, so the zoom cap can
/// track the upcoming texture's 1:1 to the pixel.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
#[must_use]
pub(crate) fn resized_dims(src_w: u32, src_h: u32, max_edge: u32) -> (u32, u32) {
    let scale = f32::min(
        1.0,
        max_edge as f32 / f32::max(src_w.max(1) as f32, src_h.max(1) as f32),
    );
    (
        u32::max((src_w as f32 * scale) as u32, 1),
        u32::max((src_h as f32 * scale) as u32, 1),
    )
}

/// Downscales an interleaved buffer of `channels`-component samples so no
/// dimension exceeds `max`, never upscaling. Each output pixel averages the
/// block of source pixels that maps into it, which suppresses noise and grain
/// aliasing.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn resize_area(
    samples: &[f32],
    width: u32,
    height: u32,
    max: u32,
    channels: usize,
) -> (Vec<f32>, u32, u32) {
    let (out_width, out_height) = resized_dims(width, height, max);

    let mut resized = Vec::with_capacity(out_width as usize * out_height as usize * channels);
    let mut sums = vec![0.0_f32; channels];
    for out_y in 0..out_height {
        let row_start = range(out_y, height, out_height);
        let rows = range_len(row_start, range(out_y + 1, height, out_height));
        for out_x in 0..out_width {
            let col_start = range(out_x, width, out_width);
            let cols = range_len(col_start, range(out_x + 1, width, out_width));
            let count = (rows * cols) as f32;

            sums.fill(0.0);
            for y in row_start..row_start + rows {
                for x in col_start..col_start + cols {
                    let offset = (y * width as usize + x) * channels;
                    for (channel, sum) in sums.iter_mut().enumerate() {
                        *sum += samples[offset + channel];
                    }
                }
            }

            resized.extend(sums.iter().map(|sum| sum / count));
        }
    }

    (resized, out_width, out_height)
}

/// Start of the source range that output coordinate `out` covers.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn range(out: u32, source: u32, out_total: u32) -> usize {
    (u64::from(out) * u64::from(source) / u64::from(out_total)) as usize
}

/// Length of the source range between `start` and the next output coordinate's
/// start, always covering at least one source pixel.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn range_len(start: usize, next_start: usize) -> usize {
    usize::max(next_start.saturating_sub(start), 1)
}

/// Strength of the post-downscale unsharp mask; 0 disables.
pub(crate) const UNSHARP_AMOUNT: f32 = 0.4;

/// Applies a gentle unsharp mask to a linear buffer, restoring edge punch lost
/// to heavy downscaling. Runs before tone encoding so overshoot stays out of
/// the perceptually amplified display range and shadow noise stays quiet.
pub(crate) fn unsharp_mask(samples: &mut [f32], width: usize, height: usize) {
    let blurred = blur_121(samples, width, height);
    for (slot, blur) in samples.iter_mut().zip(blurred) {
        *slot = (*slot + UNSHARP_AMOUNT * (*slot - blur)).clamp(0.0, 1.0);
    }
}

/// Separable 3x3 binomial blur ([1, 2, 1] per axis), replicating edges.
fn blur_121(samples: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0_f32; samples.len()];
    for y in 0..height {
        for x in 0..width {
            let left = samples[y * width + x.saturating_sub(1)];
            let center = samples[y * width + x];
            let right = samples[y * width + usize::min(x + 1, width - 1)];

            horizontal[y * width + x] = (left + 2.0 * center + right) / 4.0;
        }
    }

    let mut blurred = vec![0.0_f32; samples.len()];
    for y in 0..height {
        let up = y.saturating_sub(1);
        let down = usize::min(y + 1, height - 1);
        for x in 0..width {
            blurred[y * width + x] = (horizontal[up * width + x]
                + 2.0 * horizontal[y * width + x]
                + horizontal[down * width + x])
                / 4.0;
        }
    }

    blurred
}

/// Applies the RAW orientation metadata to a linear mono buffer.
///
/// Mirror of [`orient`] for `Vec<f32>` data going to the GPU shader: same
/// transformations, per-element instead of per-RGBA-bunch. Equivalent in
/// behaviour to orientation-after-quantization when the per-element source
/// `(sx, sy)` mapping is identical.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn orient_mono(
    mono: &[f32],
    width: u32,
    height: u32,
    orientation: rawloader::Orientation,
) -> (Vec<f32>, u32, u32) {
    use rawloader::Orientation;

    let (out_width, out_height) = match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transpose
        | Orientation::Transverse => (height, width),
        _ => (width, height),
    };

    let mut oriented = vec![0.0_f32; (out_width * out_height) as usize];
    for y in 0..out_height {
        for x in 0..out_width {
            let (sx, sy) = match orientation {
                Orientation::Normal | Orientation::Unknown => (x, y),
                Orientation::HorizontalFlip => (width - 1 - x, y),
                Orientation::Rotate180 => (width - 1 - x, height - 1 - y),
                Orientation::VerticalFlip => (x, height - 1 - y),
                Orientation::Transpose => (y, x),
                Orientation::Transverse => (height - 1 - y, width - 1 - x),
                Orientation::Rotate90 => (y, height - 1 - x),
                Orientation::Rotate270 => (width - 1 - y, x),
            };

            let source = (sy * width + sx) as usize;
            let target = (y * out_width + x) as usize;
            oriented[target] = mono[source];
        }
    }

    (oriented, out_width, out_height)
}

/// Applies the RAW orientation metadata to an RGBA buffer.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn orient(
    rgba: &[u8],
    width: u32,
    height: u32,
    orientation: rawloader::Orientation,
) -> (Vec<u8>, u32, u32) {
    use rawloader::Orientation;

    let (out_width, out_height) = match orientation {
        Orientation::Rotate90
        | Orientation::Rotate270
        | Orientation::Transpose
        | Orientation::Transverse => (height, width),
        _ => (width, height),
    };

    let mut oriented = vec![0_u8; (out_width * out_height * 4) as usize];
    for y in 0..out_height {
        for x in 0..out_width {
            let (sx, sy) = match orientation {
                Orientation::Normal | Orientation::Unknown => (x, y),
                Orientation::HorizontalFlip => (width - 1 - x, y),
                Orientation::Rotate180 => (width - 1 - x, height - 1 - y),
                Orientation::VerticalFlip => (x, height - 1 - y),
                Orientation::Transpose => (y, x),
                Orientation::Transverse => (height - 1 - y, width - 1 - x),
                Orientation::Rotate90 => (y, height - 1 - x),
                Orientation::Rotate270 => (width - 1 - y, x),
            };

            let source = ((sy * width + sx) * 4) as usize;
            let target = ((y * out_width + x) * 4) as usize;
            oriented[target..target + 4].copy_from_slice(&rgba[source..source + 4]);
        }
    }

    (oriented, out_width, out_height)
}
/// Runs a RAW decode plus mono reconstruction on a blocking worker thread,
/// returning TRUE sensor-linear `mono` (post-downscale, post-unsharp) oriented
/// to display upright, for every preset — the shader applies the exposure gain
/// and (for film) the density inversion per fragment. `max_edge` is the
/// downscale target for the long edge before unsharp.
///
/// The `src_long_edge` field is the sensor's true long edge AFTER cropping but
/// BEFORE the downscale — i.e. the real native long edge the overview was
/// scaled down from (< `max_edge` means the overview is already full-res). For
/// a bayer source the mono is first CFA-averaged to half resolution (see
/// [`flatten_bayer`]), so the returned `width`/`height` are half the cropped
/// sensor dims while `src_long_edge` stays in sensor pixels.
/// `inversion` is `Some((stock, base))` when the preset marks a film negative,
/// threading the clear-film anchor to the shader; the roll's `base_config`
/// resolves that anchor preset-first (calibration, then the auto opt-in, then
/// the stock's preset base).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) async fn decode_raw_detail(
    dir: PathBuf,
    name: String,
    max_edge: u32,
    preset: FilmPreset,
    base_config: BaseConfig,
) -> Result<DetailDecode, FrameError> {
    let path = dir.join(name);

    tokio::task::spawn_blocking(move || {
        let image = rawloader::decode_file(&path).map_err(|err| FrameError::Decode {
            path: path.clone(),
            message: err.to_string(),
        })?;

        let width = usize::max(image.width, 1);
        let height = usize::max(image.height, 1);

        let normalized = normalize_samples(&image);

        let (samples, width, height) = match crop_samples(&normalized, width, height, image.crops) {
            Some(cropped) => cropped,
            None => (normalized, width, height),
        };

        let src_long_edge = u32::max(width as u32, height as u32);

        let (mono, width, height) = if image.cpp >= 3 {
            if samples.len() < width * height * 3 {
                return Err(FrameError::ShortSamples);
            }

            let mut rgb = Vec::with_capacity(width * height * 3);
            for pixel in 0..width * height {
                let base = pixel * image.cpp;
                rgb.push(samples[base]);
                rgb.push(samples[base + 1]);
                rgb.push(samples[base + 2]);
            }

            (luma(&rgb), width, height)
        } else {
            if samples.len() < width * height {
                return Err(FrameError::ShortSamples);
            }

            let cfa = image.cfa.shift(image.crops[3], image.crops[0]);
            // `flatten_bayer` averages each 2×2 CFA cell into one output pixel,
            // so it returns its own (halved) dimensions.
            let (mono, out_w, out_h) = flatten_bayer(&samples, width, height, &cfa);
            (mono, out_w as usize, out_h as usize)
        };

        // True sensor-linear data for every preset: an already-positive scan
        // (None) and a film negative both stay linear `[0,1]` relative to the
        // sensor white point — the exposure gain touches the RAW values, and
        // the density inversion for a negative happens per fragment in the
        // shader. The only film-side work here is resolving the frame's
        // clear-film anchor (the inversion's black point) from the roll's base
        // mode: calibration, then the per-frame auto opt-in (measured from the
        // sensor data), then the stock's preset base.
        let inversion = preset.stock().map(|stock| {
            let measured = if base_config.auto {
                measure_base(&mono)
            } else {
                None
            };
            (stock, base_config.resolve(measured, &stock))
        });

        let (mono, width, height) = resize_area(&mono, width as u32, height as u32, max_edge, 1);

        let mut mono = mono;
        unsharp_mask(&mut mono, width as usize, height as usize);

        let (oriented, width, height) = orient_mono(&mono, width, height, image.orientation);

        Ok(DetailDecode {
            mono: oriented,
            width,
            height,
            src_long_edge,
            inversion,
        })
    })
    .await
    .unwrap_or(Err(FrameError::ThreadPanic))
}
/// A decoded true sensor-linear mono frame plus the data the detail and export
/// paths need to shape it.
///
/// `mono` is linear `[0,1]` relative to the sensor white point for EVERY preset
/// (an already-positive scan or a film negative) — the single source of truth;
/// the exposure gain touches it, and per-fragment shaping (density inversion
/// for film) happens downstream per preset.
#[derive(Debug, Clone)]
pub(crate) struct DetailDecode {
    pub(crate) mono: Vec<f32>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The sensor's true long edge AFTER cropping but BEFORE any downscale.
    /// Kept in SENSOR pixels (not the CFA-averaged mono's) so the crop margins
    /// — authored against the full-resolution display frame by the grid bake
    /// and keyboard trims — stay in one reference frame across grid, detail,
    /// and export. The detail texture is a downscale of `mono`, which for a
    /// bayer source is already 2× CFA-averaged, so its true 1:1 is reached
    /// earlier; the `src_long_edge`/texture aspect is preserved, so
    /// [`shader::DetailProgram`] still recovers the correct source dims.
    pub(crate) src_long_edge: u32,
    /// `None` for an already-positive scan; `(stock, base)` for a film negative
    /// the shader must density-invert (`base` is the clear-film anchor).
    pub(crate) inversion: Option<(MonoStock, f32)>,
}
