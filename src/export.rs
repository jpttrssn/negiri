// SPDX-License-Identifier: GPL-3.0-or-later

//! Exporting the roll's frames to JPEG/PNG: the format/size/preset options,
//! file naming (plain, dated, roll-hashed), and the encoders. Each frame is
//! decoded through the shared [`crate::pipeline`] and baked with its stored
//! edits, then written atomically via a same-directory temp+rename.

use std::path::{Path, PathBuf};

use crate::edit_manifest;
use crate::error::FrameError;
use crate::exif_writer;
use crate::film::Develop;
use crate::fl;
use crate::pipeline::{
    DetailDecode, bake_develop, bake_geometry, bake_geometry16, decode_raw_detail, scale_crop,
};

/// A curated export preset: the default combination of format, bit depth, and
/// size for a destination (cloud backup, further processing, the web).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExportPreset {
    /// Cloud: JPEG, quality 90, native resolution — the classic "hand these to
    /// an archive service" output.
    Cloud,
    /// Master: lossless 16-bit PNG at native resolution, so an external editor
    /// gets the full dynamic range to work with.
    Master,
    /// Web: JPEG, quality 82, long edge capped at 2048 px.
    Web,
}

impl ExportPreset {
    /// The user-visible label for the preset, as shown in the dialog's format
    /// dropdown.
    #[must_use]
    fn choice_label(self) -> String {
        match self {
            Self::Cloud => fl!("export-choice-jpeg-90"),
            Self::Web => fl!("export-choice-jpeg-82"),
            Self::Master => fl!("export-choice-png-16"),
        }
    }
}

/// The export container format: the sRGB-baked frame is either lossy-compressed
/// as a JPEG (the `jpeg-encoder` path) or written losslessly as a PNG via the
/// raw `png` crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExportFormat {
    Jpeg,
    Png,
}

impl ExportFormat {
    /// The filename extension for the format (`jpg` / `png`).
    #[must_use]
    fn ext(self) -> &'static str {
        match self {
            Self::Jpeg => "jpg",
            Self::Png => "png",
        }
    }
}

/// Long-edge limit for the export decode. Original keeps native resolution;
/// the numbered variants downscale the frame before baking (sharing the
/// decoded buffer with the overview renderer).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ExportSize {
    #[default]
    Original,
    LongEdge2048,
}

impl ExportSize {
    /// The target long edge in pixels, or `0` for native resolution.
    #[must_use]
    fn long_edge(self) -> u32 {
        match self {
            Self::Original => 0,
            Self::LongEdge2048 => 2048,
        }
    }
}

/// The full export parameter set: seeded by a preset on open, then adjustable
/// through the dialog's controls until Save. Behavior is driven entirely by
/// these fields — the preset that seeded them is not stored (the dropdown's
/// format choice resolves straight to the options in `options_for_choice`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExportOptions {
    format: ExportFormat,
    quality: u8,
    size: ExportSize,
    /// Pixel density tagged in the output header: 0 writes none (the encode
    /// crate's default), otherwise JPEG tags JFIF dots-per-inch and PNG tags a
    /// `pHYs` pixels-per-meter value. Pure metadata — pixel data is untouched.
    ppi: u16,
    /// Whether an existing `<stem>.<ext>` in the destination is replaced. The
    /// dialog's overwrite checkbox defaults to `false`: frames whose file
    /// already exists are skipped rather than re-encoded.
    overwrite: bool,
}

impl ExportOptions {
    /// The options a preset imports, and the exact combo the dialog's format
    /// dropdown restores for its key.
    #[must_use]
    fn for_preset(preset: ExportPreset) -> Self {
        match preset {
            ExportPreset::Cloud => Self {
                format: ExportFormat::Jpeg,
                quality: 90,
                size: ExportSize::Original,
                ppi: 300,
                overwrite: false,
            },
            ExportPreset::Master => Self {
                format: ExportFormat::Png,
                quality: 100,
                size: ExportSize::Original,
                ppi: 300,
                overwrite: false,
            },
            ExportPreset::Web => Self {
                format: ExportFormat::Jpeg,
                quality: 82,
                size: ExportSize::LongEdge2048,
                ppi: 72,
                overwrite: false,
            },
        }
    }

    /// Returns a copy with the overwrite flag set, as selected by the dialog's
    /// checkbox (the only option the user adjusts after the preset import).
    #[must_use]
    pub(crate) fn with_overwrite(self, overwrite: bool) -> Self {
        Self { overwrite, ..self }
    }
}

/// Resolves a key returned by the dialog's format choice back into the
/// export options that key stands for. Unknown keys (or a dialog backend that
/// dropped the choice) fall back to the first preset.
#[must_use]
pub(crate) fn options_for_choice(key: &str) -> ExportOptions {
    let preset = match key {
        "jpeg-82" => ExportPreset::Web,
        "png-16" => ExportPreset::Master,
        // "jpeg-90" and any unknown key (a backend that dropped the choice)
        // fall back to the first preset.
        _ => ExportPreset::Cloud,
    };
    ExportOptions::for_preset(preset)
}

/// The dialog's "format" choice: the three export presets as one dropdown,
/// defaulting to the first (JPEG 90% full-res). The response returns the
/// selected key, which [`options_for_choice`] resolves back into options.
#[must_use]
pub(crate) fn export_format_choice() -> cosmic::dialog::file_chooser::Choice {
    let jpeg_90 = ExportPreset::Cloud.choice_label();
    let jpeg_82 = ExportPreset::Web.choice_label();
    let png_16 = ExportPreset::Master.choice_label();
    cosmic::dialog::file_chooser::Choice::new("format", &fl!("export-choice-label"), "jpeg-90")
        .insert("jpeg-90", &jpeg_90)
        .insert("jpeg-82", &jpeg_82)
        .insert("png-16", &png_16)
}

/// The dialog's "overwrite" checkbox, defaulting to unchecked (existing
/// `<stem>.<ext>` files are kept rather than replaced). The response returns
/// its state as the string `"true"` / `"false"`.
#[must_use]
pub(crate) fn export_overwrite_choice() -> cosmic::dialog::file_chooser::Choice {
    cosmic::dialog::file_chooser::Choice::boolean("overwrite", &fl!("export-overwrite"), false)
}

/// Whether `name` looks like an output of this app's exporter (JPEG or PNG)
/// rather than a source negative. Negatives are RAW files (none of them use
/// these extensions), while exporting a roll into its own folder would
/// otherwise make every shipped file show back up as a fake frame — and,
/// without this, even become the roll's cover, which no scan ever searches
/// this directory for.
#[must_use]
pub(crate) fn is_export_artifact(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, ext)| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
}

/// The exported file name for a frame: the source file name with its extension
/// replaced by the format's (e.g. `img_0001.cr2` → `img_0001.jpg` or
/// `img_0001.png`). A file name with no extension simply gains the extension; a
/// hidden leading dot is part of the stem.
#[must_use]
fn export_name(name: &str, format: ExportFormat) -> PathBuf {
    let mut out = PathBuf::from(name);
    out.set_extension(format.ext());
    out
}

/// A deterministic 6-hex-char fingerprint of a roll's directory: FNV-1a over
/// the path's lossy bytes, masked to 24 bits and lowercased. Fixed-width so
/// filenames built from it sort lexicographically; specified independently of
/// Rust (`std::hash::DefaultHasher` is not stable across versions, so it is
/// never used here) and identical on every platform/run — the same directory,
/// however it was spelled, always yields the same id.
#[must_use]
fn roll_hash(dir: &std::path::Path) -> String {
    let mut hash = 0x811c_9dc5u32;
    for &byte in dir.to_string_lossy().as_bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{:06x}", hash & 0x00ff_ffff)
}

/// The dated export file name for a frame of a dated roll:
/// `YYYYMMDD-<roll-hash>-<frame>.<ext>` — e.g. `20240509-1a2b3c-01.jpg` — where
/// `YYYYMMDD` is the roll's start date (the same prefix for every frame, so
/// rolled shoots group under one day), `<roll-hash>` is the roll's 6-hex
/// [`roll_hash`] (the second sort key, so same-day rolls stay grouped when
/// filenames sort), and `<frame>` is the frame's 1-based full-roll position
/// padded to two digits (widening naturally past 99). Uses `format.ext()`.
/// Returns `None` when `start_iso` is not exactly `YYYY-MM-DD` or the position
/// does not fit a `u16` frame.
#[must_use]
fn dated_export_name(
    format: ExportFormat,
    start_iso: &str,
    index: usize,
    roll_hash: &str,
) -> Option<PathBuf> {
    let iso = start_iso.as_bytes();
    if iso.len() != 10 || iso[4] != b'-' || iso[7] != b'-' {
        return None;
    }
    let date = start_iso.get(..4)?.to_owned() + &start_iso[5..7] + &start_iso[8..10];
    let frame = u16::try_from(index).ok()? + 1;
    let mut out = PathBuf::from(format!("{date}-{roll_hash}-{frame:02}"));
    out.set_extension(format.ext());
    Some(out)
}

/// A same-directory sibling for an atomic export write: the encode goes to
/// `dest`'s `.tmp` neighbor and is renamed over `dest` once it is fully on
/// disk. Same filesystem, so the rename is atomic — a failed or interrupted
/// export leaves any pre-existing `dest` untouched instead of truncating it.
#[must_use]
fn temp_export_path(dest: &Path) -> PathBuf {
    let mut name = dest
        .file_name()
        .map_or_else(|| "export".to_owned(), |n| n.to_string_lossy().into_owned());
    name.push_str(".tmp");
    dest.with_file_name(name)
}

/// The determinate fraction (0.0..=1.0) an export batch has reached after
/// `done` of `total` frames, for the header's progress ring.
#[must_use]
#[allow(clippy::cast_precision_loss)] // frame counts are far below f32's exact range
pub(crate) fn export_fraction(done: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        done as f32 / total as f32
    }
}

/// Exports every listed frame into `dest` per the given options, sequentially,
/// so only one full-resolution decode is in flight at a time (bounded transient
/// memory). After each frame — written, skipped, or failed — `progress` is
/// called with the running `(done, total)` counts so the caller can stream
/// live progress to the UI. Returns how many frames succeeded, how many were
/// skipped (their output already existed and overwrite was off), and how many
/// failed.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn export_frames(
    dir: PathBuf,
    dest: PathBuf,
    frames: Vec<(String, Develop, edit_manifest::CropMargins, u8, usize)>,
    options: ExportOptions,
    start_date: Option<String>,
    mut progress: impl FnMut(usize, usize) + Send,
) -> (usize, usize, usize) {
    let mut ok = 0;
    let mut skipped = 0;
    let mut failed = 0;
    let total = frames.len();
    let mut done = 0;
    // Dated rolls export under `YYYYMMDD-<roll-hash>-<frame>.<ext>` (see
    // `dated_export_name`); the roll's directory fingerprint is shared by the
    // whole batch. Undated rolls keep the plain `<stem>.<ext>` names.
    let roll_hash = roll_hash(&dir);
    for (name, tone, crop, rotation, index) in frames {
        let target = dest.join(
            start_date
                .as_deref()
                .and_then(|start| dated_export_name(options.format, start, index, &roll_hash))
                .unwrap_or_else(|| export_name(&name, options.format)),
        );
        // With overwrite off, an already-present file is kept as-is: the frame
        // is skipped before any decode.
        if !options.overwrite && target.exists() {
            skipped += 1;
        } else if export_one(
            dir.clone(),
            name,
            index,
            target,
            tone,
            crop,
            rotation,
            options,
            start_date.clone(),
        )
        .await
        .is_ok()
        {
            ok += 1;
        } else {
            failed += 1;
        }
        done += 1;
        progress(done, total);
    }
    (ok, skipped, failed)
}

/// Decodes a frame (at native resolution, or downscaled when the size option
/// caps the long edge) and writes `dest` per the options: a JPEG at the chosen
/// quality, or a lossless 16-bit grayscale PNG. The stored tone curve,
/// exposure, crop, and display rotation are baked into the pixels — the same
/// edit pipeline as the detail view and thumbnails, just at the chosen scale
/// (the overview/native downscale is skipped for `Original`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn export_one(
    dir: PathBuf,
    name: String,
    index: usize,
    dest: PathBuf,
    develop: Develop,
    crop: edit_manifest::CropMargins,
    rotation: u8,
    options: ExportOptions,
    start_date: Option<String>,
) -> Result<(), FrameError> {
    let max_edge = options.size.long_edge();
    // True sensor-linear mono (masked-border cropped, downscaled per the size,
    // unsharpened — the exact detail-view decode), at the size option's long
    // edge or full native resolution. `src_long_edge` is the pre-downscale
    // long edge, so the factor pins how far below native the print is; the
    // film inversion (if any) is applied in the bake below, not in the decode.
    let DetailDecode {
        mut mono,
        width,
        height,
        src_long_edge,
    } = decode_raw_detail(
        dir.clone(),
        name.clone(),
        if max_edge == 0 { u32::MAX } else { max_edge },
    )
    .await
    .map_err(|err| {
        log::error!("export decode failed for {name}: {err}");
        err
    })?;

    // The crop margins are authored in display-upright source pixels; scale
    // them onto the (possibly downscaled) print. When the decode was already
    // at native resolution the factor is 1.0 and this is the identity.
    let factor = if max_edge > 0 && src_long_edge > max_edge {
        max_edge as f32 / src_long_edge as f32
    } else {
        1.0
    };
    let source_w = (width as f32 / factor).round() as u32;
    let source_h = (height as f32 / factor).round() as u32;
    let crop = scale_crop(crop, source_w, source_h, width, height);

    // Bake the resolved density develop then sRGB-encode via the one shared
    // tail (`bake_develop`) — the exact pointwise math the detail shader runs
    // per fragment, so export == detail hold structurally.
    bake_develop(&mut mono, develop);

    // When the roll carries a start date, stamp each output's DateTimeOriginal
    // with `start date @ 00:00:00 + frame's full-roll offset in seconds` — the
    // synthetic film-capture time that keeps exports in chronological order in
    // photo/cloud apps. DateTimeDigitized mirrors the RAW's own scan time
    // (read as its raw `YYYY:MM:DD HH:MM:SS` EXIF text) when it has one. Rolls
    // without a start date export un-stamped. The extra EXIF parse runs once
    // per frame beside the (far heavier) full decode.
    let tiff = start_date.as_deref().and_then(|start| {
        let original = exif_writer::shifted_datetime(start, index)?;
        let digitized = exif_writer::raw_datetime_original(&dir, &name);
        exif_writer::build_tiff(&original, digitized.as_deref())
    });

    match options.format {
        ExportFormat::Jpeg => export_jpeg(
            mono,
            width,
            height,
            crop,
            rotation,
            &dest,
            tiff.as_deref(),
            options,
        ),
        ExportFormat::Png => export_png(
            mono,
            width,
            height,
            crop,
            rotation,
            &dest,
            tiff.as_deref(),
            options,
        ),
    }
}

/// Writes `dest` as a quality-`options` JPEG, quantizing the sRGB mono buffer
/// to a single luminance channel (same shared bake as `export_png`). When
/// `options.ppi` is non-zero the JFIF header tags that dots-per-inch density.
/// An optional `exif_tiff` blob carries DateTimeOriginal/DateTimeDigitized and
/// is spliced into the JPEG stream as an Exif APP1 segment right after the SOI
/// byte-order marker.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[allow(clippy::too_many_arguments)]
fn export_jpeg(
    mono: Vec<f32>,
    width: u32,
    height: u32,
    crop: edit_manifest::CropMargins,
    rotation: u8,
    dest: &Path,
    exif_tiff: Option<&[u8]>,
    options: ExportOptions,
) -> Result<(), FrameError> {
    let mut rgba = Vec::with_capacity(mono.len() * 4);
    for &value in &mono {
        let level = (value * 255.0).round() as u8;
        rgba.extend_from_slice(&[level, level, level, 255]);
    }
    // Free the linear mono before the (similar-sized) steps below.
    drop(mono);

    let (rgba, width, height) = bake_geometry(rgba, width, height, crop, rotation);

    // Negatives render monochrome: all four channels carry the same level, so
    // drop to a single luminance channel. Revisit once color negatives are
    // supported (see NOTES.md).
    let mut gray: Vec<u8> = Vec::with_capacity(width as usize * height as usize);
    for px in rgba.chunks(4) {
        gray.push(px[0]);
    }
    drop(rgba);

    // The JPEG header fields are u16; exports are far below that, so the
    // conversion can only fail for absurd geometry — bail before creating any
    // temp file.
    let width = u16::try_from(width).map_err(|_| FrameError::Export {
        message: "export dimensions exceed the JPEG format's limits".to_owned(),
    })?;
    let height = u16::try_from(height).map_err(|_| FrameError::Export {
        message: "export dimensions exceed the JPEG format's limits".to_owned(),
    })?;

    // Encode into an in-memory buffer so we can splice the optional Exif APP1
    // after the SOI marker, then atomically flush the final byte stream to disk
    // with a same-directory rename.
    let mut bytes = Vec::with_capacity(gray.len() + 200);
    let mut encoder = jpeg_encoder::Encoder::new(&mut bytes, options.quality);
    // The default header is a bare (1,1) pixel-aspect-ratio; tag a real density
    // so print/layout tools scale the file by its intended PPI.
    if options.ppi != 0 {
        encoder.set_density(jpeg_encoder::PixelDensity::dpi(options.ppi));
    }
    // `encode` consumes the encoder (flush-on-drop), so its output is complete
    // by the time it returns; only then can we splice and rename.
    if encoder
        .encode(&gray, width, height, jpeg_encoder::ColorType::Luma)
        .is_err()
    {
        return Err(FrameError::Export {
            message: "JPEG encode failed".to_owned(),
        });
    }
    if let Some(tiff) = exif_tiff {
        let app1 = exif_writer::jpeg_app1(tiff);
        bytes = exif_writer::splice_after_soi(&bytes, &app1);
    }
    let tmp = temp_export_path(dest);
    std::fs::write(&tmp, &bytes).map_err(|err| FrameError::Export {
        message: format!("failed to write {}: {err}", tmp.display()),
    })?;
    std::fs::rename(&tmp, dest).map_err(|err| FrameError::Export {
        message: format!("failed to rename {} into place: {err}", tmp.display()),
    })
}

/// Writes `dest` as a lossless 16-bit grayscale PNG via the `png` crate,
/// quantizing the sRGB mono buffer to `u16` luminance samples (big-endian —
/// the byte order the PNG container requires, written as-is by the raw `png`
/// crate). When `options.ppi` is non-zero a `pHYs` chunk tags that density in
/// pixels-per-meter. An optional `exif_tiff` blob carries
/// DateTimeOriginal/DateTimeDigitized as an `eXIf` chunk (the raw TIFF, which
/// PNG stores without JPEG's `Exif\0\0` prefix).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#[allow(clippy::too_many_arguments)]
fn export_png(
    mono: Vec<f32>,
    width: u32,
    height: u32,
    crop: edit_manifest::CropMargins,
    rotation: u8,
    dest: &Path,
    exif_tiff: Option<&[u8]>,
    options: ExportOptions,
) -> Result<(), FrameError> {
    let mut rgba = Vec::with_capacity(mono.len() * 4);
    for &value in &mono {
        let level = (value * 65535.0).round() as u16;
        rgba.extend_from_slice(&[level, level, level, u16::MAX]);
    }
    drop(mono);
    let (rgba, width, height) = bake_geometry16(rgba, width, height, crop, rotation);

    // Negatives render monochrome: all four channels carry the same level, so
    // drop to a single luminance channel. Revisit once color negatives are
    // supported (see NOTES.md). Samples are stored big-endian — the byte order
    // the PNG container requires; the raw `png` crate writes them as-is (the
    // `image` wrapper used to re-swap them for us, hence the earlier
    // native-endian notes).
    let mut gray = Vec::with_capacity(width as usize * height as usize * 2);
    for px in rgba.chunks(4) {
        gray.extend_from_slice(&px[0].to_be_bytes());
    }
    drop(rgba);

    // Encode to a same-directory temp file, then rename over `dest` only once
    // every byte is on disk: a mid-encode failure must not truncate (or, with
    // overwrite on, replace) an existing file.
    let tmp = temp_export_path(dest);
    let file = std::fs::File::create(&tmp).map_err(|err| FrameError::Export {
        message: format!("failed to create {}: {err}", tmp.display()),
    })?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    // The samples are sRGB-encoded; a `sRGB` chunk makes that interpretation
    // deterministic for color-managed consumers (print pipelines especially),
    // instead of relying on an unstated viewer default.
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    // `pHYs` stores pixels per meter (1 in = 0.0254 m), unlike JFIF's per-inch.
    if options.ppi != 0 {
        let ppm = (f32::from(options.ppi) / 0.0254).round() as u32;
        encoder.set_pixel_dims(Some(png::PixelDimensions {
            xppu: ppm,
            yppu: ppm,
            unit: png::Unit::Meter,
        }));
    }
    let mut writer = encoder.write_header().map_err(|err| FrameError::Export {
        message: format!("PNG header write failed: {err}"),
    })?;
    // `eXIf` holds the raw TIFF blob (no `Exif\0\0` prefix — PNG uses the
    // chunk name to mark EXIF data), written before any IDAT as the spec asks.
    if let Some(tiff) = exif_tiff
        && writer.write_chunk(png::chunk::eXIf, tiff).is_err()
    {
        drop(writer);
        std::fs::remove_file(&tmp).ok();
        return Err(FrameError::Export {
            message: "PNG eXIf chunk write failed".to_owned(),
        });
    }
    if writer.write_image_data(&gray).is_err() {
        drop(writer);
        std::fs::remove_file(&tmp).ok();
        return Err(FrameError::Export {
            message: "PNG image data write failed".to_owned(),
        });
    }
    // `finish` writes the IEND trailer and consumes the writer; its dropped
    // BufWriter flushes any residual bytes, so the temp file is complete on
    // success and only then renamed into place.
    if writer.finish().is_err() {
        std::fs::remove_file(&tmp).ok();
        return Err(FrameError::Export {
            message: "PNG finish failed".to_owned(),
        });
    }
    std::fs::rename(&tmp, dest).map_err(|err| FrameError::Export {
        message: format!("failed to rename {} into place: {err}", tmp.display()),
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use edit_manifest::CropMargins as Marg;
    #[test]
    fn export_name_replaces_the_source_extension() {
        assert_eq!(
            export_name("img_0001.cr2", ExportFormat::Jpeg).to_string_lossy(),
            "img_0001.jpg"
        );
        assert_eq!(
            export_name("img_0001.cr2", ExportFormat::Png).to_string_lossy(),
            "img_0001.png"
        );
        assert_eq!(
            export_name("img_0001.nef", ExportFormat::Jpeg).to_string_lossy(),
            "img_0001.jpg"
        );
        // No extension in the name: the format's extension is simply appended.
        assert_eq!(
            export_name("IMG_0001", ExportFormat::Jpeg).to_string_lossy(),
            "IMG_0001.jpg"
        );
        // A dotfile: the hidden leading dot is part of the stem.
        assert_eq!(
            export_name(".hidden.cr2", ExportFormat::Jpeg).to_string_lossy(),
            ".hidden.jpg"
        );
    }

    #[test]
    fn roll_hash_is_deterministic_6_hex() {
        let a = std::path::Path::new("/home/user/Films/2024 May Costa Rica");
        let b = std::path::Path::new("/home/user/Films/2024 May Portugal");
        let (ha, ha2, hb) = (roll_hash(a), roll_hash(a), roll_hash(b));
        // Deterministic, exact width, lowercase hex regardless of path content.
        assert_eq!(ha, ha2);
        assert_eq!(ha.len(), 6);
        assert!(
            ha.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
        // Different rolls must (almost always) differ.
        assert_ne!(ha, hb);
    }

    #[test]
    fn dated_export_name_builds_the_scheme() {
        // `YYYYMMDD-<roll-hash>-<frame>.<ext>`; frame is the 1-based full-roll
        // position padded to two digits; the source stem is deliberately gone.
        let name = |index, hash| {
            dated_export_name(ExportFormat::Jpeg, "2024-05-09", index, hash)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(name(0, "1a2b3c"), "20240509-1a2b3c-01.jpg");
        assert_eq!(name(9, "1a2b3c"), "20240509-1a2b3c-10.jpg");
        assert_eq!(name(99, "1a2b3c"), "20240509-1a2b3c-100.jpg");
        assert_eq!(
            dated_export_name(ExportFormat::Png, "2024-05-09", 1, "1a2b3c")
                .unwrap()
                .to_string_lossy(),
            "20240509-1a2b3c-02.png"
        );
    }

    #[test]
    fn dated_names_group_by_roll_when_sorted() {
        // Hash-first ordering means a same-day multi-roll folder sorts into
        // roll groups, each roll's frames in capture order — the property that
        // put the hash before the frame number.
        let name = |hash: &str, index: usize| {
            dated_export_name(ExportFormat::Jpeg, "2024-05-09", index, hash)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        };
        let mut names = vec![
            name("bbbbbb", 1),
            name("aaaaaa", 1),
            name("aaaaaa", 0),
            name("bbbbbb", 0),
        ];
        names.sort();
        assert_eq!(
            names,
            vec![
                "20240509-aaaaaa-01.jpg",
                "20240509-aaaaaa-02.jpg",
                "20240509-bbbbbb-01.jpg",
                "20240509-bbbbbb-02.jpg",
            ]
        );
    }

    #[test]
    fn dated_export_name_rejects_a_malformed_start_date() {
        assert!(dated_export_name(ExportFormat::Jpeg, "2024-5-09", 0, "1a2b3c").is_none());
        assert!(dated_export_name(ExportFormat::Jpeg, "2024/05/09", 0, "1a2b3c").is_none());
        assert!(dated_export_name(ExportFormat::Jpeg, "", 0, "1a2b3c").is_none());
    }

    #[test]
    fn export_presets_define_the_expected_defaults() {
        // Cloud: JPEG q90 at native resolution, tagged 300 dpi for print.
        let cloud = ExportOptions::for_preset(ExportPreset::Cloud);
        assert_eq!(cloud.format, ExportFormat::Jpeg);
        assert_eq!(cloud.quality, 90);
        assert_eq!(cloud.size, ExportSize::Original);
        assert_eq!(cloud.ppi, 300);
        // Master: lossless 16-bit PNG at native resolution, tagged 300 dpi.
        let master = ExportOptions::for_preset(ExportPreset::Master);
        assert_eq!(master.format, ExportFormat::Png);
        assert_eq!(master.size, ExportSize::Original);
        assert_eq!(master.ppi, 300);
        // Web: JPEG q82 downscaled to a 2048px long edge, tagged 72 dpi.
        let web = ExportOptions::for_preset(ExportPreset::Web);
        assert_eq!(web.format, ExportFormat::Jpeg);
        assert_eq!(web.quality, 82);
        assert_eq!(web.size, ExportSize::LongEdge2048);
        assert_eq!(web.ppi, 72);
        // Overwriting is off by default (the dialog checkbox defaults to "no").
        assert!(!cloud.overwrite && !master.overwrite && !web.overwrite);
    }

    #[test]
    fn export_artifacts_are_identified_by_extension() {
        // Outputs of this app's exporter must never be mistaken for negatives.
        assert!(is_export_artifact("img_0001.jpg"));
        assert!(is_export_artifact("IMG_0002.JPEG"));
        assert!(is_export_artifact("scan.png"));
        assert!(!is_export_artifact("img_0001.cr2"));
        assert!(!is_export_artifact("IMG_0002.nef"));
        assert!(!is_export_artifact("scan.dng"));
        assert!(!is_export_artifact(".hidden"));
        assert!(!is_export_artifact("no_extension"));
    }

    #[test]
    fn options_for_choice_resolves_the_dropdown_keys() {
        // Each file-picker format choice key maps to its preset's options.
        let cloud = options_for_choice("jpeg-90");
        assert_eq!(cloud.format, ExportFormat::Jpeg);
        assert_eq!(cloud.quality, 90);
        let web = options_for_choice("jpeg-82");
        assert_eq!(web.quality, 82);
        assert_eq!(web.size, ExportSize::LongEdge2048);
        let master = options_for_choice("png-16");
        assert_eq!(master.format, ExportFormat::Png);
        // An unknown key (or a backend that dropped the choice) falls back to
        // the first preset.
        let fallback = options_for_choice("nonsense");
        assert_eq!(fallback.format, ExportFormat::Jpeg);
        assert_eq!(fallback.quality, 90);
    }

    #[test]
    fn png_export_writes_grayscale_16bit_round_trip() {
        // 2x2 solid mid-gray frame. 0.5 is exactly representable at 16-bit
        // (32768), so decoding the written PNG and checking that value pins
        // both the color type and the sample byte order: the raw `png` crate
        // writes big-endian samples as-is, and the old little-endian buffer
        // swapped every sample (0x8000 was stored and read back as 0x0080 = 128).
        let mono = vec![0.5, 0.5, 0.5, 0.5];
        let dest = std::env::temp_dir().join(format!(
            "curvectrl_png_smoke_{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        let result = export_png(
            mono,
            2,
            2,
            Marg::default(),
            0,
            &dest,
            None,
            ExportOptions {
                format: ExportFormat::Png,
                quality: 100,
                size: ExportSize::Original,
                ppi: 72,
                overwrite: false,
            },
        );
        let bytes = std::fs::read(&dest).unwrap();
        std::fs::remove_file(&dest).ok();
        assert_eq!(result, Ok(()));

        let image = image::load_from_memory(&bytes).unwrap();
        assert_eq!(image.color(), image::ColorType::L16);
        assert_eq!(
            image.as_luma16().unwrap().get_pixel(0, 0).0[0],
            32768,
            "mid-gray round-trips at 16-bit in native byte order"
        );

        // The 72 dpi request must produce a real pHYs chunk (2835 px/m). `read_info`
        // walks metadata up to the first IDAT (pHYs sits there); the bare
        // `read_header_info` stops at IHDR with the chunk fields still `None`.
        let decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
        let reader = decoder.read_info().expect("png header parses");
        let dims = reader
            .info()
            .pixel_dims
            .expect("a 72 dpi request tags a pHYs chunk");
        assert_eq!(dims.xppu, 2835);
        assert_eq!(dims.yppu, 2835);
        assert_eq!(dims.unit, png::Unit::Meter);

        // The export always tags the samples as sRGB, so color-managed
        // consumers interpret them deterministically rather than guessing.
        assert_eq!(
            reader.info().srgb,
            Some(png::SrgbRenderingIntent::Perceptual)
        );
    }

    #[test]
    fn jpeg_export_tags_the_jfif_density() {
        // 2x2 solid mid-gray through `export_jpeg` with the Web preset's 72 dpi.
        // The JFIF APP0 header should read: "JFIF\0" + version 1.2 + unit 01
        // (dots per inch) + Xdensity 72 (00 48 BE) + Ydensity 72.
        let mono = vec![0.5; 4];
        let dest = std::env::temp_dir().join(format!(
            "curvectrl_jpeg_dpi_{}.jpg",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        let result = export_jpeg(
            mono,
            2,
            2,
            Marg::default(),
            0,
            &dest,
            None,
            ExportOptions {
                format: ExportFormat::Jpeg,
                quality: 82,
                size: ExportSize::LongEdge2048,
                ppi: 72,
                overwrite: false,
            },
        );
        let bytes = std::fs::read(&dest).unwrap();
        std::fs::remove_file(&dest).ok();
        assert_eq!(result, Ok(()));

        let expected = [
            b'J', b'F', b'I', b'F', 0x00, 0x01, 0x02, 0x01, 0x00, 0x48, 0x00, 0x48,
        ];
        assert!(
            bytes.windows(expected.len()).any(|w| w == expected),
            "JFIF header must tag 72 dpi"
        );
    }

    #[test]
    fn jpeg_encoder_smoke_writes_quality_90_soi() {
        // Grayscale, matching the export path's Luma encode.
        let mut gray = Vec::with_capacity(16 * 16);
        for y in 0..16u8 {
            for x in 0..16u8 {
                gray.push(x * 16 ^ y * 16);
            }
        }

        // A unique temp path per run so parallel test threads never collide.
        let mut dest = std::env::temp_dir();
        dest.push(format!(
            "curvectrl_jpeg_smoke_{}.jpg",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));

        let encoder = jpeg_encoder::Encoder::new_file(&dest, 90).unwrap();
        encoder
            .encode(&gray, 16, 16, jpeg_encoder::ColorType::Luma)
            .unwrap();

        let bytes = std::fs::read(&dest).unwrap();
        std::fs::remove_file(&dest).ok();

        // A JPEG stream always starts with the SOI marker FF D8 FF.
        assert!(bytes.len() > 4, "encoded stream has payload");
        assert_eq!(&bytes[..3], &[0xFF, 0xD8, 0xFF]);
    }
}
