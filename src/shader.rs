// SPDX-License-Identifier: GPL-3.0-or-later

//! GPU detail shader — renders mono image data with live EV adjustment.
//!
//! The mono `Vec<f32>` is uploaded to the GPU once as an `R16Float` texture.
//! Exposure and the tone curve are applied as shader uniforms (`2^EV` gain and
//! a pivoted `ratio * p^exp` remap) — zero CPU re-encoding, zero new `Handle`
//! per frame.

use cosmic::iced::core::{Length, Rectangle};
use cosmic::iced::wgpu::util::DeviceExt;
use cosmic::iced::widget::shader::{Pipeline, Primitive, Program, Shader, Viewport};

use crate::edit_manifest::CropMargins;
use crate::film::{MonoStock, invert_value};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// GPU-backed detail view image with live exposure adjustment.
pub struct DetailProgram {
    mono: Vec<f32>,
    width: u32,
    height: u32,
    /// The full-resolution (post-masked-border, pre-downscale) display-oriented
    /// source dimensions the crop margins are authored against, restored from
    /// the downscaled texture + the sensor's true long edge at construction
    /// (both share the same aspect, so the scale per axis is recovered). The
    /// texture is a downscale of this; the shader maps source-pixel crop
    /// margins onto the texture by `tex/src` per axis. At the native level-up
    /// (texture == source) the scale collapses to 1 and the crop is exact.
    src_w: u32,
    src_h: u32,
    /// Raw user-facing EV value in stops; converted to a linear-light gain once
    /// on the GPU side per slider change — `2^EV` for an already-positive
    /// scan, `2^-EV` for an inverted (film) negative so +EV always brightens
    /// the final positive (see [`sensor_gain`]).
    exposure: f32,
    /// Detail-view zoom in `log2` units: 1.0 = contain fit, each +1 doubles
    /// the rendered scale (see [`DetailPrimitive::prepare`]).
    zoom: f32,
    /// Pan offset of the image center from the widget center, in logical
    /// points; converted to physical pixels on the GPU side.
    pan: cosmic::iced::Point,
    /// Median of the uploaded positive: the mid-gray the contrast curve
    /// pivots around. Measured once at construction from `mono`.
    mid: f32,
    /// 98th percentile of the uploaded positive: the white point the
    /// shadows curve pivots around.
    white: f32,
    /// 10th percentile of the uploaded positive: the shadow anchor the
    /// highlights curve pivots around. Measured once at construction from `mono`.
    shadow: f32,
    /// Contrast power `kc`: the live tone curve pivots at `mid`,
    /// `C(p) = mid^(1-kc) * p^kc`. `1.0` = identity (no contrast change).
    contrast: f32,
    /// Highlights power `kh`: the live tone curve pivots at `shadow`,
    /// `H(p) = shadow^(1-kh) * p^kh`. `1.0` = identity; raising it brightens the
    /// upper tones (a highlight lift, pinning the shadow anchor).
    highlights: f32,
    /// Shadows power `ks`: the live tone curve pivots at `white`,
    /// `S(p) = white^(1-ks) * p^ks`. `1.0` = identity; lowering it brightens the
    /// lower tones (a shadow lift, pinning the white point).
    shadows: f32,
    /// Live source-pixel crop margins removed from each edge of the
    /// full-resolution display-oriented frame. Applied as a uniform UV-remap
    /// (no texture re-upload): at render time [`crop_uv_geometry`] scales them
    /// onto the downscaled texture by `tex_axis/src_axis`, so the live trim and
    /// the CPU thumbnail bake (which crops the same source-pixel margins onto
    /// the print) stay in the same reference frame. A keyboard trim is instant.
    /// `CropMargins::default()` (all zero) shows the full frame.
    crop: CropMargins,
    /// User display rotation as cumulative counter-clockwise 90° quarter-turns
    /// (`0`…`3`), applied ON TOP of the EXIF-upright texture as a uniform-only
    /// UV remap (no texture re-upload). The crop sub-rectangle is authored in
    /// the EXIF-upright frame and is rotated along with the display, so the
    /// crop math is untouched. `0` = identity.
    rotation: u8,
    /// Whether the "view dimmed crop area" overlay is shown: the full
    /// uncropped frame drawn on top of the zoomed crop with everything outside
    /// the crop rectangle dimmed. View-only — never persisted or baked.
    show_mask: bool,
    /// Minimum padding (logical points) kept around the image in crop mode:
    /// the WGSL contain-fit base shrinks by `pad` on every side so a white
    /// margin is always visible; zooming in grows the image into it. `0.0`
    /// (normal mode) is the exact previous layout. Set via [`Self::set_pad`].
    pad: f32,
    /// Whether the mono texture holds a true sensor-linear NEGATIVE that the
    /// shader must density-invert per fragment (a film preset), instead of an
    /// already-positive scan. When inverted, the exposure gain multiplies the
    /// sensor data BEFORE the inversion and the EV sign flips (see
    /// [`sensor_gain`]), keeping the user-facing slider semantics identical
    /// across presets.
    inverted: bool,
    /// The inversion's clear-film transmission anchor (black point): the stock's
    /// calibrated `base` or a per-frame measurement from the sensor mono. Only
    /// meaningful when `inverted`; shader-uniform value.
    inv_base: f32,
    /// The stock's usable density range above the base (`MonoStock::d_max`),
    /// driving the inversion's white point. Shader-uniform value.
    inv_d_max: f32,
    /// The stock's density-space tone exponent (`MonoStock::gamma`). Only
    /// meaningful when `inverted`; shader-uniform value.
    inv_gamma: f32,
    /// The film stock profile inverting this texture, when the preset is an
    /// inverted film negative. Drives re-deriving the tone pivots analytically
    /// for the live EV (see [`film_pivots_at_gain`]); `None` for a positive
    /// scan.
    stock: Option<MonoStock>,
    /// Raw sensor-linear transmission fractiles (ranks {0.90, 0.50, 0.02}) the
    /// inverted preset's tone pivots are derived from at each EV. Because the
    /// density inversion is monotone-decreasing, the rendered positive's
    /// percentile `q` hangs off the sensor rank `1-q`; keeping the RANK in
    /// sensor space lets every frame's pivots be exact at the CURRENT exposure
    /// instead of frozen EV-0 values. `None` for a positive scan, whose pivots
    /// come straight from [`tone_anchors`] (EV-independent by construction).
    anchor_fractiles: Option<(f32, f32, f32)>,
    /// The GPU tone LUT as `R16Float` bytes, rebuilt whenever the pivots or curve
    /// powers change (see [`Self::rebuild_tone_lut`]).
    tone_lut: Vec<u8>,
    /// Monotonic id bumped on every tone-LUT rebuild; the pipeline re-uploads
    /// the texture when it changes.
    tone_version: u64,
    /// Monotonic id bumped by the app model on each new detail decode. Used
    /// to detect image changes and rebuild the GPU texture/bind group.
    image_id: u64,
}

impl DetailProgram {
    /// Create a new program for the given mono image.
    ///
    /// `mono` is TRUE sensor-linear data — linear `[0,1]` relative to the
    /// sensor white point — the same for every preset (one `f32` per pixel,
    /// row-major, top-to-bottom). `width`/`height` are the **texture** dims
    /// (the downscale of the source, display-oriented). `src_long_edge` is the
    /// sensor's true long edge AFTER masked-border cropping and BEFORE the
    /// downscale — it restores the full-resolution display-oriented source dims
    /// (`source_dimensions`) that the crop margins are authored against.
    /// `exposure` is the raw user-facing EV value; the gain sent to the GPU is
    /// `2^EV` for a non-inverted preset and `2^-EV` for an inverted one (see
    /// [`sensor_gain`]), so +EV always brightens the final positive.
    /// `crop` is the frame's stored source-pixel crop (all-zero for a fresh
    /// frame). `rotation` is the cumulative counter-clockwise 90° quarter-turn
    /// display rotation on top of the EXIF-upright texture (`0` = none). The
    /// view starts at contain fit (zoom 1.0, no pan) with an identity tone
    /// curve. `inversion` is `Some((stock, base))` when the texture is a film
    /// negative the shader must density-invert per fragment (`base` is the
    /// clear-film transmission anchor). An inverted preset's shadow/mid-gray/
    /// white-point pivots derive from the raw sensor fractiles at the CURRENT
    /// exposure (see [`film_pivots_at_gain`]); a positive scan's pivot from
    /// [`tone_anchors`].
    #[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
    pub fn new(
        mono: Vec<f32>,
        width: u32,
        height: u32,
        exposure: f32,
        crop: CropMargins,
        rotation: u8,
        image_id: u64,
        src_long_edge: u32,
        inversion: Option<(MonoStock, f32)>,
    ) -> Self {
        let (shadow, mid, white, stock, anchor_fractiles) = if let Some((stock, base)) = inversion {
            let fractiles = film_anchor_fractiles(&mono);
            let (shadow, mid, white) =
                film_pivots_at_gain(fractiles, base, sensor_gain(exposure, true), stock);
            (shadow, mid, white, Some(stock), Some(fractiles))
        } else {
            let (shadow, mid, white) = tone_anchors(&mono);
            (shadow, mid, white, None, None)
        };
        let (inverted, inv_base, inv_d_max, inv_gamma) = match inversion {
            Some((stock, base)) => (true, base, stock.d_max, stock.gamma),
            None => (false, 1.0, 1.0, 1.0),
        };
        // The texture shares the display source's aspect (resize_area +
        // orientation preserve it within floor-rounding), so the two source
        // axes are recovered from the long edge: the long-edged axis equals
        // `src_long_edge`, the short axis scales its texture counterpart.
        let (src_w, src_h) = if width >= height {
            (
                src_long_edge.max(1),
                (((u64::from(height) * u64::from(src_long_edge)) + u64::from(width) / 2)
                    / u64::from(width.max(1))) as u32,
            )
        } else {
            (
                (((u64::from(width) * u64::from(src_long_edge)) + u64::from(height) / 2)
                    / u64::from(height.max(1))) as u32,
                src_long_edge.max(1),
            )
        };
        Self {
            mono,
            width,
            height,
            src_w,
            src_h,
            exposure,
            zoom: 1.0,
            pan: cosmic::iced::Point::default(),
            mid,
            white,
            shadow,
            contrast: 1.0,
            highlights: 1.0,
            shadows: 1.0,
            crop,
            rotation,
            show_mask: false,
            pad: 0.0,
            inverted,
            inv_base,
            inv_d_max,
            inv_gamma,
            stock,
            anchor_fractiles,
            tone_lut: build_tone_lut(1.0, 1.0, 1.0, shadow, mid, white),
            tone_version: 0,
            image_id,
        }
    }

    /// Rebuild the GPU tone LUT from the current pivots + curve powers and bump
    /// its version, so the pipeline re-uploads the texture. Called whenever a
    /// slider moves (`set_exposure` re-derives film pivots, `set_curve` changes
    /// the powers). ~2048 powf ≈ µs — negligible on a drag tick.
    fn rebuild_tone_lut(&mut self) {
        self.tone_lut = build_tone_lut(
            self.contrast,
            self.highlights,
            self.shadows,
            self.shadow,
            self.mid,
            self.white,
        );
        self.tone_version = self.tone_version.wrapping_add(1);
    }

    /// Update the exposure value (called on slider drag).
    ///
    /// An inverted (film) preset re-derives its tone pivots from the raw
    /// sensor fractiles at the incoming EV, so the shadow/mid/white anchors
    /// always describe the ACTUAL render instead of the EV at decode. A
    /// positive scan's anchors are EV-independent (the gain never touches the
    /// curve input) and stay as measured.
    pub fn set_exposure(&mut self, ev: f32) {
        self.exposure = ev;
        if let (Some(stock), Some(fractiles)) = (self.stock, self.anchor_fractiles) {
            (self.shadow, self.mid, self.white) = film_pivots_at_gain(
                fractiles,
                self.inv_base,
                sensor_gain(ev, true),
                stock,
            );
        }
        self.rebuild_tone_lut();
    }

    /// Update the zoom/pan transform (called on detail-view wheel/drag).
    ///
    /// `zoom` is in `log2` units (1.0 = contain fit). `pan` is in logical
    /// points relative to the widget center.
    pub fn set_view(&mut self, zoom: f32, pan: cosmic::iced::Point) {
        self.zoom = zoom;
        self.pan = pan;
    }

    /// Update the contrast/highlights/shadows tone curve (called on editing
    /// drawer sliders). Only the two remap uniforms change — the uploaded
    /// texture stays the fixed render, re-curved per pixel in WGSL.
    ///
    /// `contrast` pivots at the image's measured mid-gray, `highlights` at the
    /// measured 10th-percentile shadow anchor, `shadows` at the measured white
    /// point (`1.0` = identity for each); all three compose into the single
    /// `ratio * p^exp` remap the shader applies. The Highlights/Shadows slider
    /// and keyboard layers present these powers through a stop-based "lift
    /// value" (`app::highlight_lift`/`app::shadow_lift`, 0 = identity at the
    /// track center, +n = n stops lifting) so increasing on screen brightens
    /// the region; this method is the unconverted raw-power boundary.
    pub fn set_curve(&mut self, contrast: f32, highlights: f32, shadows: f32) {
        self.contrast = contrast;
        self.highlights = highlights;
        self.shadows = shadows;
        self.rebuild_tone_lut();
    }

    /// Update the live crop margins (called on each keyboard trim).
    ///
    /// Only the four crop uniforms change — the uploaded texture stays intact.
    /// The WGSL zoom-to-fits the CROPPED region into the widget; the overlay
    /// full-frame layer (see [`Self::set_show_mask`]) dims outside the crop.
    pub fn set_crop(&mut self, crop: CropMargins) {
        self.crop = crop;
    }

    /// Update the live display rotation (called on each rotate press/click).
    ///
    /// Only the one uniform changes — the uploaded texture stays intact. The
    /// WGSL rotates the already-cropped display frame counter-clockwise by
    /// `rotation & 3` quarter-turns; `0` is the identity (EXIF-upright).
    pub fn set_rotation(&mut self, rotation: u8) {
        self.rotation = rotation;
    }

    /// Show or hide the "view dimmed crop area" overlay (called on toggle).
    ///
    /// When on, the shader draws the full uncropped frame on top of the zoomed
    /// crop (same zoom/pan) and dims everything outside the crop rectangle,
    /// letting the user see where the crop lands over the whole negative.
    /// View-only state: never persisted or applied to grid/cover bakes.
    pub fn set_show_mask(&mut self, on: bool) {
        self.show_mask = on;
    }

    /// Set the minimum padding (logical points) kept around the image in crop
    /// mode. The WGSL contain-fit base shrinks by `pad` on every side so a
    /// white margin is always visible; zooming in grows the image into it.
    /// `0.0` (normal mode) restores the exact unpadded layout.
    pub fn set_pad(&mut self, pad: f32) {
        self.pad = pad;
    }

    /// The full-resolution display-oriented source dimensions the persisted
    /// crop margins are authored against (the sensor's post-masked-border dims,
    /// oriented, before any downscale). The crop keyboard math must bound its
    /// trims against THIS frame — the downscaled texture dims would store a
    /// texture-pixel crop the thumbnail bake mis-scales.
    #[must_use]
    pub fn source_dimensions(&self) -> (u32, u32) {
        (self.src_w, self.src_h)
    }

    /// The zoom level (in `log2` units, `1.0` = contain fit) at which the image
    /// renders at 100% — one texture pixel per one physical screen pixel.
    ///
    /// Mirrors the WGSL's `contain = min(sc_w/dw, sc_h/dh)` with the live crop
    /// and display rotation applied, so the cap tracks the ACTUAL on-screen
    /// scale. `widget_w`/`widget_h` are the preview area's LOGICAL size and
    /// `scale_factor` converts them to physical pixels, matching the shader's
    /// `sc_w = bounds · scale_factor`. Floored at 1.0 (contain fit) so the cap
    /// never drops below the minimum zoom.
    #[must_use]
    pub fn zoom_100(&self, widget_w: f32, widget_h: f32, scale_factor: f32) -> f32 {
        self.zoom_100_for(self.width, self.height, widget_w, widget_h, scale_factor)
    }

    /// The same 1:1 ("100%") cap math as [`Self::zoom_100`], but for a
    /// hypothetical texture of `tex_w` × `tex_h` under the live crop, display
    /// rotation and pad. Lets the detail view project the cap a pending native
    /// (level-up) decode will unlock while only the coarse overview is
    /// installed, so zooming can continue through the load without ever
    /// passing the point the upcoming texture's pixels cover.
    #[must_use]
    pub fn zoom_100_for(
        &self,
        tex_w: u32,
        tex_h: u32,
        widget_w: f32,
        widget_h: f32,
        scale_factor: f32,
    ) -> f32 {
        let (_, (cw, ch)) = crop_uv_geometry(self.crop, tex_w, tex_h, self.src_w, self.src_h);
        let (dw, dh) = if (self.rotation & 1) != 0 { (ch, cw) } else { (cw, ch) };
        // The 1:1 cap mirrors the WGSL contain-fit, which in crop mode shrinks
        // the available box by the minimum padding on every side (`pad` is in
        // logical points, converted to physical like the scissor rect).
        let sc_w = ((widget_w - 2.0 * self.pad) * scale_factor).max(1.0);
        let sc_h = ((widget_h - 2.0 * self.pad) * scale_factor).max(1.0);
        1.0 + (dw / sc_w).max(dh / sc_h).log2().max(0.0)
    }

    /// Wrap in a `Shader` widget sized to fill the parent.
    pub fn view<M>(&self) -> Shader<M, Self> {
        Shader::new(self.clone())
            .width(Length::Fill)
            .height(Length::Fill)
    }
}

impl Clone for DetailProgram {
    fn clone(&self) -> Self {
        Self {
            mono: self.mono.clone(),
            width: self.width,
            height: self.height,
            src_w: self.src_w,
            src_h: self.src_h,
            exposure: self.exposure,
            zoom: self.zoom,
            pan: self.pan,
            mid: self.mid,
            white: self.white,
            shadow: self.shadow,
            contrast: self.contrast,
            highlights: self.highlights,
            shadows: self.shadows,
            crop: self.crop,
            rotation: self.rotation,
            show_mask: self.show_mask,
            pad: self.pad,
            inverted: self.inverted,
            inv_base: self.inv_base,
            inv_d_max: self.inv_d_max,
            inv_gamma: self.inv_gamma,
            stock: self.stock,
            anchor_fractiles: self.anchor_fractiles,
            tone_lut: self.tone_lut.clone(),
            tone_version: self.tone_version,
            image_id: self.image_id,
        }
    }
}

impl std::fmt::Debug for DetailProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetailProgram")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("src_w", &self.src_w)
            .field("src_h", &self.src_h)
            .field("exposure", &self.exposure)
            .field("zoom", &self.zoom)
            .field("pan", &self.pan)
            .field("mid", &self.mid)
            .field("white", &self.white)
            .field("shadow", &self.shadow)
            .field("contrast", &self.contrast)
            .field("highlights", &self.highlights)
            .field("shadows", &self.shadows)
            .field("crop", &self.crop)
            .field("rotation", &self.rotation)
            .field("show_mask", &self.show_mask)
            .field("pad", &self.pad)
            .field("inverted", &self.inverted)
            .field("inv_base", &self.inv_base)
            .field("inv_d_max", &self.inv_d_max)
            .field("inv_gamma", &self.inv_gamma)
            .field("stock", &self.stock)
            .field("anchor_fractiles", &self.anchor_fractiles)
            .field("tone_lut_len", &self.tone_lut.len())
            .field("tone_version", &self.tone_version)
            .field("mono_len", &self.mono.len())
            .field("image_id", &self.image_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// iced::widget::shader::Program implementation
// ---------------------------------------------------------------------------

impl<M> Program<M> for DetailProgram {
    type State = ();
    type Primitive = DetailPrimitive;

    fn draw(
        &self,
        _state: &(),
        _cursor: cosmic::iced::mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        DetailPrimitive {
            mono: self.mono.clone(),
            exposure: self.exposure,
            zoom: self.zoom,
            pan: self.pan,
            crop: self.crop,
            rotation: self.rotation,
            show_mask: self.show_mask,
            pad: self.pad,
            width: self.width,
            height: self.height,
            src_w: self.src_w,
            src_h: self.src_h,
            inverted: self.inverted,
            inv_base: self.inv_base,
            inv_d_max: self.inv_d_max,
            inv_gamma: self.inv_gamma,
            tone_lut: self.tone_lut.clone(),
            tone_version: self.tone_version,
            image_id: self.image_id,
        }
    }
}

// ---------------------------------------------------------------------------
// Crop-geometry + tone-curve pure helpers
// ---------------------------------------------------------------------------

/// The live-crop sub-rectangle in **texture space** as `(origin, size)`, both
/// `(x, y)` pairs, given the source-pixel crop margins, the downscaled texture
/// dims (`tex_w`/`tex_h`) and the full-resolution display-oriented source dims
/// (`src_w`/`src_h`) the margins are authored against. Each margin axis scales
/// by `tex_axis/src_axis`, so a stored source-pixel crop removes the same
/// *fraction of the frame* at any texture resolution (overview 2048 or native)
/// and matches what the CPU thumbnail bake does. An all-zero crop yields origin
/// `(0, 0)` and size equal to the texture — the identity no-op the WGSL remap
/// leaves untouched.
#[allow(clippy::cast_precision_loss)]
#[must_use]
fn crop_uv_geometry(
    crop: CropMargins,
    tex_w: u32,
    tex_h: u32,
    src_w: u32,
    src_h: u32,
) -> ((f32, f32), (f32, f32)) {
    let sx = tex_w as f32 / src_w.max(1) as f32;
    let sy = tex_h as f32 / src_h.max(1) as f32;
    (
        (crop.left as f32 * sx, crop.top as f32 * sy),
        (
            crop.cropped_width(src_w) as f32 * sx,
            crop.cropped_height(src_h) as f32 * sy,
        ),
    )
}

/// Bins per axis for the anchor histogram. 4096 bins over [0,1] resolve the
/// median and the 98th-percentile white point to ~2.4e-4 absolute — far finer
/// than the f16 texture's ~2^-11 and more than any preview needs.
const ANCHOR_BINS: usize = 4096;

/// Subsample stride for [`film_anchor_fractiles`]: every Nth sensor sample
/// lands in the fractile histogram. Percentile pivots are insensitive to the
/// subsample, and the stride bounds the per-decode cost at native resolution
/// (an 8K² buffer would otherwise histogram ~64M samples).
const ANCHOR_INV_STRIDE: usize = 4;

/// Smallest anchor accepted. Guards `pow(0, negative)` → NaN for degenerate
/// all-black frames, where the 50th/98th percentile of noise can land at 0.
pub(crate) const MIN_ANCHOR: f32 = 1e-3;

/// The shadow pivot is floored at this share of the mid-gray pivot so the
/// Shadows power stays usable when the frame's bottom decile is literal black
/// (the clear-film plateau at EV ≤ 0): a pivot pinned to [`MIN_ANCHOR`] makes
/// `s^(1-ks)` explode (`s≈0` lifts the whole positive to white), while a
/// pivot at ~15% of the mid keeps the control acting on the darkest real
/// detail. Tunable during visual calibration.
const SHADOW_PIVOT_FLOOR: f32 = 0.15;

/// The linear-light sensor gain for a user-facing EV value: for an
/// already-positive preset the gain is `2^EV` (multiplies the final positive),
/// for an INVERTED (film) preset the sign flips — `2^-EV` multiplies the true
/// sensor-linear NEGATIVE before the density inversion, so +EV makes the film
/// denser and the positive brighter. Identical slider semantics (+EV =
/// brighter) across presets; the flip happens at the point EV meets the sensor
/// data.
///
/// Kept pure + unit-tested so the GPU shader, the CPU thumbnail bake, and the
/// export path cannot drift.
#[must_use]
pub fn sensor_gain(ev: f32, inverted: bool) -> f32 {
    if inverted {
        (-ev).exp2()
    } else {
        ev.exp2()
    }
}

/// Measure the tone anchors a pivoted curve needs, from the uploaded positive
/// (`p` in [0,1]): the 10th percentile (the shadow anchor, robust against a
/// few pure-black pixels), the median (a stable mid-gray), and the 98th
/// percentile (a robust white point, insensitive to a few hot specular
/// pixels). Single pass over the mono, O(n) — no sort of a 2048² buffer.
///
/// `pub(crate)` so the CPU thumbnail bake reuses the same anchor machinery the
/// detail shader does, keeping grid and detail measurements aligned.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn tone_anchors(mono: &[f32]) -> (f32, f32, f32) {
    anchors_from(mono.iter().copied(), mono.len())
}

/// The raw sensor-linear fractiles a film inversion derives its tone pivots
/// from: the transmission values at ranks {0.90, 0.50, 0.02}. Because the
/// density inversion is monotone-decreasing, the rendered positive's
/// percentile `q` hangs off the sensor rank `1−q` — the 10th-percentile shadow
/// from the 90th, the mid-gray from the median, and the 98th-percentile white
/// point from the 2nd (the densest real data, insensitive to a few saturated
/// or dusty samples at the high end).
///
/// Keeping the RANK in raw sensor space (instead of density-inverting first)
/// lets the EV-exact pivot for any exposure be derived analytically with
/// `invert_value(fractile · gain)` — the exact transform the WGSL applies per
/// fragment — instead of freezing the EV-0 anchors while the frame renders at
/// +0.7 EV. Samples every [`ANCHOR_INV_STRIDE`]-th pixel to bound the cost on
/// a native level-up decode.
pub(crate) fn film_anchor_fractiles(mono: &[f32]) -> (f32, f32, f32) {
    let count = mono.len().div_ceil(ANCHOR_INV_STRIDE);
    let (bins, total) = histogram_from(mono.iter().step_by(ANCHOR_INV_STRIDE).copied(), count);
    if total == 0 {
        return (MIN_ANCHOR, MIN_ANCHOR, MIN_ANCHOR);
    }
    let white = percentile(&bins, total, 0.02);
    let mid = percentile(&bins, total, 0.5);
    let shadow = percentile(&bins, total, 0.90);
    (shadow, mid, white)
}

/// Derive the tone pivots (shadow/mid-gray/white) for a film negative rendered
/// at linear gain `g` from its raw sensor fractiles: the sensor value at rank
/// `r` maps through `invert_value(fractile · g)` — the exact transform the
/// WGSL applies per fragment. Each pivot is floored at [`MIN_ANCHOR`] like
/// [`tone_anchors`] guards its degenerate all-black frames.
///
/// `pub(crate)` so the CPU bake tail (`app::bake_tone`) derives the same
/// EV-exact pivots the detail shader does, keeping grid == detail == export.
pub(crate) fn film_pivots_at_gain(
    fractiles: (f32, f32, f32),
    base: f32,
    gain: f32,
    stock: MonoStock,
) -> (f32, f32, f32) {
    let (fs, fm, fw) = fractiles;
    (
        invert_value(fs * gain, base, &stock).max(MIN_ANCHOR),
        invert_value(fm * gain, base, &stock).max(MIN_ANCHOR),
        invert_value(fw * gain, base, &stock).max(MIN_ANCHOR),
    )
}

/// Bin values into the anchor histogram, returning `(bins, total)` where
/// `total` is the sample count supplied by the caller (the percentile targets
/// scale against it). The histogram-bin core shared by [`tone_anchors`] and
/// [`film_anchor_fractiles`].
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn histogram_from(values: impl Iterator<Item = f32>, count: usize) -> (Vec<u64>, usize) {
    // Heap-allocated (4096 × u64 ≈ 32 KB would trip `large_stack_arrays`).
    let mut bins = vec![0_u64; ANCHOR_BINS];
    for v in values {
        let v = v.clamp(0.0, 1.0);
        // `(v * (BINS-1))` maps 0..1 to bin 0..BINS-1; truncation is fine
        // because binning is deliberately approximate.
        let idx = (v * (ANCHOR_BINS - 1) as f32) as usize;
        bins[idx.min(ANCHOR_BINS - 1)] += 1;
    }
    (bins, count)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn anchors_from(values: impl Iterator<Item = f32>, count: usize) -> (f32, f32, f32) {
    let (bins, total) = histogram_from(values, count);
    if count == 0 {
        return (MIN_ANCHOR, MIN_ANCHOR, MIN_ANCHOR);
    }
    let shadow = percentile(&bins, total, 0.10);
    let mid = percentile(&bins, total, 0.5);
    let white = percentile(&bins, total, 0.98);
    (
        shadow.max(MIN_ANCHOR),
        mid.max(MIN_ANCHOR),
        white.max(MIN_ANCHOR),
    )
}

/// The value of the `q`-quantile (0..=1) of the bin counts, as a coordinate
/// in [0,1]. Walks the cumulative distribution until it reaches the
/// `q·total`-th element; for a non-trivial `total` this lands just past the
/// low-population boundary bins that a strict `>` test would skip.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn percentile(bins: &[u64], total: usize, q: f32) -> f32 {
    let target = (total as f32 * q) as u64;
    let mut cumulative = 0_u64;
    for (i, &count) in bins.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return i as f32 / (ANCHOR_BINS - 1) as f32;
        }
    }
    1.0
}

/// Precompute the per-frame tone remap `T(p) = clamp(ratio * p^exp, 0, 1)`.
///
/// Contrast, highlights, and shadows are power curves pivoted at the image's
/// measured mid-gray (`mid`), shadow anchor (`shadow`), and white point
/// (`white`):
///
/// `C(p) = mid^(1-kc) · p^kc`          (pivot at `mid`: `C(mid) = mid`)
/// `H(p) = shadow^(1-kh) · p^kh`        (pivot at `shadow`: `H(shadow) =
///                                                    shadow`)
/// `S(p) = white^(1-ks) · p^ks`         (pivot at `white`: `S(white) = white`)
///
/// Three monotone powers compose exactly into one power, so all three fit a
/// single `(ratio, exp)` pair the WGSL shader applies as `ratio·p^exp`:
/// `H∘S∘C(p) = (shadow^(1-kh) · white^(kh(1-ks)) · mid^(kh·ks(1-kc))) · p^(kc·ks·kh)`.
/// At `kc = kh = ks = 1` the remap is the identity (`ratio = 1`, `exp = 1`),
/// so untouched renders stay byte-identical.
///
/// `pub(crate)` — the shared helper the CPU thumbnail bake reuses so grid and
/// detail agree, alongside the GPU `prepare()` fold.
#[allow(clippy::too_many_arguments)]
pub(crate) fn curve_remap(
    contrast: f32,
    highlights: f32,
    shadows: f32,
    shadow: f32,
    mid: f32,
    white: f32,
) -> (f32, f32) {
    // A shadow pivot pinned to literal black (the clear-film plateau at EV ≤ 0)
    // would make the highlights power explode: `s^(1-kh)` near `s ≈ 0` lifts the
    // whole positive to white. Floor the pivot to a share of the mid so the
    // control keeps acting on the darkest detail; at the identity defaults
    // (`kh = 1`) every pivot power vanishes and the remap stays exactly 1.
    let shadow = shadow.max(mid * SHADOW_PIVOT_FLOOR);
    let ratio = shadow.powf(1.0 - highlights)
        * white.powf(highlights * (1.0 - shadows))
        * mid.powf(highlights * shadows * (1.0 - contrast));
    let exponent = contrast * highlights * shadows;
    (ratio, exponent)
}

/// Apply the composed tone remap `T(p) = clamp(ratio · p^exp, 0, 1)` to a mono
/// positive in place.
///
/// The CPU twin of the tone model the GPU consumes through its tone LUT (see
/// [`tone_model`] + [`build_tone_lut`]) — the shared math the grid thumbnail
/// and export bakes apply exactly, so a baked tile and the detail shader
/// produce identical tones within the tested LUT tolerance. Identity at the
/// `(1.0, 1.0, 1.0)` defaults; `shadow`/`mid`/`white` are the same anchors
/// [`tone_anchors`] measures. A separate op from exposure (which the shader
/// applies after), so callers must apply this *before* the `2^EV` gain to
/// mirror the shader.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_curve(
    mono: &mut [f32],
    contrast: f32,
    highlights: f32,
    shadows: f32,
    shadow: f32,
    mid: f32,
    white: f32,
) {
    let (ratio, exponent) = curve_remap(contrast, highlights, shadows, shadow, mid, white);
    for value in mono.iter_mut() {
        *value = tone_model(*value, ratio, exponent);
    }
}

/// The single per-pixel tone expression `clamp(ratio · p^exp, 0, 1)` — the
/// only place the tone-remap math lives for the non-shader paths. The GPU
/// texture-samples it from the tone LUT ([`build_tone_lut`]) instead of
/// re-deriving the expression per fragment, so the tone model can change on
/// the CPU without the WGSL ever changing.
#[must_use]
pub(crate) fn tone_model(p: f32, ratio: f32, exponent: f32) -> f32 {
    (ratio * p.powf(exponent)).clamp(0.0, 1.0)
}

/// Number of entries in the GPU tone LUT. 2048 on a smooth monotone curve is
/// far below a u8 level after half-float quantization, so the interpolation is
/// visually lossless while keeping the LUT a ~4 KB texture.
pub(crate) const TONE_LUT_ENTRIES: usize = 2048;

/// The LUT's sampling domain exponent: the LUT is built over `t ∈ [0,1]` where
/// the curve's input is `p = t^G`. Power tone curves with `exp < 1` (a shadow
/// lift, e.g. `p^0.3`) have an infinite slope at `p = 0`, so a uniform grid in
/// `p` badly undershoots the darkest entries. Sampling on `t = p^(1/G)` packs
/// the grid toward black, where the curve becomes near-linear in `t` and the
/// interpolation error collapses. Must match the WGSL's `pow(p, 1/G)`.
pub(crate) const TONE_LUT_GAMMA: f32 = 2.2;

/// Pre-scale factor applied to the LUT's stored values (the shader divides it
/// back out after sampling). The LUT stores the PRE-exposure curve output,
/// which for a darkening curve can sit far below the half-float normal floor
/// (6.1e-5) and would flush to zero — while the CPU bake preserves it in f32
/// and the later `2^EV` gain amplifies it. Scaling by 512 shifts the floor to
/// ~1.2e-7, deep below anything the ±4 EV slider can recover. Scale commutes
/// with linear interpolation, so the sampled value is exact after the divide.
pub(crate) const TONE_LUT_SCALE: f32 = 512.0;

/// Build the GPU tone LUT: [`TONE_LUT_ENTRIES`] samples of [`tone_model`] over
/// `p = t^G ∈ [0,1]` (see [`TONE_LUT_GAMMA`]), scaled by [`TONE_LUT_SCALE`] and
/// returned as `R16Float` (half) bytes ready for `write_texture`. The WGSL
/// samples this with linear interpolation at `t = p^(1/G)` and divides out the
/// scale; the CPU bakes call [`tone_model`] directly (exact), so the only
/// approximation anywhere is the LUT itself — bounded and unit-tested.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn build_tone_lut(
    contrast: f32,
    highlights: f32,
    shadows: f32,
    shadow: f32,
    mid: f32,
    white: f32,
) -> Vec<u8> {
    let (ratio, exponent) = curve_remap(contrast, highlights, shadows, shadow, mid, white);
    let mut lut = Vec::with_capacity(TONE_LUT_ENTRIES * 2);
    for i in 0..TONE_LUT_ENTRIES {
        let t = i as f32 / (TONE_LUT_ENTRIES - 1) as f32;
        let p = t.powf(TONE_LUT_GAMMA);
        lut.extend_from_slice(
            &f32_to_half(tone_model(p, ratio, exponent) * TONE_LUT_SCALE).to_le_bytes(),
        );
    }
    lut
}

/// Sample a decoded tone LUT with the same linear interpolation a GPU sampler
/// applies, mapping the curve input through the gamma domain exactly as the
/// WGSL does (`t = p^(1/G)`) and dividing out [`TONE_LUT_SCALE`]. Used by the
/// parity tests to simulate the GPU fragment math from the uploaded half LUT.
#[cfg(test)]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn sample_tone_lut_f32(lut: &[f32], p: f32) -> f32 {
    let t = p.clamp(0.0, 1.0).powf(1.0 / TONE_LUT_GAMMA);
    let pos = t * (TONE_LUT_ENTRIES - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(TONE_LUT_ENTRIES - 1);
    let frac = pos - lo as f32;
    (lut[lo] * (1.0 - frac) + lut[hi] * frac) / TONE_LUT_SCALE
}

// ---------------------------------------------------------------------------
// iced::wgpu::primitive::Primitive implementation
// ---------------------------------------------------------------------------

/// Per-frame data sent from `Program::draw()` to the GPU pipeline.
///
/// Carries a clone of the mono data on the first frame so the pipeline can
/// create the GPU texture.  `image_id` lets the pipeline detect when a new
/// image is selected (iceD caches pipelines per type, so we cannot rely on
/// `initialized` alone).
#[derive(Debug, Clone)]
pub struct DetailPrimitive {
    mono: Vec<f32>,
    exposure: f32,
    /// Detail-view zoom in `log2` units; 1.0 = contain fit.
    zoom: f32,
    /// Pan offset of the image center from the widget center (logical points).
    pan: cosmic::iced::Point,
    /// Live source-pixel crop margins removed from each edge (uniform UV-remap).
    crop: CropMargins,
    /// User display rotation (cumulative CCW 90° quarter-turns, `0`…`3`),
    /// applied as a uniform UV remap on top of the cropped EXIF-upright frame.
    rotation: u8,
    /// Whether the "view dimmed crop area" overlay is shown (full frame on top
    /// of the zoomed crop, dimmed outside the crop rect).
    show_mask: bool,
    /// Minimum padding (logical points) kept around the image in crop mode.
    pad: f32,
    /// Whether the mono is a true sensor-linear NEGATIVE to density-invert per
    /// fragment (film preset). Inverts the exposure gain sign (see
    /// [`sensor_gain`]) and drives the WGSL inversion branch.
    inverted: bool,
    /// Clear-film transmission anchor (inversion black point), shader uniform.
    inv_base: f32,
    /// Usable density range above the base (stock `d_max`), shader uniform.
    inv_d_max: f32,
    /// Density-space tone exponent (stock `gamma`), shader uniform.
    inv_gamma: f32,
    width: u32,
    height: u32,
    /// Full-resolution display-oriented source dims the crop margins are
    /// authored against (see [`crop_uv_geometry`]).
    src_w: u32,
    src_h: u32,
    /// The GPU tone LUT (`R16Float` bytes) and its version, so `prepare`
    /// re-uploads the texture when the curve/pivots change.
    tone_lut: Vec<u8>,
    tone_version: u64,
    image_id: u64,
}

impl Primitive for DetailPrimitive {
    type Pipeline = DetailPipeline;

    // Sequential wgpu uploads (texture, tone LUT, then uniforms) that must run
    // in this exact order each frame; splitting them into helper methods would
    // only scatter the pipeline state they all touch.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        device: &cosmic::iced::wgpu::Device,
        queue: &cosmic::iced::wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        // --- Rebuild texture + bind group if image identity changed ---
        // `iced` caches the `DetailPipeline` across image selections because
        // both selections share the same pipeline type. `initialized` only
        // catches the very first frame; tracking `image_id` catches every
        // subsequent image swap too.
        let needs_new_texture =
            !pipeline.initialized || pipeline.current_image_id != Some(self.image_id);

        if needs_new_texture {
            // Pre-filter the linear mono into a full mip chain so minified
            // views (zoom back out after the native level-up) sample averaged
            // texels instead of aliasing the full-res grain into a moiré. The
            // averaging runs in linear light on the TRUE sensor data, and the
            // WGSL's per-fragment tone/inversion applies after sampling — the
            // same flatten-before-average ordering the CPU thumbnail bake uses.
            let mips = build_mip_chain(&self.mono, self.width, self.height);
            let tex = device.create_texture(&cosmic::iced::wgpu::TextureDescriptor {
                label: Some("exposure mono"),
                size: cosmic::iced::wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mips.len() as u32,
                sample_count: 1,
                dimension: cosmic::iced::wgpu::TextureDimension::D2,
                format: cosmic::iced::wgpu::TextureFormat::R16Float,
                usage: cosmic::iced::wgpu::TextureUsages::TEXTURE_BINDING
                    | cosmic::iced::wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            for (level, mip) in mips.iter().enumerate() {
                queue.write_texture(
                    cosmic::iced::wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level as u32,
                        origin: cosmic::iced::wgpu::Origin3d::ZERO,
                        aspect: cosmic::iced::wgpu::TextureAspect::All,
                    },
                    &mip.data,
                    cosmic::iced::wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(mip.width * 2),
                        rows_per_image: Some(mip.height),
                    },
                    cosmic::iced::wgpu::Extent3d {
                        width: mip.width,
                        height: mip.height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            let texture_view =
                tex.create_view(&cosmic::iced::wgpu::TextureViewDescriptor::default());

            pipeline.texture = Some(texture_view);
            pipeline.bind_group = Some(device.create_bind_group(
                &cosmic::iced::wgpu::BindGroupDescriptor {
                    label: Some("exposure bg"),
                    layout: &pipeline.bind_group_layout,
                    entries: &[
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 0,
                            resource: cosmic::iced::wgpu::BindingResource::TextureView(
                                pipeline.texture.as_ref().unwrap(),
                            ),
                        },
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 1,
                            resource: cosmic::iced::wgpu::BindingResource::Sampler(
                                &pipeline.sampler,
                            ),
                        },
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 2,
                            resource: pipeline.uniform_buf.as_entire_binding(),
                        },
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 3,
                            resource: cosmic::iced::wgpu::BindingResource::TextureView(
                                &pipeline.tone_lut_view,
                            ),
                        },
                        cosmic::iced::wgpu::BindGroupEntry {
                            binding: 4,
                            resource: cosmic::iced::wgpu::BindingResource::Sampler(
                                &pipeline.sampler,
                            ),
                        },
                    ],
                },
            ));
            pipeline.current_image_id = Some(self.image_id);
            pipeline.initialized = true;
        }

        // --- Upload the tone LUT when the curve/pivots changed ---
        // A fixed 2048×1 R16Float texture created once in `Pipeline::new`; a
        // curve slider (`set_curve`) or EV drag on an inverted preset
        // (`set_exposure` re-derives pivots) bumps `tone_version`, and this
        // re-uploads the ~4 KB LUT. The WGSL texture-samples it per fragment.
        // `tone_version` resets per program (each image install re-derives it),
        // but the pipeline is cached across image selections — so also force a
        // re-upload whenever a new image is installed, or the second frame on
        // would keep sampling the previous frame's LUT until the user edits.
        if needs_new_texture || pipeline.current_tone_version != Some(self.tone_version) {
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let lut_w = TONE_LUT_ENTRIES as u32;
            queue.write_texture(
                cosmic::iced::wgpu::TexelCopyTextureInfo {
                    texture: &pipeline.tone_lut_tex,
                    mip_level: 0,
                    origin: cosmic::iced::wgpu::Origin3d::ZERO,
                    aspect: cosmic::iced::wgpu::TextureAspect::All,
                },
                &self.tone_lut,
                cosmic::iced::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(lut_w * 2),
                    rows_per_image: Some(1),
                },
                cosmic::iced::wgpu::Extent3d {
                    width: lut_w,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            pipeline.current_tone_version = Some(self.tone_version);
        }

        // --- Per-frame: update uniform buffer ---
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let sf = viewport.scale_factor() as f32;
        #[allow(clippy::cast_precision_loss)]
        let tex_w = self.width as f32;
        #[allow(clippy::cast_precision_loss)]
        let tex_h = self.height as f32;
        // Live crop: the texture sub-rectangle to show. Origin is the removed
        // left/top margins and size the kept region, mapped from the stored
        // source-pixel crop onto the DOWNSCALED texture by `tex/src` per axis
        // (see [`crop_uv_geometry`]); at the native level-up the texture IS the
        // source and the map is 1:1. The WGSL contain-fits the crop into the
        // widget (base view) and, when the dim overlay is on, also lays out the
        // full texture to show the crop in context. An all-zero crop → origin 0
        // and size == texture.
        let (crop_origin, (crop_w, crop_h)) =
            crop_uv_geometry(self.crop, self.width, self.height, self.src_w, self.src_h);
        let (crop_l, crop_t) = crop_origin;
        // Display rotation as a float quarter-turn count for the WGSL remap.
        // `0.0` = identity; the WGSL keeps it in {0,1,2,3} by construction.
        let rot = f32::from(self.rotation & 3);
        let uniforms = Uniforms {
            // Convert raw EV (slider value) to the linear-light sensor gain
            // once per frame; for an inverted (film) preset the sign flips so
            // the gain multiplies the sensor-linear NEGATIVE before the WGSL
            // inversion (see `sensor_gain`). The WGSL shader reads this as a
            // direct multiplier.
            exposure: sensor_gain(self.exposure, self.inverted),
            // Texture dimensions feed the WGSL's contained-fit math so the
            // shader mirrors `widget::image.content_fit(ContentFit::Contain)`.
            tex_w,
            tex_h,
            sc_x: bounds.x * sf,
            sc_y: bounds.y * sf,
            sc_w: bounds.width * sf,
            sc_h: bounds.height * sf,
            // Detail-view transform: zoom in log2 units (1.0 = contain fit);
            // pan in logical points converted to physical pixels, same
            // convention as the scissor rect above.
            zoom: self.zoom,
            pan_x: self.pan.x * sf,
            pan_y: self.pan.y * sf,
            crop_l,
            crop_t,
            crop_w,
            crop_h,
            rot,
            // Whether to draw the full-frame dim overlay (see set_show_mask).
            show_mask: if self.show_mask { 1.0 } else { 0.0 },
            // Crop-mode minimum padding, logical points → physical pixels like
            // the pan (see set_pad).
            pad: self.pad * sf,
            // Film-negative inversion: on when the texture is a true
            // sensor-linear negative the WGSL must density-invert per fragment.
            inv: if self.inverted { 1.0 } else { 0.0 },
            inv_base: self.inv_base,
            inv_d_max: self.inv_d_max,
            inv_gamma: self.inv_gamma,
        };
        queue.write_buffer(&pipeline.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut cosmic::iced::wgpu::CommandEncoder,
        target: &cosmic::iced::wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let (_tex_view, Some(pipeline_obj), Some(bind_group)) = (
            &pipeline.texture,
            &pipeline.render_pipeline,
            &pipeline.bind_group,
        ) else {
            return;
        };

        {
            let mut pass = encoder.begin_render_pass(&cosmic::iced::wgpu::RenderPassDescriptor {
                label: Some("exposure render"),
                color_attachments: &[Some(cosmic::iced::wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: cosmic::iced::wgpu::Operations {
                        load: cosmic::iced::wgpu::LoadOp::Load,
                        store: cosmic::iced::wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            #[allow(clippy::cast_precision_loss)]
            pass.set_viewport(
                clip_bounds.x as f32,
                clip_bounds.y as f32,
                clip_bounds.width as f32,
                clip_bounds.height as f32,
                0.0,
                1.0,
            );
            pass.set_pipeline(pipeline_obj);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU pipeline (created once per image, stored in iced's primitive storage)
// ---------------------------------------------------------------------------

/// GPU resources shared across all `DetailPrimitive` instances of the same
/// image.  Created lazily on the first `prepare()` call (needs the mono data
/// to build the texture).
pub struct DetailPipeline {
    texture: Option<cosmic::iced::wgpu::TextureView>,
    uniform_buf: cosmic::iced::wgpu::Buffer,
    sampler: cosmic::iced::wgpu::Sampler,
    bind_group_layout: cosmic::iced::wgpu::BindGroupLayout,
    bind_group: Option<cosmic::iced::wgpu::BindGroup>,
    render_pipeline: Option<cosmic::iced::wgpu::RenderPipeline>,
    /// The fixed 2048×1 `R16Float` tone LUT texture (created once here) + its
    /// view; `prepare()` re-uploads the bytes when `tone_version` changes.
    tone_lut_tex: cosmic::iced::wgpu::Texture,
    tone_lut_view: cosmic::iced::wgpu::TextureView,
    /// Version of the tone LUT currently uploaded; compared against the
    /// primitive's `tone_version` every frame to detect curve/pivot edits.
    current_tone_version: Option<u64>,
    initialized: bool,
    /// Identity of the image currently installed in the GPU texture. Compared
    /// against the primitive's `image_id` on every `prepare()` call to detect
    /// that the user has selected a different photo.
    current_image_id: Option<u64>,
}

impl std::fmt::Debug for DetailPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetailPipeline")
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl Pipeline for DetailPipeline {
    fn new(
        device: &cosmic::iced::wgpu::Device,
        _queue: &cosmic::iced::wgpu::Queue,
        _format: cosmic::iced::wgpu::TextureFormat,
    ) -> Self {
        let shader = load_shader(device);
        let bind_group_layout = build_bind_group_layout(device);
        let render_pipeline = build_render_pipeline(device, &shader, &bind_group_layout);

        let uniform_buf =
            device.create_buffer_init(&cosmic::iced::wgpu::util::BufferInitDescriptor {
                label: Some("exposure uniforms"),
                contents: bytemuck::bytes_of(&Uniforms::default()),
                usage: cosmic::iced::wgpu::BufferUsages::UNIFORM
                    | cosmic::iced::wgpu::BufferUsages::COPY_DST,
            });

        let sampler = device.create_sampler(&cosmic::iced::wgpu::SamplerDescriptor {
            address_mode_u: cosmic::iced::wgpu::AddressMode::ClampToEdge,
            address_mode_v: cosmic::iced::wgpu::AddressMode::ClampToEdge,
            mag_filter: cosmic::iced::wgpu::FilterMode::Linear,
            min_filter: cosmic::iced::wgpu::FilterMode::Linear,
            // Trilinear minification: the detail texture carries a full mip
            // chain (see [`build_mip_chain`]), so zooming back out past the
            // native level-up minifies through pre-filtered levels instead of
            // aliasing the full-res grain into a moiré pattern. A no-op for the
            // single-level tone LUT, which shares this sampler.
            mipmap_filter: cosmic::iced::wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // The tone LUT texture lives for the pipeline's lifetime: a fixed
        // 2048×1 R16Float strip re-uploaded by `prepare()` when the curve or
        // pivots change (the mono texture + bind group are still created
        // lazily below on the first frame, since they need the mono data).
        let tone_lut_tex = device.create_texture(&cosmic::iced::wgpu::TextureDescriptor {
            label: Some("exposure tone lut"),
            size: cosmic::iced::wgpu::Extent3d {
                #[allow(clippy::cast_possible_truncation)]
                width: TONE_LUT_ENTRIES as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: cosmic::iced::wgpu::TextureDimension::D2,
            format: cosmic::iced::wgpu::TextureFormat::R16Float,
            usage: cosmic::iced::wgpu::TextureUsages::TEXTURE_BINDING
                | cosmic::iced::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tone_lut_view =
            tone_lut_tex.create_view(&cosmic::iced::wgpu::TextureViewDescriptor::default());

        // Bind group and texture are created lazily in `prepare()` on the
        // first frame (they require the mono data to build the texture).

        Self {
            texture: None,
            uniform_buf,
            sampler,
            bind_group_layout,
            bind_group: None,
            render_pipeline: Some(render_pipeline),
            tone_lut_tex,
            tone_lut_view,
            current_tone_version: None,
            initialized: false,
            current_image_id: None,
        }
    }

    fn trim(&mut self) {
        // Nothing to trim — the texture lives for the image's lifetime.
    }
}

/// Load the WGSL source from disk and create a shader module.
fn load_shader(device: &cosmic::iced::wgpu::Device) -> cosmic::iced::wgpu::ShaderModule {
    device.create_shader_module(cosmic::iced::wgpu::ShaderModuleDescriptor {
        label: Some("exposure shader"),
        source: cosmic::iced::wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "shader/exposure.wgsl"
        ))),
    })
}

fn build_bind_group_layout(
    device: &cosmic::iced::wgpu::Device,
) -> cosmic::iced::wgpu::BindGroupLayout {
    device.create_bind_group_layout(&cosmic::iced::wgpu::BindGroupLayoutDescriptor {
        label: Some("exposure bgl"),
        entries: &[
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Texture {
                    sample_type: cosmic::iced::wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: cosmic::iced::wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Sampler(
                    cosmic::iced::wgpu::SamplerBindingType::Filtering,
                ),
                count: None,
            },
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: cosmic::iced::wgpu::ShaderStages::VERTEX
                    | cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Buffer {
                    ty: cosmic::iced::wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // The 2048×1 R16Float tone LUT + its (shared) linear sampler.
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Texture {
                    sample_type: cosmic::iced::wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: cosmic::iced::wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            cosmic::iced::wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: cosmic::iced::wgpu::ShaderStages::FRAGMENT,
                ty: cosmic::iced::wgpu::BindingType::Sampler(
                    cosmic::iced::wgpu::SamplerBindingType::Filtering,
                ),
                count: None,
            },
        ],
    })
}

fn build_render_pipeline(
    device: &cosmic::iced::wgpu::Device,
    shader: &cosmic::iced::wgpu::ShaderModule,
    bind_group_layout: &cosmic::iced::wgpu::BindGroupLayout,
) -> cosmic::iced::wgpu::RenderPipeline {
    let pipeline_layout =
        device.create_pipeline_layout(&cosmic::iced::wgpu::PipelineLayoutDescriptor {
            label: Some("exposure pl"),
            bind_group_layouts: &[bind_group_layout],
            immediate_size: 0,
        });

    device.create_render_pipeline(&cosmic::iced::wgpu::RenderPipelineDescriptor {
        label: Some("exposure rp"),
        layout: Some(&pipeline_layout),
        vertex: cosmic::iced::wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: cosmic::iced::wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(cosmic::iced::wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(cosmic::iced::wgpu::ColorTargetState {
                format: cosmic::iced::wgpu::TextureFormat::Bgra8Unorm,
                // Alpha-blend so the WGSL's `alpha=0` letterbox bars let the
                // COSMIC panel background show through. Image pixels have
                // `alpha=1`, so they composite the same as `REPLACE` would.
                blend: Some(cosmic::iced::wgpu::BlendState::ALPHA_BLENDING),
                write_mask: cosmic::iced::wgpu::ColorWrites::ALL,
            })],
            compilation_options: cosmic::iced::wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: cosmic::iced::wgpu::PrimitiveState {
            topology: cosmic::iced::wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: cosmic::iced::wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

// ---------------------------------------------------------------------------
// Uniform buffer layout (must match the WGSL `Uniforms` struct)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Uniforms {
    exposure: f32,
    /// Texture width in pixels — drives the WGSL contained-fit math.
    tex_w: f32,
    /// Texture height in pixels.
    tex_h: f32,
    sc_x: f32,
    sc_y: f32,
    sc_w: f32,
    sc_h: f32,
    /// Detail-view zoom: 1.0 = contain fit, each +1 doubles scale.
    zoom: f32,
    /// Pan offset in physical pixels (logical points × scale factor).
    pan_x: f32,
    pan_y: f32,
    /// Live crop, in source pixels: the sub-rectangle origin (left/top margins
    /// removed) and its size (source − removed margins). All zero/no-op when
    /// `crop_w == tex_w` and `crop_l == 0`. These are the EXIF-upright texture
    /// coordinates; the display rotation below composes after them.
    crop_l: f32,
    crop_t: f32,
    crop_w: f32,
    crop_h: f32,
    /// User display rotation as counter-clockwise 90° quarter-turns (`0`…`3`).
    /// `0.0` is the identity: the EXIF-upright frame, byte-identical render.
    rot: f32,
    /// Non-zero when the "view dimmed crop area" overlay is shown: the shader
    /// draws the full uncropped frame on top of the zoomed crop and dims
    /// everything outside the crop rectangle.
    show_mask: f32,
    /// Minimum padding (physical pixels) kept around the image in crop mode.
    /// The WGSL contain-fit base shrinks by `pad` on every side so a white
    /// margin is always visible; zooming in grows the image into it.
    pad: f32,
    /// Non-zero when the texture is a true sensor-linear NEGATIVE the shader
    /// must density-invert per fragment (a film preset). Drives the WGSL
    /// inversion branch in `shade()`.
    inv: f32,
    /// Clear-film transmission anchor (inversion black point): `film::positive`
    /// maps `value == inv_base` to positive black.
    inv_base: f32,
    /// Usable density range above the base (stock `d_max`); `density == d_max`
    /// maps to positive white.
    inv_d_max: f32,
    /// Density-space tone exponent (stock `gamma`) applied to normalized
    /// density.
    inv_gamma: f32,
}

// SAFETY: Uniforms is repr(C) with all f32 fields.
unsafe impl bytemuck::Pod for Uniforms {}
unsafe impl bytemuck::Zeroable for Uniforms {}

// ---------------------------------------------------------------------------
// f32 → f16 (IEEE 754 binary16) bit conversion
// ---------------------------------------------------------------------------
//
// `R16Float` is filterable on WebGPU; `R32Float` is not. Half-float has
// 1 sign + 5 exponent + 10 mantissa — plenty of precision for our
// post-inversion linear-light tonal data (mostly 0..1.5, occasionally
// higher after a stop of exposure). The conversion runs once per decode
// (~16 MB f32 → 8 MB u16 at 2048²).
//
// Subnormal f32 inputs flush to zero and overflows flush to infinity;
// round-to-nearest-even is used for mantissa truncation. No half-float
// subnormals are produced (input range is large enough that the small
// loss of precision below ~6e-8 linear is inaudible).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_field = ((bits >> 23) & 0xff).cast_signed();
    let mant = bits & 0x007f_ffff;

    if exp_field == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    if exp_field == 0 {
        return sign;
    }

    let unbiased = exp_field - 127;

    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        return sign;
    }

    let half_exp = (unbiased + 15) as u32;
    let mut half_mant = mant >> 13;
    let round_part = mant & 0x1fff;

    // Round-to-nearest-even.
    if round_part > 0x1000 || (round_part == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
        if half_mant >= 0x400 {
            // Mantissa overflow → bump exponent (mantissa bits implicitly 0).
            let new_exp = half_exp + 1;
            if new_exp >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((new_exp as u16) << 10);
        }
    }

    sign | ((half_exp as u16) << 10) | (half_mant as u16)
}

/// Decode a half-float bit pattern to f32 — the inverse of [`f32_to_half`].
/// Used by the parity tests to simulate the GPU's half-float LUT texture.
#[cfg(test)]
#[must_use]
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    if exp == 0 {
        // Zero or subnormal → flush to zero (matches f32_to_half's subnormal
        // handling: it never emits half subnormals).
        return f32::from_bits(sign << 31);
    }
    if exp == 0x1f {
        return if mant == 0 {
            f32::from_bits((sign << 31) | 0x7f80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7fc0_0000)
        };
    }
    // Re-bias the exponent: half bias 15 → f32 bias 127.
    let f32_bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    f32::from_bits(f32_bits)
}

/// Convert a `Vec<f32>` of linear mono values into an `R16Float`-compatible
/// byte buffer for `queue.write_texture`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mono_to_half_bytes(mono: &[f32]) -> Vec<u8> {
    let mut half = Vec::with_capacity(mono.len() * 2);
    for &v in mono {
        half.extend_from_slice(&f32_to_half(v).to_le_bytes());
    }
    half
}

/// One level of the detail texture's mip chain, ready for `write_texture`.
struct MipLevel {
    /// Mip width (`max(1, floor(parent/2))`, or the texture width at level 0).
    width: u32,
    /// Mip height, halving like the width.
    height: u32,
    /// Tightly packed `R16Float` (half) bytes: `width × height × 2`.
    data: Vec<u8>,
}

/// Build the full mip chain for a linear mono texture: level 0 is the raw
/// half-float upload, each level below is a 2×2 area-average (box) downsample
/// of the previous in LINEAR light, halving to `max(1, floor(dim/2))` and
/// stopping at 1×1. Edge blocks average only their real contributing texels.
///
/// The averaging happens on the TRUE sensor data before any tone mapping, so
/// the WGSL's per-fragment curve/inversion — which samples the minified texels
/// and THEN applies the tone — sees correctly pre-filtered values (the same
/// flatten-before-average ordering the CPU thumbnail bake uses). Without this
/// chain the single-level texture aliases the high-frequency grain into a
/// moiré whenever the view minifies (notably after the native level-up installs
/// an up-to-8192 texture and the user zooms back out to contain fit).
fn build_mip_chain(mono: &[f32], width: u32, height: u32) -> Vec<MipLevel> {
    let mut levels = Vec::new();
    levels.push(MipLevel {
        width,
        height,
        data: mono_to_half_bytes(mono),
    });

    let mut src = mono.to_vec();
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let (next, nw, nh) = downsample_half(&src, w, h);
        levels.push(MipLevel {
            width: nw,
            height: nh,
            data: mono_to_half_bytes(&next),
        });
        src = next;
        w = nw;
        h = nh;
    }
    levels
}

/// 2×2 area-average downsample of a mono buffer: output pixel `(x, y)` is the
/// mean of the source's `2x`..`2x+2` × `2y`..`2y+2` block, clamped to the
/// source bounds so odd-dimension edges average their actual texels.
#[allow(clippy::cast_precision_loss)]
fn downsample_half(mono: &[f32], width: u32, height: u32) -> (Vec<f32>, u32, u32) {
    let out_w = u32::max(width / 2, 1);
    let out_h = u32::max(height / 2, 1);
    let mut out = vec![0.0_f32; out_w as usize * out_h as usize];
    for y in 0..out_h {
        for x in 0..out_w {
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for dy in 0..2_u32 {
                let sy = y * 2 + dy;
                if sy >= height {
                    continue;
                }
                for dx in 0..2_u32 {
                    let sx = x * 2 + dx;
                    if sx >= width {
                        continue;
                    }
                    sum += mono[sy as usize * width as usize + sx as usize];
                    count += 1;
                }
            }
            out[y as usize * out_w as usize + x as usize] = sum / count as f32;
        }
    }
    (out, out_w, out_h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::ACTIVE_STOCK;

    #[test]
    fn crop_uv_geometry_zero_crop_is_identity() {
        // Same source and texture dims → the margin mapping is a no-op.
        let (origin, size) = crop_uv_geometry(CropMargins::default(), 400, 300, 400, 300);
        assert_eq!(origin, (0.0, 0.0));
        assert_eq!(size, (400.0, 300.0));
    }

    #[test]
    fn crop_uv_geometry_offsets_origin_by_left_top_and_shrinks_size() {
        // Source == texture (native): margins map 1:1 onto the texture.
        let crop = CropMargins {
            top: 10,
            right: 5,
            bottom: 30,
            left: 15,
        };
        let (origin, size) = crop_uv_geometry(crop, 400, 300, 400, 300);
        assert_eq!(origin, (15.0, 10.0));
        assert_eq!(size, (380.0, 260.0));
    }

    #[test]
    fn crop_uv_geometry_scales_source_px_onto_the_downscaled_texture() {
        // A 6000x4000 source shown at a 0.1× 600x400 texture: a stored
        // source-pixel crop (100 left, 50 top) must remove 10 and 5 texture
        // pixels respectively so the live view and the thumbnail bake stay in
        // the same reference frame. Before the src-aware fix each axis was
        // treated as texture pixels, so the downscaled view truncated ~10× more
        // than the bake revealed.
        let crop = CropMargins {
            top: 50,
            right: 100,
            bottom: 150,
            left: 100,
        };
        let (origin, size) = crop_uv_geometry(crop, 600, 400, 6000, 4000);
        assert_eq!(origin, (10.0, 5.0));
        assert_eq!(size, (580.0, 380.0));
    }

    #[test]
    fn crop_uv_geometry_scales_a_portrait_crop_onto_the_rotated_texture() {
        // Rotate90 display source (4000x6000) downscaled to a 256x384 portrait
        // texture (~0.064×, the THUMB-long-edge ratio is coincidental here).
        let crop = CropMargins {
            top: 300,
            right: 0,
            bottom: 300,
            left: 0,
        };
        let (origin, size) = crop_uv_geometry(crop, 256, 384, 4000, 6000);
        assert_eq!(origin, (0.0, 19.2));
        assert_eq!(size, (256.0, 384.0 - 38.4));
    }

    #[test]
    fn zoom_100_mirrors_the_wgsl_contain_math() {
        // 1:1 (one texture pixel per physical screen pixel) on a 2000×1000
        // texture in a 1000×1000 logical widget at scale factor 1.0: the width
        // axis needs 2× contain, so the cap is 1 + log2(2) = 2.0.
        let program = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            0.0,
            CropMargins::default(),
            0,
            1,
            2000,
            None,
        );
        assert!((program.zoom_100(1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);

        // An odd display rotation swaps which axis contains: in a 2000×1000
        // widget the cropped height (1000) is the limiting axis at 2× → still
        // 2.0; without rotation the width (2000) exactly fills → cap 1.0.
        let rotated = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            0.0,
            CropMargins::default(),
            1,
            1,
            2000,
            None,
        );
        assert!((rotated.zoom_100(2000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);
        assert!((program.zoom_100(2000.0, 1000.0, 1.0) - 1.0).abs() < 1e-5);

        // Scale factor converts logical → physical: at 2× UI the same texture
        // already fills a 1000-logical widget at 1:1, so the cap is contain.
        assert!((program.zoom_100(1000.0, 1000.0, 2.0) - 1.0).abs() < 1e-5);

        // A crop that halves the frame halves the cap (you magnify the crop).
        let cropped = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            0.0,
            CropMargins {
                top: 0,
                right: 1000,
                bottom: 0,
                left: 0,
            },
            0,
            1,
            2000,
            None,
        );
        assert!((cropped.zoom_100(1000.0, 1000.0, 1.0) - 1.0).abs() < 1e-5);

        // A widget larger than the texture never drops the cap below contain.
        assert!((program.zoom_100(4000.0, 4000.0, 1.0) - 1.0).abs() < 1e-5);

        // Crop-mode padding shrinks the contain-fit base: on a 2000×1000
        // texture in a 1000×1000 widget, 100px padding leaves 800×800 for the
        // image, so the width axis needs 2.5× (2000/800) → cap 1 + log2(2.5).
        let mut padded = program.clone();
        padded.set_pad(100.0);
        assert!((padded.zoom_100(1000.0, 1000.0, 1.0) - (1.0 + 2.5f32.log2())).abs() < 1e-5);

        // A widget smaller than 2× the padding clamps the available box to 1
        // physical pixel on each side (no divide-by-zero / negative cap) —
        // the cap is then just the 1:1 point for a 1px box.
        assert!(
            (padded.zoom_100(50.0, 50.0, 1.0) - (1.0 + 2000.0f32.log2())).abs() < 1e-5
        );

        // Zero padding is the unpadded layout (regression guard for the mirror).
        padded.set_pad(0.0);
        assert!((padded.zoom_100(1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn zoom_100_for_delegates_on_the_own_texture_and_scales_in_log2() {
        // The projection helper with the shader's own texture dims is exactly
        // `zoom_100` (same 2000x1000 texture → cap 2.0 in a 1000x1000 widget).
        let program = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            0.0,
            CropMargins::default(),
            0,
            1,
            2000,
            None,
        );
        assert!(
            (program.zoom_100_for(2000, 1000, 1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5
        );
        assert!(
            (program.zoom_100_for(2000, 1000, 1000.0, 1000.0, 1.0)
                - program.zoom_100(1000.0, 1000.0, 1.0))
                .abs()
                < 1e-5
        );

        // Doubling both texture axes doubles the physical pixel count at the
        // same scale, so the 1:1 cap grows by exactly one log2 unit — this is
        // what lets the detail view project the native level-up's cap.
        let native = program.zoom_100_for(4000, 2000, 1000.0, 1000.0, 1.0);
        assert!((native - (1.0 + 4.0f32.log2())).abs() < 1e-5);

        // Cropping half the *projected* frame magnifies the projected cap the
        // same way it does the real one (crop is authored in source space).
        let cropped = DetailProgram::new(
            vec![0.0; 2000 * 1000],
            2000,
            1000,
            0.0,
            CropMargins {
                top: 0,
                right: 1000,
                bottom: 0,
                left: 0,
            },
            0,
            1,
            2000,
            None,
        );
        assert!((cropped.zoom_100_for(4000, 2000, 1000.0, 1000.0, 1.0) - 2.0).abs() < 1e-5);

        // The 1:1 projection is floored at contain fit like the real cap.
        assert!((program.zoom_100_for(1000, 500, 4000.0, 4000.0, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn new_restores_display_source_dims_from_the_long_edge() {
        // Landscape source: texture 600x400 (downscale of 6000x4000) → source
        // long edge 6000 lands on the width axis.
        let program = DetailProgram::new(
            vec![0.0; 600 * 400],
            600,
            400,
            0.0,
            CropMargins::default(),
            0,
            1,
            6000,
            None,
        );
        assert_eq!(program.source_dimensions(), (6000, 4000));

        // Portrait (rotated) source: texture 400x600 → long edge lands height.
        let program = DetailProgram::new(
            vec![0.0; 400 * 600],
            400,
            600,
            0.0,
            CropMargins::default(),
            0,
            1,
            6000,
            None,
        );
        assert_eq!(program.source_dimensions(), (4000, 6000));

        // Native decode (texture == source): the long edge restores exactly.
        let program = DetailProgram::new(
            vec![0.0; 400 * 600],
            400,
            600,
            0.0,
            CropMargins::default(),
            0,
            1,
            600,
            None,
        );
        assert_eq!(program.source_dimensions(), (400, 600));
    }

    /// Mirrors the WGSL `rotate_uv`/`rotated_dims` display rotation: mapping a
    /// point `(u, v)` over the DISPLAYED (rotated) frame back into the
    /// EXIF-upright texture's `(tu, tv)` normalized coordinates, for a
    /// cumulative counter-clockwise `rot` quarter-turns. The WGSL selection
    /// chain must match this exactly; testing the Rust mirror catches
    /// arithmetic contradictions even though the WGSL itself only parses here.
    fn rot_uv(rot: u8, u: f32, v: f32) -> (f32, f32) {
        match rot & 3 {
            0 => (u, v),
            1 => (1.0 - v, u),
            2 => (1.0 - u, 1.0 - v),
            _ => (v, 1.0 - u),
        }
    }

    #[test]
    fn rotated_dimensions_swap_axes_for_odd_turns() {
        // An odd quarter-turn swaps which frame axis is the display horizontal:
        // the contain-fit (and the thumbnail print) become as tall as the
        // source was wide.
        fn rotated_dims<W: Copy>(w: W, h: W, rot: u8) -> (W, W) {
            match rot & 3 {
                0 | 2 => (w, h),
                _ => (h, w),
            }
        }
        assert_eq!(rotated_dims(3_u32, 2_u32, 0), (3, 2));
        assert_eq!(rotated_dims(3_u32, 2_u32, 1), (2, 3));
        assert_eq!(rotated_dims(3_u32, 2_u32, 2), (3, 2));
        assert_eq!(rotated_dims(3_u32, 2_u32, 3), (2, 3));
    }

    #[test]
    fn rotate_uv_maps_display_quarter_turns_to_texture_uv() {
        // Identity: the displayed (u, v) IS the texture coordinate.
        assert_eq!(rot_uv(0, 0.25, 0.75), (0.25, 0.75));
        // One CCW 90°: the texture's top edge lands on the display's left, so
        // the display's LEFT (u=0) samples the texture's TOP row (tv=0) while
        // moving right across the display walks the texture top→bottom.
        assert_eq!(rot_uv(1, 0.0, 0.0), (1.0, 0.0));
        assert_eq!(rot_uv(1, 0.0, 1.0), (0.0, 0.0));
        assert_eq!(rot_uv(1, 1.0, 1.0), (0.0, 1.0));
        assert_eq!(rot_uv(1, 1.0, 0.0), (1.0, 1.0));
        // Two turns: pure 180° reversal of both axes.
        assert_eq!(rot_uv(2, 0.25, 0.75), (0.75, 0.25));
        // Three turns (one CW): the texture's top edge lands on the display's
        // right; the display's right (u=1) samples the top row (tv=0).
        assert_eq!(rot_uv(3, 1.0, 0.0), (0.0, 0.0));
        assert_eq!(rot_uv(3, 1.0, 1.0), (1.0, 0.0));
        assert_eq!(rot_uv(3, 0.0, 0.0), (0.0, 1.0));
        // Four turns return to the identity, and the count wraps mod 4 the way
        // the shader's `rotation & 3` write keeps it.
        assert_eq!(rot_uv(4, 0.25, 0.75), (0.25, 0.75));
    }

    #[test]
    fn f32_to_half_matches_canonical_values() {
        assert_eq!(f32_to_half(0.0), 0x0000, "+0");
        assert_eq!(f32_to_half(-0.0), 0x8000, "-0");
        assert_eq!(f32_to_half(1.0), 0x3c00, "1.0");
        assert_eq!(f32_to_half(-1.0), 0xbc00, "-1.0");
        assert_eq!(f32_to_half(0.5), 0x3800, "0.5");
        assert_eq!(f32_to_half(2.0), 0x4000, "2.0");
        assert_eq!(f32_to_half(65504.0), 0x7bff, "largest finite f16");
        assert_eq!(f32_to_half(65536.0), 0x7c00, "overflow → +∞");
        assert_eq!(f32_to_half(-65536.0), 0xfc00, "overflow → -∞");
        assert_eq!(f32_to_half(0.1), 0x2e66, "0.1");
        assert_eq!(f32_to_half(-0.1), 0xae66, "-0.1");
        assert_eq!(f32_to_half(0.000_488_281_25), 0x1000, "2^-11");
        assert_eq!(
            f32_to_half(0.000_061_035_156_25),
            0x0400,
            "smallest f16 normal ≈ 2^-14"
        );
        // Values smaller than the smallest f16 normal range flush to zero;
        // we don't produce half-float subnormals.
        assert_eq!(
            f32_to_half(0.000_030_517_578_125),
            0x0000,
            "below smallest normal flushes to 0 (no subnormal handling)"
        );
    }

    #[test]
    fn f32_to_half_underflow_flushes_to_zero() {
        // Anything smaller than the smallest f16 normal (≈ 6.1e-5) flushes
        // to zero; we do not produce half-float subnormals.
        assert_eq!(f32_to_half(1.0e-7), 0x0000, "tiny positive → +0");
        assert_eq!(f32_to_half(-1.0e-7), 0x8000, "tiny negative → -0");
    }

    #[test]
    fn f32_to_half_handles_infinity_and_nan() {
        assert_eq!(f32_to_half(f32::INFINITY), 0x7c00, "+∞");
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xfc00, "-∞");
        assert_eq!(f32_to_half(f32::NAN) & 0x7c00, 0x7c00, "NaN keeps exp");
        assert_eq!(f32_to_half(f32::NAN) & 0x0200, 0x0200, "NaN keeps mant bit");
    }

    #[test]
    fn mono_to_half_bytes_lays_out_u16_le() {
        // For 1.0 we expect two bytes: 0x00 0x3C (little-endian u16 0x3C00).
        let bytes = mono_to_half_bytes(&[1.0]);
        assert_eq!(bytes, [0x00, 0x3c]);
    }

    #[test]
    fn build_mip_chain_halves_dimensions_down_to_one() {
        // A 2000×1000 buffer: levels halve per axis (floor) and the chain stops
        // at 1×1. `floor(log2(2000)) + 1 = 11` levels (2000→1000→…→3→1).
        let mono: Vec<f32> = vec![0.0; 2000 * 1000];
        let mips = build_mip_chain(&mono, 2000, 1000);
        assert_eq!(mips.len(), 11);
        assert_eq!((mips[0].width, mips[0].height), (2000, 1000));
        assert_eq!((mips[1].width, mips[1].height), (1000, 500));
        assert_eq!((mips[2].width, mips[2].height), (500, 250));
        assert_eq!((mips[3].width, mips[3].height), (250, 125));
        assert_eq!((mips[4].width, mips[4].height), (125, 62));
        assert_eq!(
            (mips[10].width, mips[10].height),
            (1, 1),
            "last level is 1×1"
        );
        for mip in &mips {
            assert_eq!(
                mip.data.len(),
                mip.width as usize * mip.height as usize * 2,
                "tightly packed half bytes"
            );
        }
    }

    #[test]
    fn build_mip_chain_level0_is_the_raw_half_data() {
        let mono: Vec<f32> = (0..100).map(|v| v as f32 / 99.0).collect();
        let mips = build_mip_chain(&mono, 10, 10);
        assert_eq!(mips[0].width, 10);
        assert_eq!(mips[0].height, 10);
        assert_eq!(mips[0].data, mono_to_half_bytes(&mono));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn build_mip_chain_box_averages_source_blocks_in_linear_light() {
        // A 4×4 ramp: mip 1's (0,0) texel must be the mean of the base's 2×2
        // top-left block, and (1,1) the mean of the 2×2 bottom-right block.
        let mono: Vec<f32> = (0..16).map(|v| v as f32 / 15.0).collect();
        let mips = build_mip_chain(&mono, 4, 4);
        assert_eq!((mips[1].width, mips[1].height), (2, 2));

        let decode = |mip: &MipLevel, x: u32, y: u32| {
            let offset = (y as usize * mip.width as usize + x as usize) * 2;
            half_to_f32(u16::from_le_bytes([mip.data[offset], mip.data[offset + 1]]))
        };
        let mean = |coords: &[(usize, usize)]| {
            let sum: f32 = coords.iter().map(|&(x, y)| mono[y * 4 + x]).sum();
            sum / coords.len() as f32
        };
        assert!((decode(&mips[1], 0, 0) - mean(&[(0, 0), (1, 0), (0, 1), (1, 1)])).abs() < 1e-3);
        assert!((decode(&mips[1], 1, 1) - mean(&[(2, 2), (3, 2), (2, 3), (3, 3)])).abs() < 1e-3);
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn build_mip_chain_handles_odd_dimensions() {
        // A 3×2 buffer (row-major: [0,0.1,0.2, 0.3,0.4,0.5]) has no exact 2×2
        // tiling: strict halving drops the odd column/row tail, so the single
        // 1×1 mip 1 averages the 2×2 top-left block — 0.0, 0.1, 0.3, 0.4.
        let mono: Vec<f32> = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let mips = build_mip_chain(&mono, 3, 2);
        assert_eq!(mips.len(), 2);
        assert_eq!((mips[1].width, mips[1].height), (1, 1));
        let expected = (0.0 + 0.1 + 0.3 + 0.4) / 4.0;
        let decoded = half_to_f32(u16::from_le_bytes([mips[1].data[0], mips[1].data[1]]));
        assert!((decoded - expected).abs() < 1e-3);
    }

    #[test]
    fn build_mip_chain_single_pixel_is_a_single_level() {
        let mips = build_mip_chain(&[0.42], 1, 1);
        assert_eq!(mips.len(), 1);
        assert_eq!((mips[0].width, mips[0].height), (1, 1));
    }

    #[test]
    fn curve_remap_is_identity_at_defaults() {
        let (ratio, exponent) = curve_remap(1.0, 1.0, 1.0, 0.2, 0.4, 0.92);
        assert!((ratio - 1.0).abs() < 1e-6);
        assert!((exponent - 1.0).abs() < 1e-6);
    }

    #[test]
    fn contrast_pivots_around_the_image_midgray() {
        let (shadow, mid, white) = (0.15_f32, 0.4_f32, 0.92_f32);
        for kc in [0.6_f32, 1.3] {
            let (ratio, exponent) = curve_remap(kc, 1.0, 1.0, shadow, mid, white);
            let t_mid = (ratio * mid.powf(exponent)).clamp(0.0, 1.0);
            assert!(
                (t_mid - mid).abs() < 1e-5,
                "contrast {kc}: T(mid) = {t_mid}"
            );
        }
    }

    #[test]
    fn highlights_pivots_around_the_image_shadow_anchor() {
        let (shadow, mid, white) = (0.15_f32, 0.4_f32, 0.92_f32);
        for kh in [0.6_f32, 1.3] {
            let (ratio, exponent) = curve_remap(1.0, kh, 1.0, shadow, mid, white);
            let t_shadow = (ratio * shadow.powf(exponent)).clamp(0.0, 1.0);
            assert!(
                (t_shadow - shadow).abs() < 1e-5,
                "highlights {kh}: T(shadow) = {t_shadow}"
            );
        }
    }

    #[test]
    fn shadows_pivots_around_the_image_white_point() {
        let (shadow, mid, white) = (0.15_f32, 0.4_f32, 0.92_f32);
        for ks in [0.6_f32, 1.3] {
            let (ratio, exponent) = curve_remap(1.0, 1.0, ks, shadow, mid, white);
            let t_white = (ratio * white.powf(exponent)).clamp(0.0, 1.0);
            assert!(
                (t_white - white).abs() < 1e-5,
                "shadows {ks}: T(white) = {t_white}"
            );
        }
    }

    #[test]
    fn highlights_and_shadows_lift_their_own_region() {
        // The user-facing direction: raising a lift value presses `2^-L` (shadow
        // power, pivot white) or `2^+L` (highlight power, pivot shadow). Each
        // must brighten its named region while pinning the far anchor.
        let (shadow, mid, white) = (0.15_f32, 0.4_f32, 0.92_f32);
        let dark = 0.2_f32;
        let bright = 0.7_f32;
        let tone = |kh: f32, ks: f32, p: f32| {
            let (ratio, exponent) = curve_remap(1.0, kh, ks, shadow, mid, white);
            (ratio * p.powf(exponent)).clamp(0.0, 1.0)
        };
        // Highlight lift: power up, the bright patch rises, the deep shadow
        // barely moves (pivoted at the shadow anchor).
        let (hid, hlift) = (1.0_f32, 2.0_f32);
        assert!(tone(hlift, 1.0, bright) > tone(hid, 1.0, bright));
        // Shadow lift: power down, the dark patch rises, white stays pinned.
        let (sid, slift) = (1.0_f32, 0.5_f32);
        assert!(tone(1.0, slift, dark) > tone(1.0, sid, dark));
        assert!((tone(1.0, slift, white) - white).abs() < 1e-5);
        // Both directions are the identity at the centered lift value.
        assert!((tone(1.0, 1.0, bright) - bright).abs() < 1e-5);
    }

    #[test]
    fn curve_remap_composes_the_three_pivots_exactly() {
        // Applying the highlights and shadows powers after the contrast power
        // must equal the single (ratio, exp) the shader applies — the
        // composition is exact for the underlying power functions. The shader
        // applies one final clamp (never an intermediate one), so compare raw
        // then both-clamped.
        let (shadow, mid, white) = (0.15_f32, 0.4_f32, 0.92_f32);
        let (kc, kh, ks) = (1.25_f32, 0.75_f32, 1.4_f32);
        let (ratio, exponent) = curve_remap(kc, kh, ks, shadow, mid, white);
        for p in [0.0_f32, 0.05, 0.15, 0.4, 0.6, 0.92, 1.0] {
            let c = mid.powf(1.0 - kc) * p.powf(kc);
            let s = white.powf(1.0 - ks) * c.powf(ks);
            let sequential = shadow.powf(1.0 - kh) * s.powf(kh);
            let composed = ratio * p.powf(exponent);
            assert!(
                (sequential - composed).abs() < 1e-4,
                "p {p}: sequential {sequential} vs composed {composed}"
            );
            assert!(
                (sequential.clamp(0.0, 1.0) - composed.clamp(0.0, 1.0)).abs() < 1e-4,
                "p {p}: clamped sequential vs clamped composed"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn tone_anchors_find_shadow_mid_and_white_on_a_known_ramp() {
        let mono: Vec<f32> = (0..=2000).map(|v| v as f32 / 2000.0).collect();
        let (shadow, mid, white) = tone_anchors(&mono);
        assert!((shadow - 0.1).abs() < 0.02, "shadow {shadow}");
        assert!((mid - 0.5).abs() < 0.01, "median {mid}");
        assert!((white - 0.98).abs() < 0.02, "white {white}");
    }

    #[test]
    fn tone_anchors_guard_degenerate_black_frames() {
        let (shadow, mid, white) = tone_anchors(&[0.0; 256]);
        assert_eq!(shadow, MIN_ANCHOR);
        assert_eq!(mid, MIN_ANCHOR);
        assert_eq!(white, MIN_ANCHOR);
    }

    #[test]
    fn sensor_gain_flips_sign_for_inverted_presets() {
        // An already-positive preset multiplies the positive by 2^EV: +1 EV
        // doubles it, EV 0 is the identity.
        assert_eq!(sensor_gain(0.0, false), 1.0);
        assert!((sensor_gain(1.0, false) - 2.0).abs() < 1e-6);
        assert!((sensor_gain(-1.0, false) - 0.5).abs() < 1e-6);
        // An INVERTED preset multiplies the sensor-negative by 2^-EV: +1 EV
        // halves the transmission, densifying the film and brightening the
        // positive — the same user-facing "brighter" direction as non-inverted.
        assert!((sensor_gain(1.0, true) - 0.5).abs() < 1e-6);
        assert!((sensor_gain(-1.0, true) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn film_anchor_fractiles_measure_sensor_ranks() {
        // On a uniform transmission ramp the fractiles land at their nominal
        // ranks: shadow from the 90th, mid from the median, white from the 2nd.
        let mono: Vec<f32> = (0..=2000).map(|v| v as f32 / 2000.0).collect();
        let (shadow, mid, white) = film_anchor_fractiles(&mono);
        assert!((shadow - 0.90).abs() < 0.02, "shadow-rank {shadow}");
        assert!((mid - 0.50).abs() < 0.01, "median {mid}");
        assert!((white - 0.02).abs() < 0.02, "white-rank {white}");
    }

    #[test]
    fn film_pivots_at_ev0_match_anchors_of_an_inverted_ramp() {
        // At EV 0 the derived pivots must equal the regular anchors of a buffer
        // that was density-inverted first — the CPU twin of what the WGSL
        // renders. Fractiles in rank space + `invert_value` reproduce the
        // old percentiles exactly (parity, so existing renders don't shift).
        let mono: Vec<f32> = (0..=2000).map(|v| v as f32 / 2000.0).collect();
        let stock = crate::film::ACTIVE_STOCK;
        let fractiles = film_anchor_fractiles(&mono);
        let (shadow, mid, white) = film_pivots_at_gain(fractiles, ACTIVE_STOCK.base, 1.0, stock);

        let positive: Vec<f32> = mono
            .iter()
            .map(|&v| invert_value(v, ACTIVE_STOCK.base, &stock))
            .collect();
        let (shadow_direct, mid_direct, white_direct) = tone_anchors(&positive);

        assert!((shadow - shadow_direct).abs() < 0.01, "{shadow} vs {shadow_direct}");
        assert!((mid - mid_direct).abs() < 0.01, "{mid} vs {mid_direct}");
        // The white anchor sits on the steepest part of this synthetic ramp
        // (near-transmission samples map to near-black positive), so the
        // `ANCHOR_INV_STRIDE` subsampling shifts the fractile separator sample
        // slightly and the histograms quantize the steep transform — absorb
        // that (real negatives don't sit exactly on the clear-film endpoint).
        assert!((white - white_direct).abs() < 0.03, "{white} vs {white_direct}");

        // The direction is inverted: a ramp of transmissions (clearer at the
        // high end) maps to a descending positive, so the positive's mid-gray
        // sits at the LOW end of the transmission ramp.
        assert!(mid < 0.5, "inverted mid-anchor {mid} should sit below 0.5");
    }

    #[test]
    fn film_pivots_track_the_current_ev() {
        // The drift fix: pivots must anchor on the ACTUAL render at any EV, not
        // the EV-0 the fractiles were captured at. Derive the pivots for +1 EV,
        // then CPU-render the same frame at +1 EV and measure ITS percentiles —
        // grid/detail/export must agree.
        let mono: Vec<f32> = (0..=2000).map(|v| v as f32 / 2000.0).collect();
        let stock = crate::film::ACTIVE_STOCK;
        let base = ACTIVE_STOCK.base;
        let gain = sensor_gain(1.0, true); // +1 EV on a film negative = × ½ sensor

        let fractiles = film_anchor_fractiles(&mono);
        let (shadow, mid, white) = film_pivots_at_gain(fractiles, base, gain, stock);

        let rendered: Vec<f32> = mono
            .iter()
            .map(|&t| invert_value(t * gain, base, &stock))
            .collect();
        let (shadow_direct, mid_direct, white_direct) = tone_anchors(&rendered);

        assert!((shadow - shadow_direct).abs() < 0.02, "{shadow} vs {shadow_direct}");
        assert!((mid - mid_direct).abs() < 0.01, "{mid} vs {mid_direct}");
        assert!((white - white_direct).abs() < 0.03, "{white} vs {white_direct}");

        // And the derive drifts with EV: brightening the film lifts the white
        // pivot above the frozen EV-0 value — the defect the fractiles fix.
        let (_, _, white_ev0) = film_pivots_at_gain(fractiles, base, 1.0, stock);
        assert!(white > white_ev0, "white pivot {white_ev0} must rise to {white}");
    }

    #[test]
    fn set_exposure_rederives_inverted_pivots() {
        let mono: Vec<f32> = (0..=2000).map(|v| v as f32 / 2000.0).collect();
        let mut program = DetailProgram::new(
            mono,
            2001,
            1,
            0.0,
            CropMargins::default(),
            0,
            1,
            2001,
            Some((ACTIVE_STOCK, ACTIVE_STOCK.base)),
        );
        let (white_ev0, mid_ev0) = (program.white, program.mid);

        // +1 EV on the negative densifies the film → the positive brightens and
        // its pivots must re-anchor on the new render (the drift fix).
        program.set_exposure(1.0);
        assert!(
            program.white > white_ev0,
            "white pivot {white_ev0} → {}",
            program.white
        );
        assert!(
            program.mid > mid_ev0,
            "mid pivot {mid_ev0} → {}",
            program.mid
        );
    }

    #[test]
    fn shadow_pivot_floor_keeps_the_highlights_power_usable() {
        // A frame whose bottom decile is literal black (the clear-film plateau
        // at EV ≤ 0) yields a shadow pivot near MIN_ANCHOR. Without the floor,
        // `s^(1-kh)` ≈ 31× at kh=1.5 would blow the whole positive to white.
        let (shadow, mid, white) = (MIN_ANCHOR, 0.3_f32, 0.9_f32);
        let (ratio, exp) = curve_remap(1.0, 1.5, 1.0, shadow, mid, white);
        let lifted_mid = ratio * mid.powf(exp);
        assert!(lifted_mid < 1.0, "highlights power blew out the mid-gray: {lifted_mid}");

        // The identity still travels through the floored pivot untouched.
        let (ratio_id, exp_id) = curve_remap(1.0, 1.0, 1.0, shadow, mid, white);
        assert!((ratio_id - 1.0).abs() < 1e-6 && (exp_id - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_curve_is_identity_at_defaults() {
        // The grid thumbnail bake calls `apply_curve` with the identity curve,
        // so untouched renders must be byte-identical (no phantom tone shift).
        let mut values = vec![0.0_f32, 0.13, 0.5, 0.84, 1.0];
        let original = values.clone();
        apply_curve(&mut values, 1.0, 1.0, 1.0, 0.15, 0.45, 0.92);
        assert!(
            values
                .iter()
                .zip(&original)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "identity curve changed values: {values:?}"
        );
    }

    #[test]
    fn apply_curve_matches_tone_model() {
        // `apply_curve` folds the pivoted powers with `curve_remap` then applies
        // the per-pixel `tone_model` expression — the single source of the tone
        // math (the WGSL now texture-samples a LUT of it instead of re-deriving
        // the expression). Re-derive the expression independently and confirm
        // they agree across the input range.
        let (shadow, mid, white) = (0.18_f32, 0.42_f32, 0.93_f32);
        let (contrast, highlights, shadows) = (1.25_f32, 0.7_f32, 1.3_f32);
        let (ratio, exponent) = curve_remap(contrast, highlights, shadows, shadow, mid, white);

        for p in [0.0_f32, 0.05, 0.18, 0.25, 0.42, 0.7, 0.93, 1.0, 3.0] {
            let expected = (ratio * p.powf(exponent)).clamp(0.0, 1.0);
            let mut v = [p];
            apply_curve(&mut v, contrast, highlights, shadows, shadow, mid, white);
            assert!((v[0] - expected).abs() < 1e-6, "p {p}: {v:?} vs {expected}");
        }
    }

    #[test]
    fn tone_lut_reproduces_tone_model_within_tolerance() {
        // The GPU samples a 2048×1 half-float LUT of `tone_model` with linear
        // interpolation; the CPU bakes call `tone_model` exactly. Bound that
        // approximation: building the LUT, decoding it back to f32 (as the
        // R16Float sampler would), and interpolating must stay within a small
        // tolerance of the exact expression everywhere.
        let (shadow, mid, white) = (0.15_f32, 0.45_f32, 0.92_f32);
        for (contrast, highlights, shadows) in [
            (1.0_f32, 1.0_f32, 1.0_f32),
            (1.25, 0.7, 1.3),
            (0.5, 2.0, 0.4),
            (1.8, 1.1, 2.5),
            // The widened contrast endpoints (±3 stops).
            (8.0, 1.0, 1.0),
            (0.125, 1.0, 1.0),
        ] {
            let (ratio, exponent) = curve_remap(contrast, highlights, shadows, shadow, mid, white);
            let bytes = build_tone_lut(contrast, highlights, shadows, shadow, mid, white);
            let lut: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            for p in [0.0_f32, 0.01, 0.1, 0.5, 0.9, 0.999, 1.0] {
                let exact = tone_model(p, ratio, exponent);
                let sampled = sample_tone_lut_f32(&lut, p);
                assert!(
                    (sampled - exact).abs() <= 1e-3,
                    "curve {contrast}/{highlights}/{shadows} p {p}: lut {sampled} vs exact {exact}"
                );
            }
        }
    }

    #[test]
    fn wgsl_source_parses_and_validates() {
        // The shader is compiled by wgpu at first launch, so a parse error
        // would only surface at runtime. Validate the source we embed every
        // test run instead.
        let source = include_str!("shader/exposure.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|err| panic!("WGSL failed to parse: {err}"));

        let stages: Vec<_> = module.entry_points.iter().map(|ep| ep.stage).collect();
        assert!(stages.contains(&naga::ShaderStage::Vertex));
        assert!(stages.contains(&naga::ShaderStage::Fragment));
    }
}
