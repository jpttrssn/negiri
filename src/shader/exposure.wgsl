// SPDX-License-Identifier: GPL-3.0-or-later

// How much to darken the region outside the crop in the "view dimmed crop
// area" overlay: the outside-crop color is multiplied by `1 - DIM_AMOUNT`.
// 0.5 keeps the surrounding frame readable at half brightness.
const DIM_AMOUNT: f32 = 0.5;

// Canonical fullscreen triangle: three NDC vertices whose triangle
// clips to the entire viewport rectangle. The (-1,-1) corner is
// shared; the other two overshoot to x=3 or y=3 so the diagonal
// across (-1,3)→(3,-1) covers every pixel when the render-pass
// scissor clips to the [-1,1]² viewport.
fn vertex_position(vertex_index: u32) -> vec2<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  3.0),
        vec2<f32>( 3.0, -1.0),
    );
    return positions[vertex_index];
}

struct Uniforms {
    exposure: f32,   // linear-light sensor gain: 2^EV, or 2^-EV when inverted (film) so +EV brightens the positive
    tex_w: f32,      // texture width in pixels
    tex_h: f32,      // texture height in pixels
    sc_x: f32,       // scissor rect origin x (physical pixels)
    sc_y: f32,       // scissor rect origin y (physical pixels)
    sc_w: f32,       // scissor rect width
    sc_h: f32,       // scissor rect height
    zoom: f32,       // detail-view zoom, 1.0 = contain fit, each +1 doubles scale
    pan_x: f32,      // pan offset of the image center, physical pixels
    pan_y: f32,
    // Live keyboard crop, IN TEXTURE PIXELS (the source-pixel margins scaled by
    // tex/src per axis on the CPU): the sub-rectangle of the texture to show.
    // `crop_l`/`crop_t` is the region origin (left/top margins removed) and
    // `crop_w`/`crop_h` its size (texture minus the removed margins). These
    // stay in the EXIF-upright frame; the display rotation below composes
    // AFTER them (the whole composed frame rotates). All four are zero/identity
    // when there is no crop, reproducing the full frame.
    crop_l: f32,
    crop_t: f32,
    crop_w: f32,
    crop_h: f32,
    // User display rotation as counter-clockwise 90° quarter-turns (0..3) on
    // top of the EXIF-upright texture. 0.0 is identity; the Rust side keeps it
    // in {0, 1, 2, 3} via `rotation & 3`.
    rot: f32,
    // When non-zero, draw the "view dimmed crop area" overlay: the full uncropped
    // frame at its own contain-fit on top of the zoomed crop, dimming everything
    // outside the crop rectangle so the user can see where the crop lands. The
    // overlay reuses the same zoom/pan; anything beyond the shader region is
    // clipped by the viewport. 0.0 = base zoom-crop view only.
    show_mask: f32,
    // Minimum padding (physical pixels) kept around the image in crop mode: the
    // contain-fit base shrinks by `pad` on every side, so a white margin is
    // always visible around the frame. Zooming in grows the image into the
    // padding (it is a minimum, not a clip). 0.0 = no padding (normal mode).
    pad: f32,
    // When non-zero, the texture is a TRUE sensor-linear NEGATIVE (a film
    // preset) that must be density-inverted per fragment: `exposure` multiplies
    // the sensor data FIRST, then the inversion maps transmission → positive,
    // and only then the curve applies. When zero (an already-positive scan) the
    // pipeline is unchanged: curve, then `exposure` gain, then sRGB.
    inv: f32,
    // Inversion params (valid when `inv != 0.0`), mirroring film::invert_value:
    // `inv_base` is the clear-film transmission (positive black), `inv_d_max`
    // the usable density range (positive white), `inv_gamma` the density-space
    // tone exponent.
    inv_base: f32,
    // The EFFECTIVE usable density range: the stock's `d_max` scaled by the
    // user Contrast (`d_max · 2^-contrast`), so the window can be tightened or
    // widened in log-density.
    inv_d_max: f32,
    inv_gamma: f32,
    // Density-domain region lifts (signed, normalized-positive units) for the
    // film path: a toe mask weighted `(1-p)^2` and a shoulder mask `p^2`, each
    // vanishing at the opposite end so the controls stay independent.
    region_shadows: f32,
    region_highlights: f32,
};

@group(0) @binding(0) var t_mono: texture_2d<f32>;
@group(0) @binding(1) var s_mono: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var t_tone: texture_2d<f32>;
@group(0) @binding(4) var s_tone: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Positions are already NDC; render-pass viewport clips the
    // oversize corners to the actual framebuffer rect.
    out.position = vec4<f32>(vertex_position(vertex_index), 0.0, 1.0);
    return out;
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        return c * 12.92;
    }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

/// Maps a fragment position (physical pixels) to the texture UV to sample,
/// applying the detail view's zoom/pan transform and the live keyboard crop.
///
/// The image scale is anchored to the CROPPED frame's contain-fit base:
/// `scale = min(sc_w/dw, sc_h/dh) * 2^(zoom - 1)` where `(dw, dh)` are the
/// crop dims as DISPLAYED — swapped by an odd rotation. At `zoom == 1` the
/// remaining (uncropped-away) region contain-fits the widget, so trimming edges
/// re-fits/zooms the kept content to fill the box. Each +1 zoom unit doubles the
/// scale; the image stays centered, displaced by `pan` physical px.
///
/// The sample position maps `rel` (over the rotated crop box) through
/// [`rotate_uv`] into the EXIF-upright cropped sub-rectangle
/// (`crop_l..crop_l+crop_w` × `crop_t..crop_t+crop_h`, in texture pixels), so the
/// UV spans only the kept content. A zero crop (`crop_w == tex_w`, `crop_l == 0`)
/// and zero rotation are exact identity.
///
/// When `show_mask != 0.0`, [`overlay_sample`] additionally lays out the full
/// uncropped frame on top (same zoom/pan, its own contain-fit on the full
/// texture) and flags whether each fragment lies outside the crop rectangle so
/// [`fs_main`] can dim it — the "view dimmed crop area" overlay.
struct ViewSample {
    uv: vec2<f32>,
    inside: bool,
}

fn in_bounds(uv: vec2<f32>) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

/// Maps a point `(u, v)` over the DISPLAYED (rotated) frame back into the
/// EXIF-upright texture's normalized `(tu, tv)` coordinates, for a cumulative
/// counter-clockwise `rot` quarter-turns:
///   rot 0 → (u,   v)
///   rot 1 → (1-v, u)
///   rot 2 → (1-u, 1-v)
///   rot 3 → (v,   1-u)
/// so after one CCW turn the texture's top edge lands on the display's left.
/// `rot == 0` is the identity, keeping the untouched render byte-identical.
/// (Mirror-tested in Rust as `rot_uv`.) `u`/`v` may leave [0, 1] while the
/// image is letterboxed or panned; the caller's `in_bounds(rel)` gates that.
fn rotate_uv(u: f32, v: f32, rot: f32) -> vec2<f32> {
    if rot == 1.0 {
        return vec2<f32>(1.0 - v, u);
    }
    if rot == 2.0 {
        return vec2<f32>(1.0 - u, 1.0 - v);
    }
    if rot == 3.0 {
        return vec2<f32>(v, 1.0 - u);
    }
    return vec2<f32>(u, v);
}

/// The dimensions a `w`×`h` frame presents after `rot` quarter turns: odd
/// turns swap the axes (a landscape 3×2 displays as 2×3).
fn rotated_dims(w: f32, h: f32, rot: f32) -> vec2<f32> {
    if rot == 1.0 || rot == 3.0 {
        return vec2<f32>(h, w);
    }
    return vec2<f32>(w, h);
}

/// The crop box (as fractions of the DISPLAYED frame) after `rot` quarter
/// turns: rotating the full frame also rotates the EXIF-upright axis-aligned
/// crop sub-rectangle into another axis-aligned sub-rectangle of the rotated
/// frame. `u0/u1`/`v0/v1` are the crop's horizontal/vertical fraction bounds
/// in the EXIF-upright frame; the returned `(du0, du1, dv0, dv1)` are the same
/// bounds in the DISPLAYED frame (used by [`overlay_sample`]'s dim rect).
fn crop_box_display(rot: f32, u0: f32, u1: f32, v0: f32, v1: f32) -> vec4<f32> {
    if rot == 1.0 {
        return vec4<f32>(v0, v1, 1.0 - u1, 1.0 - u0);
    }
    if rot == 2.0 {
        return vec4<f32>(1.0 - u1, 1.0 - u0, 1.0 - v1, 1.0 - v0);
    }
    if rot == 3.0 {
        return vec4<f32>(1.0 - v1, 1.0 - v0, u0, u1);
    }
    return vec4<f32>(u0, u1, v0, v1);
}

/// The "view dimmed crop area" overlay sample: `inside` is true when the
/// fragment lies on the full uncropped frame (laid out over the SAME contain-fit
/// box as the base view, so the mask is always exactly the size of the image),
/// and `dim` is 1.0 when it falls OUTSIDE the crop rectangle (in that layout) so
/// [`fs_main`] can darken it. Reuses the same zoom/pan as the base view; the
/// overlay never re-sizes or re-zooms the image. Any part of the full frame
/// beyond the shader region is clipped by the render-pass viewport.
struct OverlaySample {
    uv: vec2<f32>,
    inside: bool,
    dim: f32,
}

fn overlay_sample(frag: vec2<f32>) -> OverlaySample {
    let rot = uniforms.rot;
    // Contain both axes of the CROPPED frame as DISPLAYED (rotated) — the same
    // contain-fit box the base view uses — so the dim overlay is always the
    // same size as the image and covers it exactly, whatever the crop's aspect.
    // (The overlay still samples the FULL texture over that box; the crop rect
    // and dim logic are expressed in full-frame fractions, unchanged.) The
    // crop-mode minimum padding (`uniforms.pad`) shrinks the available box on
    // every side so the overlay's white margin matches the base view's.
    let dims = rotated_dims(uniforms.crop_w, uniforms.crop_h, rot);
    let avail_w = max(uniforms.sc_w - 2.0 * uniforms.pad, 1.0);
    let avail_h = max(uniforms.sc_h - 2.0 * uniforms.pad, 1.0);
    let contain = min(avail_w / dims.x, avail_h / dims.y);
    let scale = contain * exp2(uniforms.zoom - 1.0);
    let rw = dims.x * scale;
    let rh = dims.y * scale;
    let box_x = uniforms.sc_x + (uniforms.sc_w - rw) * 0.5 + uniforms.pan_x;
    let box_y = uniforms.sc_y + (uniforms.sc_h - rh) * 0.5 + uniforms.pan_y;
    let rel = vec2<f32>(
        (frag.x - box_x) / rw,
        (frag.y - box_y) / rh,
    );
    let upright = rotate_uv(rel.x, rel.y, rot);
    // The cropped sub-rectangle expressed in this rotated full-frame screen
    // layout (still axis-aligned after quarter turns; see crop_box_display).
    let u0 = uniforms.crop_l / uniforms.tex_w;
    let u1 = (uniforms.crop_l + uniforms.crop_w) / uniforms.tex_w;
    let v0 = uniforms.crop_t / uniforms.tex_h;
    let v1 = (uniforms.crop_t + uniforms.crop_h) / uniforms.tex_h;
    let box = crop_box_display(rot, u0, u1, v0, v1);
    let cx0 = box_x + box.x * rw;
    let cx1 = box_x + box.y * rw;
    let cy0 = box_y + box.z * rh;
    let cy1 = box_y + box.w * rh;
    let in_crop_rect = frag.x >= cx0 && frag.x <= cx1 && frag.y >= cy0 && frag.y <= cy1;
    let inside = in_bounds(rel);
    let dim = select(1.0, 0.0, !inside || in_crop_rect);
    return OverlaySample(upright, inside, dim);
}

fn view_uv(frag: vec2<f32>) -> ViewSample {
    let rot = uniforms.rot;
    // Contain both axes of the CROPPED frame AS DISPLAYED (rotated); an odd
    // rotation swaps which frame axis contains the widget. Trimming re-fits/
    // zooms the kept content to fill the box at zoom 1. In crop mode the
    // minimum padding (`uniforms.pad`) shrinks the available box on every side,
    // so a white margin always surrounds the image at contain fit; zooming in
    // grows the image into the padding.
    let dims = rotated_dims(uniforms.crop_w, uniforms.crop_h, rot);
    let avail_w = max(uniforms.sc_w - 2.0 * uniforms.pad, 1.0);
    let avail_h = max(uniforms.sc_h - 2.0 * uniforms.pad, 1.0);
    let contain = min(avail_w / dims.x, avail_h / dims.y);
    let scale = contain * exp2(uniforms.zoom - 1.0);
    let rw = dims.x * scale;
    let rh = dims.y * scale;
    let box_x = uniforms.sc_x + (uniforms.sc_w - rw) * 0.5 + uniforms.pan_x;
    let box_y = uniforms.sc_y + (uniforms.sc_h - rh) * 0.5 + uniforms.pan_y;
    // Position over the (rotated) crop box.
    let rel = vec2<f32>(
        (frag.x - box_x) / rw,
        (frag.y - box_y) / rh,
    );
    // Map display → EXIF-upright crop sub-rect, then into texture UV.
    let upright = rotate_uv(rel.x, rel.y, rot);
    let uv = vec2<f32>(
        uniforms.crop_l / uniforms.tex_w + upright.x * (uniforms.crop_w / uniforms.tex_w),
        uniforms.crop_t / uniforms.tex_h + upright.y * (uniforms.crop_h / uniforms.tex_h),
    );
    let inside = in_bounds(rel);
    return ViewSample(uv, inside);
}

/// Sample + tone-curve + exposure + sRGB-encode, shared by both the base
/// zoom-crop layer and the full-frame dim overlay so they match exactly.
fn shade(uv: vec2<f32>) -> f32 {
    // Sample within [0, 1] (clamped to avoid sampler wrap reads at sub-rect
    // edges when rasterizing across the contained boundary). The texture is
    // TRUE sensor-linear data for every preset: linear `[0,1]` relative to the
    // sensor white point.
    let mono_linear = textureSample(t_mono, s_mono, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0))).r;

    var v: f32;
    if uniforms.inv != 0.0 {
        // --- Film negative: EV acts on the true sensor data FIRST ---
        // The gain already carries the inverted sign (`2^-EV` from the CPU), so
        // +EV lowers the transmission, raises the density, and brightens the
        // positive — the same direction the non-inverted preset shows.
        let transmission = mono_linear * uniforms.exposure;
        // Density-space inversion (CPU twin: `film::invert_value`):
        //   value = clamp(transmission, MIN_TRANSMISSION, base)
        //   density = log10(base / value)
        //   position = clamp(density / d_max, 0, 1)
        //   positive = position^gamma
        let clamped = clamp(transmission, 1e-6, uniforms.inv_base);
        let density = -log(clamped / uniforms.inv_base) / log(10.0);
        let position = clamp(density / uniforms.inv_d_max, 0.0, 1.0);
        let positive = pow(position, uniforms.inv_gamma);
        // Density-domain region shape (CPU twin: `film::region_shape`): a toe
        // lift and a shoulder lift, each a region-local additive mask whose
        // weight vanishes at the opposite end, so the two controls are
        // independent. `K = 2` matches `film::REGION_MASK_K`.
        let toe = pow(1.0 - positive, 2.0);
        let shoulder = pow(positive, 2.0);
        v = clamp(
            positive
                + uniforms.region_shadows * toe
                + uniforms.region_highlights * shoulder,
            0.0,
            1.0,
        );
    } else {
        // --- Already-positive scan (unchanged path) ---
        // Live tone curve re-shapes the baked positive's values via the same
        // gamma-domain tone LUT: the CPU folds the contrast power (pivot at the
        // image's measured mid-gray), the shadows power (pivot at the measured
        // white point), and the highlights power (pivot at the measured shadow
        // anchor) into the LUT at build time. Identity at the defaults
        // (byte-identical render).
        let remapped = textureSample(t_tone, s_tone, vec2<f32>(pow(mono_linear, 0.4545455), 0.5)).r
            / 512.0;
        // Linear-light exposure via the Rust-computed 2^EV gain.
        v = clamp(remapped * uniforms.exposure, 0.0, 1.0);
    }
    return linear_to_srgb(v);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let frag = input.position.xy;
    let sample = view_uv(frag);
    let srgb = shade(sample.uv);

    // When the dim-overlay is on, take the fragment's color from the full-frame
    // overlay wherever it is on the uncropped frame, darkening outside the crop.
    var rgb = srgb;
    var alpha = select(0.0, 1.0, sample.inside);
    if uniforms.show_mask != 0.0 {
        let overlay = overlay_sample(frag);
        if overlay.inside {
            let overlay_srgb = shade(overlay.uv);
            // Dim (multiply down) the region outside the crop rectangle.
            let factor = 1.0 - DIM_AMOUNT * overlay.dim;
            rgb = overlay_srgb * factor;
            alpha = 1.0;
        }
    }

    // Outside both the base image rect and any overlay region (letterbox/
    // pillarbox bars at low zoom, the edges revealed while panning, or the
    // trimmed margins), output transparent (alpha=0) so the COSMIC panel
    // background shows through via `BlendState::ALPHA_BLENDING`.
    return vec4<f32>(rgb, rgb, rgb, alpha);
}