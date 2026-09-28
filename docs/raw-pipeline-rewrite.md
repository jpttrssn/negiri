# RAW pipeline rewrite: film-only, density-domain, pointwise

Status: in progress. Scope: `src/pipeline.rs`, `src/shader.rs`, `src/film.rs`
(plus manifest/UI fields and a histogram widget). The app shell, library, crop,
export, and keyboard model are retained.

## Goal

Replace the accumulated two-implementation tone machinery (GPU LUT + CPU pivoted
powers, film presets, region masks, unsharp) with a small, single-sourced,
**pointwise** develop function operating in **optical density**. Film negatives
only; color film must remain possible later.

## Requirements

- **R1** Decode a color-CFA RAW of a B&W negative; reconstruct monochrome
  luminance with per-class cast neutralization (kept for color-readiness).
- **R2** Invert in optical density using a measured clear-film base (from a
  required calibration frame) and a per-frame density range.
- **R3** Five develop controls, all in density space, pointwise:
  **Exposure, Contrast, Black, White, Midtone pivot**.
- **R4** A calibration frame is **required** per roll; its measured base pins the
  black anchor for all frames (adjustable per frame; "reset to calibration
  frame" resets black only).
- **R5** Persist per-frame resolved `{ base, white, pivot_offset, contrast,
  exposure }` + crop/rotation; per-roll calibration frame id + its measured base.
- **R6** Developed histogram + full-`T(p)` curve overlay at the top of the detail
  drawer.
- **R7** `detail == export` at the tone level (single-sourced pointwise `T`,
  parity-tested).
- **R8** Export full-res positives + existing date/EXIF naming.

Non-functional: pointwise develop (no spatial stages); no LUT; no film presets;
no per-frame auto-base heuristic; validate direct GPU evaluation performance
before any LUT fallback.

## The develop function `T` (density space)

1. `density = log10(base / (transmission · 2^-EV))` — Exposure = density offset.
2. Normalize with the anchors: `x = (density − black) / (white − black)`.
3. Contrast: monotone power about the pivot; pivot = window midpoint + offset
   (default offset 0 ⇒ identity when contrast = 1).
4. Clamp → `[0,1]` positive, then sRGB for display.

### Slider semantics (right = increase/brighten)

| Control | Rightward drag | Density effect |
|---|---|---|
| Exposure | brightens | `+` density offset |
| Contrast | more contrast | steeper `T` |
| Black | raise black point (darken/clip darks) | black density anchor `+` |
| White | raise white point (brighten/clip brights) | white density anchor `+` |
| Midtone pivot | bend toward brights | pivot offset `+` from midpoint |

Each direction gets a unit test asserting the monotone effect, so a sign cannot
silently flip.

## Architecture (B2: shared pointwise formula, GPU preview + CPU bakes)

```
RAW (color CFA)
  → decode (rawloader)
  → mono reconstruction (CPU, once): normalize → crop → per-class base/gain → luminance
  → pointwise develop  T: density inversion + Exposure/Contrast/Black/White/Pivot
  → sRGB
  → display: WGSL fragment shader, live uniforms
  → bakes:   identical Rust T for grid/thumbnails/export
  → histogram: adaptive-stride subsample pass → canvas widget (drawer top)
```

## Deletions

`curve_remap`, `tone_model`, `build_tone_lut`, the `t_tone` LUT texture/bindings,
`tone_version`, `tone_anchors`/`film_anchor_fractiles`/`film_pivots_at_gain`,
`SHADOW_PIVOT_FLOOR`, `MonoStock`/`FILM_CHOICES`/`ACTIVE_STOCK`, film presets,
the `ToneEdit` power/lift/region-mask machinery, `film::region_shape` /
`effective_d_max`, and `unsharp_mask`. `film.rs` shrinks to base measurement +
class gains + the density `T`.

## Retained

`flatten_bayer` (CFA-aware reconstruction), `class_gains`,
`downsample_thumbnail`, `resize_area`, crop/rotate/orient, export naming/EXIF,
manifest structure (new fields).

## Histogram + curve

- Live, adaptive-stride subsample (~`BASE_SAMPLE_TARGET` samples) of the decoded
  mono, develop `T` applied, binned, computed on the blocking pool with a
  "latest wins" serial guard.
- Post-sRGB display bins; curve overlay plots `srgb(T(p))` so both share the
  `[0,1]` display axis. Drawn with `iced::widget::canvas` at the drawer top.

## Execution order (commit after each step)

1. Benchmark direct WGSL `T` at native resolution (LUT-vs-direct decision).
2. `film.rs`: shrink to base measurement + class gains; add the Rust density `T`.
3. `shader.rs`: replace the LUT/pivot machinery with the WGSL `T` + uniforms.
4. `pipeline.rs`: thread the new `T`; remove `unsharp_mask`.
5. Manifest: new fields + anchor-resolution logic (calibration base + trims + reset).
6. Histogram + canvas widget.
7. Tests: `T` parity, identity, slider direction, anchor resolution, binning.
8. `just check` + `cargo test --locked`; manual visual check.
