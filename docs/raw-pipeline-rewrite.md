# RAW pipeline rewrite: film-only, density-domain, pointwise

Status: steps 1–5 done (film-only path landed; histogram widget remains).
Scope: `src/pipeline.rs`, `src/shader.rs`, `src/film.rs` (plus manifest/UI
fields and a histogram widget). The app shell, library, crop, export, and
keyboard model are retained.

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

## Step 1 result: no LUT

Micro-benchmark of the develop function `T` (film density path, ~5 transcendentals):
single-threaded CPU ~18.7 ns/pixel (≈1.25 s for an 8192² frame). The CPU bake is a
background batch and parallelizable. The GPU already evaluates a `log`+`pow`
per fragment in the current film branch, so direct pointwise evaluation is the
same performance class as what already ships. **Decision: no tone LUT** — evaluate
`T` directly in WGSL and in Rust, deleting the LUT/tolerance/parity machinery.

## Steps 3–5 result: the film-only density path

- **`film::Develop`** (base-relative `density = log10(base / (transmission·2^-EV))`)
  is the single pointwise `T`: `base`, `black`, `white`, `contrast`, `pivot_offset`,
  `exposure_ev`. `Develop::apply` is mirrored exactly by the WGSL `develop()`
  (`develop_matches_wgsl_transcription` parity test).
- **`shader.rs`** carries a `Develop` (no more pivots/LUT/`tone_version`); the
  bind group is texture + sampler + uniform, and the WGSL `dev_base/black/white/
  contrast/pivot` uniforms drive the per-fragment develop.
- **`pipeline.rs`** drops `unsharp_mask`/`blur_121`/`UNSHARP_AMOUNT`, the pivot
  machinery, and the two-path `render_tail`; `bake_develop` is the one CPU tail.
  `decode_raw_detail` returns only the true sensor-linear mono + geometry.
- **`edit_manifest.rs`** stores `{exposure_ticks, contrast_lift_ticks, black_ticks,
  white_ticks, pivot_ticks}` (schema v1, legacy keys re-edited, no migration) plus
  the roll-level `base` + `calibration_frame`.
- **UI**: the editing panel is Exposure / Contrast / Black / White / Midtone;
  the roll-info drawer's film-preset and base-mode dropdowns are gone. A roll's
  calibration frame defaults to its first frame and is measured on roll open;
  Edit → "Calibrate from this frame" re-designates it.

Deviations from the original R5 field list: `black` is stored per frame too
(it is an independent density anchor, not derivable from `base`), and the
per-frame `base` is resolved from the roll-level calibration (no per-frame base
override yet). Film presets / `MonoStock` / `BaseMode` are removed outright.
