# Tone-curve isolation redesign

Status: proposed. Supersedes the "three-pivoted-power" tone model's shape, not its
anchor measurement or its LUT-based CPU/GPU single-sourcing.

## Problem

The three tone controls (`contrast`, `highlights`, `shadows`) currently collapse
into a single power law

```
T(p) = clamp(ratio · p^exp, 0, 1)
ratio = s^(1-kh) · w^(kh·(1-ks)) · m^(kh·ks·(1-kc))
exp   = kc · ks · kh
```

(`shader::curve_remap`). A power law is a global shape: each pivot only pins one
point, and the slope change applies over the whole input. Measured with
film-ish anchors `shadow = 0.15`, `mid = 0.40`, `white = 0.92`, one stop of lift:

- **Shadows +1 stop** (power `0.5`, white-pivoted):
  `p=0.05 → 0.212` (4.2×), `p=0.15 → 0.367` (2.45×), `p=0.40 → 0.600` (1.5×),
  `p=0.92 → 0.92`, `p=1.0 → 0.95`.
- **Highlights +1 stop** (power `2.0`, shadow-pivoted):
  `p=0.05 → 0.017` (0.33×, deep-shadow crush), `p=0.15 → 0.15`,
  `p=0.40 → 1.07` (clipped), `p=0.70` (clipped).

So a shadows move drags midtones and pulls pure white down, and a highlights move
crushes the black end and clips mids to white. Contrast additionally multiplies
the other two (`exp = kc·ks·kh`), so the controls are not independent.

This is the deferred "region-localizing curve shape" candidate (item 8 of
`NOTES.md`); the label/direction swap has already been fixed.

## Decision

Replace the composed-power model with a **monotone cubic Hermite curve through
the measured anchors**, with Contrast, Highlights, and Shadows each setting one
knot tangent.

The WGSL is **not touched**: it already samples an arbitrary 1D LUT
(`exposure.wgsl` `textureSample(t_tone, s_tone, vec2(pow(positive, 1/G), 0.5))`).
Any pointwise monotone curve therefore preserves the GPU architecture, the LUT
format (2048×1 `R16Float`, gamma-domain packed, ×512 pre-scale), the exposure
order, and grid == detail == export.

## Design

### Knots and tangents

Knots are strictly increasing, with values on the diagonal so every anchor is
pinned exactly:

```
x = [0, s, m, w, 1],    y_i = x_i
```

where `s`, `m`, `w` are the existing measured anchors (10th / 50th / 98th
percentile of the displayed positive; for film negatives derived per-EV by
`film_pivots_at_gain`, unchanged).

Tangents `T = [t0, ts, tm, th, t4]`, identity `1.0`:

| Slider     | Tangent         | Knot  |
|------------|-----------------|-------|
| Shadows    | `ts`            | `s`   |
| Highlights | `th`            | `w`   |
| (mid)      | `tm = 1` (fixed) | `m`   |
| (endpoint) | `t0 = 1`        | `0`   |
| (endpoint) | `t4 = 1`        | `1`   |

`t0 = t4 = 1` keeps the extreme ends linear.

**Contrast stays an outer global power, not a tangent.** The Hermite `H` carries
only the regional Shadows/Highlights shape; the final curve is

```
T(p) = C(H(p)),   C(q) = mid^(1-kc)·q^kc
```

with `kc` the existing mid-pivoted contrast power. This keeps contrast's full
±3-stop range (`0.125..8.0`) decoupled from the tangent monotonicity bound, and
it is why the mid tangent is fixed at `1`.

### Control → tangent

Reuse the existing lift maps, which already produce the right direction:

- Shadows lift `L` → `ts = 2^-L` (a lift bows the lower segment above the chord)
- Highlights lift `L` → `th = 2^+L`
- Contrast is the outer power `kc` (unchanged semantics, now log2-lift UI)

### Monotonicity by construction

Because the knots lie on the diagonal, every secant is exactly `1`
(`(y_{i+1}-y_i)/(x_{i+1}-x_i) = 1`). Fritsch–Carlson guarantees monotonicity of
a cubic Hermite segment when `alpha^2 + beta^2 <= 9`, with `alpha = m_i/Δ` and
`beta = m_{i+1}/Δ`. With `Δ = 1` and the mid/endpoint tangents fixed at `1`,
each adjacent tangent pair contains at most **one** variable tangent, so a
region tangent can be clamped independently up to `sqrt(8) ≈ 2.83` (worst case
`1^2 + 2.83^2 = 9 - ε`) with no neighbor coupling:

- the curve is monotone for **all** slider combinations, and
- no neighbor coupling is introduced (each control's tangent stands alone).

Full Fritsch–Carlson (which can also rescale a tangent shared by two segments)
is rejected for the same reason: it would let a Shadows change leak across `m`
into the highlights.

`C` is a monotone power, so monotonicity of `H` carries through to `T`.

### Isolation (exact, from the knot structure)

- `H(0) = 0`, `H(s) = s`, `H(m) = m`, `H(w) = w`, `H(1) = 1` for every setting.
- The segments above `m` depend only on `(m,m)`, `(w,w)`, `(1,1)`, `th`, `t4`
  ⇒ **Shadows never changes `H(p)` for `p >= m`.**
- The segments below `m` are independent of `th` ⇒ **Highlights never changes
  `H(p)` for `p <= m`.**
- Contrast is a separate outer power applied after `H`; it deliberately touches
  the whole range (a global control), and so is not "isolated" by design.
- Endpoints are pinned and `H` is monotone, so `H ∈ [0, 1]`; `C` is a monotone
  power mapping `[0,1]→[0,1]`, so `T ∈ [0, 1]` with **no interior clipping** — a
  highlight lift can no longer blow mids to white.

### Evaluation

Given segment `i` containing `p`:

```
h = x[i+1] - x[i]
t = (p - x[i]) / h
y = (2t^3 - 3t^2 + 1)·x[i]
  + (t^3 - 2t^2 + t)·h·m[i]
  + (-2t^3 + 3t^2)·x[i+1]
  + (t^3 - t^2)·h·m[i+1]
```

No `powf`; this is cheaper than the current `build_tone_lut` (2048 powf).

### Degenerate anchors

- Keep the `SHADOW_PIVOT_FLOOR` floor for `s`.
- Enforce a minimum knot separation (`s <= m - eps`, `w >= m + eps`, `s >= eps`,
  `w <= 1 - eps`); fall back to fixed knots (`0.15`, `0.4`, `0.9`) for collapsed
  histograms.

## What changes

- `src/shader.rs`
  - Replace `curve_remap` / `tone_model` with a `ToneCurve` holding the Hermite
    knots/tangents **plus** the contrast power, and `ToneCurve::eval` applying
    `C(H(p))`.
  - `build_tone_lut` and `apply_curve` keep their current signatures, so
    `pipeline.rs` call sites are untouched.
  - `DetailProgram`: `highlights` / `shadows` become region-lift tangents;
    `contrast` stays the outer mid-pivoted power.
  - Rewrite the pivot/identity/LUT-parity tests (see below).
- `src/app.rs`
  - Region tangent clamp: independently up to `sqrt(8) ≈ 2.83` (was `[1/3, 2.0]`
    under a shared-mid design).
  - `contrast_lift` / `contrast_power_for_lift` (already added with the ±3-stop
    log2 UI) drive the outer power, unchanged.
  - `shadow_lift` / `highlight_lift` / `shadow_power_for_lift` /
    `highlight_power_for_lift` are unchanged.
- `src/ui.rs` / `src/edit_manifest.rs`
  - Unchanged by the redesign: contrast is already a ±3-stop log2 control
    (`contrast_lift`, `curve_contrast_lift_ticks`) and the region sliders are
    already ±2-stop lifts.
- `src/pipeline.rs`, `src/shader/exposure.wgsl`
  - Unchanged.
- `NOTES.md`
  - Update the three-pivoted-power ADR, the tone-model description, and item 8
    (mark isolation shipped; drop candidate (ii)).

## Tests / acceptance criteria

- **Identity**: all tangents `1` and contrast `1` ⇒ `T(p) = p` exactly (a
  Hermite with all tangents equal to the common secant collapses to the line).
- **Pinning**: `H(0)=0`, `H(s)=s`, `H(m)=m`, `H(w)=w`, `H(1)=1` across
  randomized anchors and strengths.
- **Isolation**: changing the Shadows tangent leaves `H(p)` bit-identical for
  `p >= m`; changing the Highlights tangent leaves `H(p)` bit-identical for
  `p <= m`.
- **Monotonicity**: a dense value grid × randomized anchors × slider extremes is
  non-decreasing.
- **No clipping**: `T(p) ∈ [0, 1]` for all inputs.
- **LUT parity**: `tone_lut_reproduces_tone_model_within_tolerance` (≤1e-3) and
  `cpu_bake_matches_gpu_lut_simulation_on_random_frames` (≤5e-4), unchanged in
  spirit, exercised against the new curve.
- `just check` and `cargo test --locked` must pass.

## Calibration

The sliders have no numeric readout, so "stops" in comments is only a label.
After implementing, measure `ΔT` per slider step on a real film negative and a
real positive RAW and choose the final region tangent clamp up to `sqrt(8) ≈
2.83` so the arms feel comparable. Contrast is independent (`contrast_lift`) and
already calibrated.

## Risks / open questions

- **Region range**: the independent tangent clamp caps at `sqrt(8) ≈ 2.83`
  (i.e. ~±1.5 stops of tangent swing); beyond that, exact isolation would have to
  give way to full Fritsch–Carlson. Contrast is unaffected (outer power, ±3
  stops).
- Every persisted edit re-renders once (acceptable pre-release).
- Anchors remain the current percentile definition; changing it later changes
  every edit's look.
- The per-slider `ΔT` magnitude differs from the power model, so keyboard step
  sizes (`EDIT_STEP_CURVE` / `EDIT_NUDGE_CURVE`) may need re-tuning during
  calibration.
