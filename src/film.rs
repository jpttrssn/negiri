// SPDX-License-Identifier: GPL-3.0-or-later

//! Film-negative develop: density-domain tone mapping and clear-film base
//! measurement. (Legacy stock/preset machinery is being retired; see
//! `docs/raw-pipeline-rewrite.md`.)

/// Samples at or below this transmission encode maximum density. Shared by the
/// inversion and the develop function.
// TODO(raw-pipeline-rewrite step 4): wired into `pipeline`/`shader` then.
#[allow(dead_code)]
const MIN_TRANSMISSION_DEV: f32 = 1e-6;

/// The pointwise develop function for a film negative: a monotone map from a
/// sensor-linear transmission to a display positive, evaluated entirely in
/// optical **density**.
///
/// Pipeline (all pointwise):
/// 1. Exposure is a density offset: a `2^-EV` gain on the transmission.
/// 2. `density = log10(base / transmission)` — clear film (transmission == base)
///    is density 0.
/// 3. Black/White normalize the density window: `x = (density - black) /
///    (white - black)`, clamped to `[0,1]`.
/// 4. Contrast bends `x` about the midtone pivot with the two-sided power
///    (`x/p)^k` / `1 - (1-p)·((1-x)/(1-p))^k`, identity at `k == 1` for every
///    pivot. The pivot is a signed offset from the window midpoint (default 0).
///
/// Every control is independent and monotone; each slider's rightward direction
/// is pinned by the tests (`slider_directions_*`). The identity (`exposure=0`,
/// `contrast=1`, `pivot=0`, and `black`/`white` == the measured anchors) is
/// exactly the plain normalized density, so untouched values render unchanged.
// TODO(raw-pipeline-rewrite step 4): wired into `pipeline`/`shader` then.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Develop {
    /// Exposure in EV; a density offset (positive brightens the positive).
    pub exposure_ev: f32,
    /// Density mapped to output black (`density == black` ⇒ 0).
    pub black: f32,
    /// Density mapped to output white (`density == white` ⇒ 1).
    pub white: f32,
    /// Contrast power about [`Self::pivot`]; `1.0` = identity.
    pub contrast: f32,
    /// Signed offset from the `[black, white]` midpoint the contrast bends
    /// around. `0.0` = identity.
    pub pivot_offset: f32,
}

#[allow(dead_code)]
impl Develop {
    /// The simplest valid develop for the given anchors: identity contrast and
    /// pivot, no exposure offset.
    #[must_use]
    pub fn with_anchors(black: f32, white: f32) -> Self {
        Self {
            exposure_ev: 0.0,
            black,
            white,
            contrast: 1.0,
            pivot_offset: 0.0,
        }
    }

    /// Applies the develop to a sensor-linear transmission, returning a
    /// `[0,1]` positive (pre-sRGB).
    ///
    /// Mirrored exactly by the WGSL `develop()`; keep the two in lockstep (the
    /// `develop_matches_wgsl` parity test uses a Rust transcription of the
    /// shader's expression).
    #[must_use]
    pub fn apply(&self, transmission: f32) -> f32 {
        let gain = (-self.exposure_ev).exp2();
        let value = (transmission * gain).clamp(MIN_TRANSMISSION_DEV, 1.0);
        let density = f32::log10(1.0 / value); // base folded into black/white
        self.apply_density(density)
    }

    /// Applies the develop to a density already measured relative to the clear
    /// film (`density = log10(base / transmission)`), which is what the shader
    /// computes so it can clamp at `base` before the log.
    #[must_use]
    pub fn apply_density(&self, density: f32) -> f32 {
        let span = (self.white - self.black).abs().max(1e-4);
        let x = ((density - self.black) / span).clamp(0.0, 1.0);
        let pivot = (0.5 + self.pivot_offset).clamp(1e-4, 1.0 - 1e-4);
        let shaped = if x <= pivot {
            pivot * (x / pivot).powf(self.contrast)
        } else {
            1.0 - (1.0 - pivot) * ((1.0 - x) / (1.0 - pivot)).powf(self.contrast)
        };
        shaped.clamp(0.0, 1.0)
    }
}

/// A developed monochrome film stock's scan-response profile.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MonoStock {
    /// Display name of the stock (shown in the film-preset pickers).
    pub name: &'static str,
    /// Scanner-linear transmission of unexposed film base + fog; the positive's
    /// black point (the clearest film areas). A preset fallback: a roll may
    /// override it with a calibrated reference or per-frame measurement (see
    /// [`resolve_base`]).
    pub base: f32,
    /// Usable density range of the film above the base; the densest useful
    /// negative area maps to the positive's white point.
    pub d_max: f32,
    /// Tone-curve exponent applied to normalized density; below 1 lifts
    /// shadows.
    pub gamma: f32,
}

/// The stock assumed for all thumbnails during the inversion proof of concept.
///
/// `base` was calibrated against real HP5+ scans (brightest plateau across
/// horizontal bands of `_MG_0828.CR2`, see NOTES.md); `d_max`/`gamma` start
/// from published HP5+ curve values pending visual tuning.
pub const ACTIVE_STOCK: MonoStock = MonoStock {
    name: "Ilford HP5+",
    base: 0.82,
    d_max: 2.4,
    gamma: 0.7,
};

/// Kodak Tri-X 400: the iconic rival to HP5+ — punchy, grainy, blocky shadows.
/// `base` from its heavier base fog; `d_max`/`gamma` from published curves
/// pending visual tuning.
pub const TRI_X: MonoStock = MonoStock {
    name: "Kodak Tri-X 400",
    base: 0.56,
    d_max: 2.6,
    gamma: 0.8,
};

/// Ilford FP4 Plus (125): a fine-grain slow film with smooth, gentle tonality.
pub const FP4_PLUS: MonoStock = MonoStock {
    name: "Ilford FP4 Plus",
    base: 0.74,
    d_max: 2.2,
    gamma: 0.7,
};

/// Kodak T-Max 400: T-grain, sharp with a steep, high microcontrast curve.
pub const TMAX_400: MonoStock = MonoStock {
    name: "Kodak T-Max 400",
    base: 0.63,
    d_max: 2.5,
    gamma: 0.85,
};

/// Fomapan 100: budget stock with heavy base fog and a soft, pronounced
/// shoulder (low `gamma` keeps shadows milky).
pub const FOMAPAN_100: MonoStock = MonoStock {
    name: "Fomapan 100",
    base: 0.56,
    d_max: 2.0,
    gamma: 0.6,
};

/// Ilford Delta 400: modern T-grain, smooth like HP5+ but finer.
pub const DELTA_400: MonoStock = MonoStock {
    name: "Ilford Delta 400",
    base: 0.66,
    d_max: 2.3,
    gamma: 0.72,
};

/// Fomapan 400: the foggiest budget stock here; wide-latitude, soft contrast.
pub const FOMAPAN_400: MonoStock = MonoStock {
    name: "Fomapan 400",
    base: 0.50,
    d_max: 2.1,
    gamma: 0.62,
};

/// Kentmere 400: Ilford's budget 400, close to HP5+ but a touch cleaner.
pub const KENTMERE_400: MonoStock = MonoStock {
    name: "Kentmere 400",
    base: 0.76,
    d_max: 2.3,
    gamma: 0.7,
};

/// The generic B&W film profile: a sane neutral fallback for rolls whose stock
/// is unknown or not in the preset list. Its values mirror an average B&W film
/// (HP5+); the auto-base strategies rescue the black point per roll anyway, so
/// the exact `base` matters less than `d_max`/`gamma` character.
pub const GENERIC_STOCK: MonoStock = MonoStock {
    name: "Generic",
    base: 0.80,
    d_max: 2.4,
    gamma: 0.7,
};

/// Every film-preset choice, in dropdown order: `None` (no inversion), the
/// generic B&W profile, then the named stocks alphabetically. The single source
/// of truth the film-preset pickers consume, so the roll-info drawer and the
/// add-roll dialog can never drift. Order MUST match [`FilmPreset::index`].
pub const FILM_CHOICES: [FilmPreset; 10] = [
    FilmPreset::None,
    FilmPreset::Generic,
    FilmPreset::Fomapan100,
    FilmPreset::Fomapan400,
    FilmPreset::Delta400,
    FilmPreset::Fp4Plus,
    FilmPreset::Hp5Plus,
    FilmPreset::Kentmere400,
    FilmPreset::TMax400,
    FilmPreset::TriX,
];

/// The film-inversion preset a roll's frames are rendered with: which
/// [`MonoStock`] profile inverts the negatives, or `None` for already-positive
/// scans (regular RAWs) that must NOT be inverted.
///
/// The base strategy (preset base / auto per frame / auto selected frame) is a
/// separate roll-level choice ([`BaseMode`]), not part of the film picker. The
/// dropdown lists **None, Generic, then the named stocks alphabetically** (see
/// [`FILM_CHOICES`]). The default is [`FilmPreset::None`], so a roll with no
/// recorded preset renders as a regular (non-inverted) scan.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum FilmPreset {
    /// No inversion: the scan is treated as an already-positive image (a
    /// regular RAW). The pipeline normalizes it in place of inverting a
    /// negative.
    #[default]
    None,
    /// Invert with the generic B&W profile — the fallback for films not in the
    /// preset list.
    Generic,
    /// Invert with the Fomapan 100 profile.
    Fomapan100,
    /// Invert with the Fomapan 400 profile.
    Fomapan400,
    /// Invert with the Ilford Delta 400 profile.
    Delta400,
    /// Invert with the Ilford FP4 Plus profile.
    Fp4Plus,
    /// Invert with the Ilford HP5+ profile, using its preset base.
    Hp5Plus,
    /// Invert with the Kentmere 400 profile.
    Kentmere400,
    /// Invert with the Kodak T-Max 400 profile.
    TMax400,
    /// Invert with the Kodak Tri-X 400 profile.
    TriX,
}

impl FilmPreset {
    /// The inversion profile for a chosen preset: [`None`] (no inversion)
    /// carries no profile; `Generic` and each stock preset carry their own.
    #[must_use]
    pub const fn stock(self) -> Option<MonoStock> {
        match self {
            Self::None => None,
            Self::Generic => Some(GENERIC_STOCK),
            Self::Fomapan100 => Some(FOMAPAN_100),
            Self::Fomapan400 => Some(FOMAPAN_400),
            Self::Delta400 => Some(DELTA_400),
            Self::Fp4Plus => Some(FP4_PLUS),
            Self::Hp5Plus => Some(ACTIVE_STOCK),
            Self::Kentmere400 => Some(KENTMERE_400),
            Self::TMax400 => Some(TMAX_400),
            Self::TriX => Some(TRI_X),
        }
    }

    /// Whether this preset marks the scan as a negative to be density-inverted
    /// (`None` = already-positive scan, no inversion).
    ///
    /// The pipeline consumes this: an inverted preset applies the exposure gain
    /// to the true sensor-linear data BEFORE the inversion and flips the EV
    /// sign at that point, so the user-facing controls (+EV = brighter) stay
    /// identical across presets.
    #[must_use]
    #[allow(dead_code)] // kept as the semantic inverse of `stock()`; used in tests
    pub const fn is_inverted(self) -> bool {
        self.stock().is_some()
    }

    /// The stable dialog and manifest storage key for this preset. `None` is
    /// the zero-value default, represented by the key's absence; only a
    /// non-default preset is ever written.
    #[must_use]
    pub const fn choice_key(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Generic => "generic",
            Self::Fomapan100 => "fomapan100",
            Self::Fomapan400 => "fomapan400",
            Self::Delta400 => "delta400",
            Self::Fp4Plus => "fp4",
            Self::Hp5Plus => "hp5",
            Self::Kentmere400 => "kentmere400",
            Self::TMax400 => "tmax400",
            Self::TriX => "tri-x",
        }
    }

    /// Resolves a stored choice key into a preset; the missing/absent key and
    /// unknown values fall back to the default [`FilmPreset::None`].
    #[must_use]
    pub fn from_key(key: &str) -> Self {
        match key {
            "generic" => Self::Generic,
            "fomapan100" => Self::Fomapan100,
            "fomapan400" => Self::Fomapan400,
            "delta400" => Self::Delta400,
            "fp4" => Self::Fp4Plus,
            "hp5" => Self::Hp5Plus,
            "kentmere400" => Self::Kentmere400,
            "tmax400" => Self::TMax400,
            "tri-x" => Self::TriX,
            _ => Self::None,
        }
    }

    /// The dropdown index ordering (None first, matching the default, then
    /// Generic, then the stocks alphabetically in [`FILM_CHOICES`] order).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::None => 0,
            Self::Generic => 1,
            Self::Fomapan100 => 2,
            Self::Fomapan400 => 3,
            Self::Delta400 => 4,
            Self::Fp4Plus => 5,
            Self::Hp5Plus => 6,
            Self::Kentmere400 => 7,
            Self::TMax400 => 8,
            Self::TriX => 9,
        }
    }

    /// Resolves a dropdown index (see [`Self::index`]) back into a preset;
    /// any out-of-range index falls back to the default [`FilmPreset::None`].
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        match index {
            1 => Self::Generic,
            2 => Self::Fomapan100,
            3 => Self::Fomapan400,
            4 => Self::Delta400,
            5 => Self::Fp4Plus,
            6 => Self::Hp5Plus,
            7 => Self::Kentmere400,
            8 => Self::TMax400,
            9 => Self::TriX,
            _ => Self::None,
        }
    }
}

/// How a roll's black point (the film's `base`) is resolved. Orthogonal to the
/// [`FilmPreset`] film choice: the film supplies `d_max`/`gamma` (character),
/// this decides where the black point comes from.
///
/// The default is [`BaseMode::Preset`]: the film's own preset base, until the
/// user opts into measurement.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum BaseMode {
    /// Use the chosen film's preset base (nothing measured).
    #[default]
    Preset,
    /// Measure each frame's own clear-film plateau as its base.
    AutoPerFrame,
    /// Measure one designated calibration frame's base and use it for the whole
    /// roll (defaults to the roll's first frame). See
    /// [`edit_manifest::RollManifest`]'s `calibration_frame`/`base` fields.
    AutoSelectedFrame,
}

impl BaseMode {
    /// The stable dialog and manifest storage key for this mode. `Preset` is
    /// the zero-value default, represented by the key's absence.
    #[must_use]
    pub const fn choice_key(self) -> &'static str {
        match self {
            Self::Preset => "preset",
            Self::AutoPerFrame => "auto-per-frame",
            Self::AutoSelectedFrame => "auto-selected-frame",
        }
    }

    /// Resolves a stored choice key into a mode; the missing/absent key and
    /// unknown values fall back to the default [`BaseMode::Preset`].
    #[must_use]
    pub fn from_key(key: &str) -> Self {
        match key {
            "auto-per-frame" => Self::AutoPerFrame,
            "auto-selected-frame" => Self::AutoSelectedFrame,
            _ => Self::Preset,
        }
    }

    /// The dropdown index ordering (Preset first, matching the default).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Preset => 0,
            Self::AutoPerFrame => 1,
            Self::AutoSelectedFrame => 2,
        }
    }

    /// Resolves a dropdown index (see [`Self::index`]) back into a mode; any
    /// out-of-range index falls back to the default [`BaseMode::Preset`].
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        match index {
            1 => Self::AutoPerFrame,
            2 => Self::AutoSelectedFrame,
            _ => Self::Preset,
        }
    }

    /// Whether this mode measures the base at all (either per frame or from the
    /// designated calibration frame).
    #[must_use]
    #[allow(dead_code)] // kept as a semantic predicate; used in tests
    pub const fn is_auto(self) -> bool {
        matches!(self, Self::AutoPerFrame | Self::AutoSelectedFrame)
    }
}

/// Samples at or below this transmission encode maximum density.
const MIN_TRANSMISSION: f32 = 1e-6;

/// Smallest usable density range accepted by [`invert_value_graded`], guarding
/// the `density / d_max` division against a degenerate (near-zero) window.
const MIN_D_MAX: f32 = 0.01;

/// Measured channel bases below this are implausible — they indicate frames
/// without any measurable clear film — and fall back to [`MonoStock::base`].
pub const MIN_PLAUSIBLE_BASE: f32 = 0.1;

/// Maps one scanned transmission to its positive tone value, anchored on its
/// channel's clear-film transmission: optical density relative to that anchor,
/// positioned within the stock's usable density range, then run through the
/// contrast curve.
///
/// `base` is the clear-film transmission ([`MonoStock::base`] or a per-frame
/// measurement); `value == base` prints black, the densest useful area prints
/// white.
pub fn invert_value(transmission: f32, base: f32, stock: &MonoStock) -> f32 {
    invert_value_graded(transmission, base, stock.d_max, stock.gamma)
}

/// The grade-aware inversion: the same density-space mapping as
/// [`invert_value`] but with an explicit usable density range and gamma, so a
/// user Contrast (which scales `d_max` via [`effective_d_max`]) flows into the
/// inversion without mutating the stock profile.
///
/// `value == base` prints black, `density == d_max_eff` prints white, and the
/// normalized position is raised to `gamma`.
#[must_use]
pub fn invert_value_graded(
    transmission: f32,
    base: f32,
    d_max_eff: f32,
    gamma: f32,
) -> f32 {
    let value = transmission.clamp(MIN_TRANSMISSION, base);
    // Density relative to the clear-film anchor.
    let density = f32::log10(base / value);
    // Position within the film's usable density range, mapped through a
    // contrast curve onto the full positive range: the clearest film
    // areas print black, the densest useful areas print white.
    let position = (density / d_max_eff.max(MIN_D_MAX)).clamp(0.0, 1.0);

    position.powf(gamma)
}

/// Inverts interleaved linear RGB scanned from a monochrome negative into a
/// positive, working in optical-density space.
///
/// `bases` holds each channel's clear-film transmission, anchoring the black
/// point per channel so capture casts (light table, sensor response) are
/// neutralized.
///
/// Unused while scans collapse to luminance before inversion; kept with its
/// tests as the per-channel path for future color stocks.
#[allow(dead_code)]
pub fn invert_mono(rgb: &mut [f32], stock: &MonoStock, bases: [f32; 3]) {
    for pixel in rgb.as_chunks_mut::<3>().0 {
        for (slot, base) in pixel.iter_mut().zip(bases) {
            *slot = invert_value(*slot, base, stock);
        }
    }
}

/// Inverts linear samples scanned from a monochrome negative into a positive,
/// working in optical-density space.
///
/// `base` is the scan's clear-film transmission and anchors the black point;
/// collapsing the capture to luminance beforehand removes any cast between
/// channels outright.
///
/// Kept as the simple (ungraded) entry point; the pipeline inlines the graded
/// form (`invert_value_graded` + `region_shape`) so a user Contrast and the
/// region lifts flow through. Used by the film unit tests.
#[allow(dead_code)]
pub fn invert_gray(samples: &mut [f32], stock: &MonoStock, base: f32) {
    for slot in samples {
        *slot = invert_value(*slot, base, stock);
    }
}

/// Estimates the base+fog transmission from normalized linear samples of an
/// exposure showing only clear film, e.g. a photographed blank leader. Takes a
/// high percentile so dust and sensor outliers are rejected.
///
/// Returns `None` for inputs without finite samples.
pub fn measure_base(samples: &[f32]) -> Option<f32> {
    let mut sorted: Vec<f32> = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();

    if sorted.is_empty() {
        return None;
    }

    sorted.sort_by(f32::total_cmp);
    Some(sorted[sorted.len() * 95 / 100])
}

/// The roll-level base-resolution state threaded from the manifest into every
/// decode and bake, so the detail shader, grid thumbnails, and exports cannot
/// drift the film's black point between renderings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BaseConfig {
    /// An explicit per-roll calibration measured once from a blank/leader
    /// frame held in the roll manifest. Wins over everything else.
    pub calibrated: Option<f32>,
    /// Opt-in per-frame auto measurement of the clear-film anchor. When off,
    /// the frame's own `measure_base` result is never computed (sorting a full
    /// buffer is wasted work).
    pub auto: bool,
}

impl BaseConfig {
    /// Resolves the tonally-effective black point for one frame of `stock`.
    ///
    /// `measured` is the frame's own `measure_base` result — pass it only when
    /// [`BaseConfig::auto`] is on; the caller must not compute it otherwise.
    #[must_use]
    pub fn resolve(self, measured: Option<f32>, stock: &MonoStock) -> f32 {
        resolve_base(self.calibrated, self.auto, measured, stock)
    }
}

/// Resolves the tonally-effective black point for one frame from the roll's
/// calibration state, preset-first:
///
/// 1. An explicit per-roll calibration wins outright (a blank/leader frame
///    photographed once, stored in the roll manifest).
/// 2. Otherwise, the per-frame auto measurement — an explicit opt-in — refines
///    the stock's preset base, skipping it when it is implausible or absent.
/// 3. Otherwise the stock's preset base is the truth.
///
/// `measured` is the frame's own `measure_base` result when `auto` is on;
/// callers must not compute it otherwise (sorting a full buffer is wasted).
///
/// Kept pure + unit-tested so the detail decode, the thumbnail bake, and the
/// export path cannot drift the black point between renderings.
#[must_use]
pub fn resolve_base(
    calibrated: Option<f32>,
    auto: bool,
    measured: Option<f32>,
    stock: &MonoStock,
) -> f32 {
    if let Some(base) = calibrated {
        base
    } else if auto {
        measured
            .filter(|value| *value >= MIN_PLAUSIBLE_BASE)
            .unwrap_or(stock.base)
    } else {
        stock.base
    }
}

/// Measures each RGB channel's clear-film transmission from interleaved linear
/// pixels, neutralizing capture casts by anchoring every channel on its own
/// base plateau (typically found in frame gaps and margins).
///
/// Channels measuring below [`MIN_PLAUSIBLE_BASE`] fall back to the active
/// stock's constant; returns `None` only when no channel has finite samples.
///
/// Unused alongside [`invert_mono`]; kept for future color stocks.
#[allow(dead_code)]
pub fn measure_base_channels(rgb: &[f32]) -> Option<[f32; 3]> {
    let mut channels: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for [r, g, b] in rgb.as_chunks::<3>().0 {
        channels[0].push(*r);
        channels[1].push(*g);
        channels[2].push(*b);
    }

    let mut bases = [0.0_f32; 3];
    for (channel, slot) in channels.iter().zip(bases.iter_mut()) {
        *slot = measure_base(channel)?;
    }

    for slot in &mut bases {
        if *slot < MIN_PLAUSIBLE_BASE {
            *slot = ACTIVE_STOCK.base;
        }
    }

    Some(bases)
}

/// The mask sharpness `K` for the density-domain toe/shoulder lifts. `K = 2`
/// makes each region mask fall off quadratically from its end, so a shadow lift
/// concentrates on the toe and a highlight lift on the shoulder while each
/// reaches zero at the opposite end (the property that makes them independent).
pub const REGION_MASK_K: f32 = 2.0;

/// The largest magnitude accepted for a region lift (toe or shoulder), in
/// normalized-positive units. Bounds the additive mask shift so the composed
/// curve stays monotone; the UI slider's stop range maps into this window.
pub const REGION_LIFT_MAX: f32 = 0.5;

/// Normalized-positive strength per stop of region lift: a ±2-stop slider spans
/// the full ±[`REGION_LIFT_MAX`] window.
pub const REGION_LIFT_PER_STOP: f32 = 0.25;

/// Maps a region lift in stops to its bounded normalized-positive strength.
#[must_use]
pub fn region_strength(lift_stops: f32) -> f32 {
    (lift_stops * REGION_LIFT_PER_STOP).clamp(-REGION_LIFT_MAX, REGION_LIFT_MAX)
}

/// The density-window scale a user Contrast value (in stops, identity `0`)
/// applies to the stock's usable range: `d_eff = d_max · 2^-contrast`. A
/// positive contrast narrows the window (more log-density contrast), a negative
/// one widens it. Clamped to a sane range so the window never collapses or
/// explodes.
#[must_use]
pub fn effective_d_max(d_max: f32, contrast: f32) -> f32 {
    (d_max * (-contrast).exp2()).clamp(0.01, 100.0)
}

/// The density-domain tone shape applied to the normalized positive `p ∈ [0,1]`
/// (the film response after `position^gamma`): a shadow (toe) lift and a
/// highlight (shoulder) lift, each a region-local additive mask.
///
/// `T(p) = clamp(p + shifts·W_toe(p) + highlights·W_shoulder(p), 0, 1)` with
/// `W_toe(p) = (1-p)^K` (peaks at black, zero at white) and
/// `W_shoulder(p) = p^K` (peaks at white, zero at black). The zeros at the
/// opposite ends make the controls structurally independent: a shadow lift
/// cannot move white, a highlight lift cannot move black.
#[must_use]
pub fn region_shape(p: f32, shadows: f32, highlights: f32) -> f32 {
    let p = p.clamp(0.0, 1.0);
    let shadows = shadows.clamp(-REGION_LIFT_MAX, REGION_LIFT_MAX);
    let highlights = highlights.clamp(-REGION_LIFT_MAX, REGION_LIFT_MAX);
    let toe = (1.0 - p).powf(REGION_MASK_K);
    let shoulder = p.powf(REGION_MASK_K);
    (p + shadows * toe + highlights * shoulder).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inversion_maps_base_and_dmax_to_endpoints() {
        // White point sits at the film's usable density limit above base.
        let white_point_input = ACTIVE_STOCK.base * 10.0_f32.powf(-ACTIVE_STOCK.d_max);

        // One gray pixel at each endpoint.
        let mut rgb = vec![ACTIVE_STOCK.base; 3];
        for _ in 0..3 {
            rgb.push(white_point_input);
        }

        invert_mono(&mut rgb, &ACTIVE_STOCK, [ACTIVE_STOCK.base; 3]);

        assert!(rgb[..3].iter().all(|value| value.abs() < 1e-5));
        assert!(rgb[3..].iter().all(|value| (*value - 1.0).abs() < 1e-5));
    }

    #[test]
    fn inversion_is_monotonic_between_endpoints() {
        let inputs = [0.05, 0.2, 0.5, ACTIVE_STOCK.base];
        let mut rgb = Vec::new();
        for value in inputs {
            for _ in 0..3 {
                rgb.push(value);
            }
        }

        invert_mono(&mut rgb, &ACTIVE_STOCK, [ACTIVE_STOCK.base; 3]);

        for pixel in 0..inputs.len() - 1 {
            // Brighter scans (closer to clear base) must print darker.
            assert!(rgb[pixel * 3] > rgb[(pixel + 1) * 3]);
        }
    }

    #[test]
    fn inversion_clamps_out_of_range_input() {
        let mut rgb = vec![1.5, 1.5, 1.5];
        for _ in 0..3 {
            rgb.push(MIN_TRANSMISSION / 2.0);
        }

        invert_mono(&mut rgb, &ACTIVE_STOCK, [ACTIVE_STOCK.base; 3]);

        assert!(rgb[..3].iter().all(|value| value.abs() < 1e-5));
        assert!(rgb[3..].iter().all(|value| (*value - 1.0).abs() < 1e-5));
    }

    #[test]
    fn measure_base_takes_a_high_percentile() {
        let mut samples = vec![0.9_f32; 100];
        samples[0] = 0.3;

        let base = measure_base(&samples).unwrap();

        assert!((base - 0.9).abs() < 1e-6);
    }

    #[test]
    fn measure_base_needs_finite_samples() {
        assert_eq!(measure_base(&[]), None);
        assert_eq!(measure_base(&[f32::NAN]), None);
    }

    #[test]
    fn resolve_base_prefers_calibration_over_auto_and_preset() {
        // A roll calibration is the truth regardless of auto or measured.
        assert_eq!(resolve_base(Some(0.71), false, Some(0.9), &ACTIVE_STOCK), 0.71);
        assert_eq!(resolve_base(Some(0.71), true, Some(0.9), &ACTIVE_STOCK), 0.71);
    }

    #[test]
    fn resolve_base_auto_uses_a_plausible_measurement() {
        assert_eq!(resolve_base(None, true, Some(0.88), &ACTIVE_STOCK), 0.88);
    }

    #[test]
    fn resolve_base_auto_falls_back_on_implausible_or_missing_measurement() {
        // A frame without measurable clear film (or an absent measurement)
        // falls back to the preset base in auto mode too.
        assert_eq!(resolve_base(None, true, Some(0.02), &ACTIVE_STOCK), ACTIVE_STOCK.base);
        assert_eq!(resolve_base(None, true, None, &ACTIVE_STOCK), ACTIVE_STOCK.base);
    }

    #[test]
    fn resolve_base_defaults_to_the_preset() {
        // Preset-first: without calibration or auto, even a frame measurement
        // is ignored — the stock's preset base is the truth.
        assert_eq!(resolve_base(None, false, Some(0.88), &ACTIVE_STOCK), ACTIVE_STOCK.base);
        assert_eq!(resolve_base(None, false, None, &ACTIVE_STOCK), ACTIVE_STOCK.base);
    }

    #[test]
    fn measure_base_channels_recovers_per_channel_bases() {
        // Clear film with uniform green excess, plus a dust outlier.
        let mut rgb = Vec::with_capacity(300);
        for _ in 0..100 {
            rgb.extend_from_slice(&[0.60, 0.66, 0.54]);
        }
        rgb[..3].copy_from_slice(&[0.2, 0.2, 0.2]);

        let bases = measure_base_channels(&rgb).unwrap();

        assert!((bases[0] - 0.60).abs() < 1e-6);
        assert!((bases[1] - 0.66).abs() < 1e-6);
        assert!((bases[2] - 0.54).abs() < 1e-6);
    }

    #[test]
    fn inversion_neutralizes_cast_against_measured_bases() {
        // Pixels sitting at their channel's own base must all land on neutral
        // black, regardless of the cast between channels.
        let mut rgb = vec![0.60, 0.66, 0.54];
        let bases = measure_base_channels(&rgb).unwrap();

        invert_mono(&mut rgb, &ACTIVE_STOCK, bases);

        assert!(rgb.iter().all(|value| value.abs() < 1e-5));
    }

    #[test]
    fn measure_base_channels_falls_back_on_implausible_channels() {
        // Blue has no measurable clear film in this frame.
        let mut rgb = vec![0.6_f32; 90];
        for [_, _, b] in rgb.as_chunks_mut::<3>().0 {
            *b = 0.001;
        }

        let bases = measure_base_channels(&rgb).unwrap();

        assert!((bases[0] - 0.6).abs() < 1e-6);
        assert!((bases[1] - 0.6).abs() < 1e-6);
        assert_eq!(bases[2], ACTIVE_STOCK.base);
    }

    #[test]
    fn inversion_gamma_below_one_lifts_midtones() {
        let mut lifted = vec![0.5_f32; 3];
        invert_mono(&mut lifted, &ACTIVE_STOCK, [ACTIVE_STOCK.base; 3]);

        let linear = MonoStock {
            name: "linear",
            base: ACTIVE_STOCK.base,
            d_max: ACTIVE_STOCK.d_max,
            gamma: 1.0,
        };
        let mut neutral = vec![0.5_f32; 3];
        invert_mono(&mut neutral, &linear, [ACTIVE_STOCK.base; 3]);

        assert!(lifted[0] > neutral[0]);
    }

    #[test]
    fn inversion_gray_maps_base_and_dmax_to_endpoints() {
        // White point sits at the film's usable density limit above base.
        let white_point_input = ACTIVE_STOCK.base * 10.0_f32.powf(-ACTIVE_STOCK.d_max);
        let mut gray = vec![ACTIVE_STOCK.base, white_point_input];

        invert_gray(&mut gray, &ACTIVE_STOCK, ACTIVE_STOCK.base);

        assert!(gray[0].abs() < 1e-5);
        assert!((gray[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn inversion_gray_is_monotonic_between_endpoints() {
        let inputs = [0.05, 0.2, 0.5, ACTIVE_STOCK.base];
        let mut gray = inputs.to_vec();

        invert_gray(&mut gray, &ACTIVE_STOCK, ACTIVE_STOCK.base);

        // Brighter scans (closer to clear base) must print darker.
        for window in gray.windows(2) {
            assert!(window[0] > window[1]);
        }
    }

    #[test]
    fn inversion_gray_matches_rgb_inversion_on_neutral_pixels() {
        let inputs = [0.05, 0.2, 0.5, ACTIVE_STOCK.base];
        let mut rgb = Vec::new();
        for value in inputs {
            rgb.extend_from_slice(&[value; 3]);
        }
        let mut gray = inputs.to_vec();

        invert_mono(&mut rgb, &ACTIVE_STOCK, [ACTIVE_STOCK.base; 3]);
        invert_gray(&mut gray, &ACTIVE_STOCK, ACTIVE_STOCK.base);

        for (pixel, expected) in rgb.as_chunks::<3>().0.iter().zip(&gray) {
            for value in pixel {
                assert!((value - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn film_preset_default_is_none() {
        assert_eq!(FilmPreset::default(), FilmPreset::None);
    }

    #[test]
    fn film_preset_stock_mapping() {
        assert_eq!(FilmPreset::None.stock(), None);
        // Generic carries the neutral fallback profile; each stock its own.
        assert_eq!(FilmPreset::Generic.stock(), Some(GENERIC_STOCK));
        assert_eq!(FilmPreset::Fomapan100.stock(), Some(FOMAPAN_100));
        assert_eq!(FilmPreset::Fomapan400.stock(), Some(FOMAPAN_400));
        assert_eq!(FilmPreset::Delta400.stock(), Some(DELTA_400));
        assert_eq!(FilmPreset::Fp4Plus.stock(), Some(FP4_PLUS));
        assert_eq!(FilmPreset::Hp5Plus.stock(), Some(ACTIVE_STOCK));
        assert_eq!(FilmPreset::Kentmere400.stock(), Some(KENTMERE_400));
        assert_eq!(FilmPreset::TMax400.stock(), Some(TMAX_400));
        assert_eq!(FilmPreset::TriX.stock(), Some(TRI_X));
    }

    #[test]
    fn film_preset_inverted_marking() {
        assert!(!FilmPreset::None.is_inverted());
        for preset in FILM_CHOICES {
            assert_eq!(preset.is_inverted(), preset != FilmPreset::None);
        }
    }

    #[test]
    fn film_preset_choice_key_round_trip() {
        for preset in FILM_CHOICES {
            assert_eq!(FilmPreset::from_key(preset.choice_key()), preset);
        }
        // The retired auto keys are no longer presets.
        assert_eq!(FilmPreset::from_key("auto-per-frame"), FilmPreset::None);
        assert_eq!(FilmPreset::from_key("auto-selected-frame"), FilmPreset::None);
        // Unknown keys resolve to the default (None), like a missing entry.
        assert_eq!(FilmPreset::from_key(""), FilmPreset::None);
        assert_eq!(FilmPreset::from_key("delta"), FilmPreset::None);
    }

    #[test]
    fn film_preset_index_round_trip() {
        for preset in FILM_CHOICES {
            assert_eq!(FilmPreset::from_index(preset.index()), preset);
        }
        assert_eq!(FilmPreset::from_index(99), FilmPreset::None);
    }

    #[test]
    fn film_preset_dropdown_order() {
        // The dropdown lists None, Generic, then the stocks alphabetically in
        // `FILM_CHOICES` order — the exact ordering the index mapping must
        // match.
        for (index, preset) in FILM_CHOICES.into_iter().enumerate() {
            assert_eq!(preset.index(), index);
            assert_eq!(FilmPreset::from_index(index), preset);
        }
    }

    #[test]
    fn base_mode_default_is_preset() {
        assert_eq!(BaseMode::default(), BaseMode::Preset);
        assert!(!BaseMode::Preset.is_auto());
    }

    #[test]
    fn base_mode_choice_key_round_trip() {
        for mode in [BaseMode::Preset, BaseMode::AutoPerFrame, BaseMode::AutoSelectedFrame] {
            assert_eq!(BaseMode::from_key(mode.choice_key()), mode);
        }
        // Unknown keys resolve to the default (Preset), like a missing entry.
        assert_eq!(BaseMode::from_key(""), BaseMode::Preset);
        assert_eq!(BaseMode::from_key("delta"), BaseMode::Preset);
    }

    #[test]
    fn base_mode_index_round_trip() {
        for mode in [BaseMode::Preset, BaseMode::AutoPerFrame, BaseMode::AutoSelectedFrame] {
            assert_eq!(BaseMode::from_index(mode.index()), mode);
        }
        assert_eq!(BaseMode::from_index(99), BaseMode::Preset);
    }

    #[test]
    fn base_mode_auto_marking() {
        assert!(!BaseMode::Preset.is_auto());
        assert!(BaseMode::AutoPerFrame.is_auto());
        assert!(BaseMode::AutoSelectedFrame.is_auto());
    }

    #[test]
    fn effective_d_max_scales_the_window_symmetrically() {
        // Identity leaves the stock's range untouched.
        assert!((effective_d_max(2.4, 0.0) - 2.4).abs() < 1e-6);
        // +1 stop halves the window (more contrast); -1 doubles it.
        assert!((effective_d_max(2.4, 1.0) - 1.2).abs() < 1e-6);
        assert!((effective_d_max(2.4, -1.0) - 4.8).abs() < 1e-6);
        // Clamped so the window can never collapse or explode.
        assert!(effective_d_max(2.4, 99.0) >= 0.01);
        assert!(effective_d_max(2.4, -99.0) <= 100.0);
    }

    #[test]
    fn region_shape_is_independent_at_the_far_end() {
        // A shadow lift leaves white untouched; a highlight lift leaves black
        // untouched — the structural isolation the model relies on.
        for p in [0.0_f32, 0.1, 0.4, 0.7, 1.0] {
            let base = region_shape(p, 0.0, 0.0);
            assert!((base - p).abs() < 1e-6, "identity moved p={p}");
        }
        for shadows in [-0.5_f32, -0.2, 0.2, 0.5] {
            assert!(
                (region_shape(1.0, shadows, 0.0) - 1.0).abs() < 1e-6,
                "shadow lift moved white: {shadows}"
            );
        }
        for highlights in [-0.5_f32, -0.2, 0.2, 0.5] {
            assert!(
                region_shape(0.0, 0.0, highlights).abs() < 1e-6,
                "highlight lift moved black: {highlights}"
            );
        }
    }

    #[test]
    fn region_shape_lifts_the_named_region() {
        let dark = 0.15_f32;
        let bright = 0.8_f32;
        assert!(region_shape(dark, 0.3, 0.0) > dark, "shadow lift raised the toe");
        assert!(region_shape(bright, 0.0, 0.3) > bright, "highlight lift raised the shoulder");
        // And a lift on one region barely moves the other (mask falls off).
        assert!((region_shape(bright, 0.3, 0.0) - bright).abs() < 0.02);
        assert!((region_shape(dark, 0.0, 0.3) - dark).abs() < 0.02);
    }

    #[test]
    fn region_shape_stays_monotone_at_the_extremes() {
        // The bounded additive masks must not introduce a non-monotone wiggle
        // anywhere across the full lift range.
        for shadows in [-0.5_f32, -0.25, 0.0, 0.25, 0.5] {
            for highlights in [-0.5_f32, -0.25, 0.0, 0.25, 0.5] {
                let mut prev = region_shape(0.0, shadows, highlights);
                let mut p = 0.0_f32;
                while p < 1.0 {
                    p = (p + 0.005).min(1.0);
                    let v = region_shape(p, shadows, highlights);
                    assert!(
                        v >= prev - 1e-6,
                        "not monotone at p={p}: {prev} -> {v} (shadows={shadows} highlights={highlights})"
                    );
                    prev = v;
                }
            }
        }
    }

    // --- New density develop function (docs/raw-pipeline-rewrite.md) ---

    /// A Rust transcription of the WGSL `develop()` expression, so the parity
    /// test pins the shader and the CPU twin together by construction.
    fn wgsl_develop(d: &Develop, transmission: f32) -> f32 {
        let gain = (-d.exposure_ev).exp2();
        let value = (transmission * gain).clamp(1e-6_f32, 1.0_f32);
        let density = -f32::ln(value) / f32::ln(10.0);
        let span = (d.white - d.black).abs().max(1e-4);
        let x = ((density - d.black) / span).clamp(0.0, 1.0);
        let pivot = (0.5 + d.pivot_offset).clamp(1e-4, 1.0 - 1e-4);
        let shaped = if x <= pivot {
            pivot * (x / pivot).powf(d.contrast)
        } else {
            1.0 - (1.0 - pivot) * ((1.0 - x) / (1.0 - pivot)).powf(d.contrast)
        };
        shaped.clamp(0.0, 1.0)
    }

    #[test]
    fn develop_is_identity_at_defaults() {
        // Identity develop: no EV, identity contrast/pivot, standard anchors.
        let d = Develop::with_anchors(0.0, 2.4);
        for t in [0.02_f32, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let out = d.apply(t);
            // Compare against the plain normalized density (the reference).
            let density = f32::log10(1.0 / t);
            let expected = (density / 2.4).clamp(0.0, 1.0);
            assert!(
                (out - expected).abs() < 1e-5,
                "t={t}: {out} vs {expected}"
            );
        }
    }

    #[test]
    fn develop_matches_wgsl_transcription() {
        // CPU develop and the WGSL-shaped expression must agree across a wide
        // parameter sweep — the parity guarantee behind `detail == export`.
        for black in [-0.1_f32, 0.0, 0.2] {
            for white in [2.0_f32, 2.4, 3.0] {
                for contrast in [0.5_f32, 1.0, 2.5] {
                    for pivot in [-0.3_f32, 0.0, 0.3] {
                        for ev in [-1.0_f32, 0.0, 1.5] {
                            let d = Develop {
                                exposure_ev: ev,
                                black,
                                white,
                                contrast,
                                pivot_offset: pivot,
                            };
                            for t in [0.001_f32, 0.05, 0.2, 0.5, 0.8, 0.99] {
                                let a = d.apply(t);
                                let b = wgsl_develop(&d, t);
                                assert!(
                                    (a - b).abs() < 1e-6,
                                    "parity {a} vs {b} (t={t}, {d:?})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn develop_is_monotone_and_in_range() {
        for contrast in [0.25_f32, 1.0, 4.0] {
            for pivot in [-0.4_f32, 0.0, 0.4] {
                let d = Develop {
                    exposure_ev: 0.3,
                    black: 0.0,
                    white: 2.4,
                    contrast,
                    pivot_offset: pivot,
                };
                let mut prev = d.apply(1.0);
                let mut t = 1.0_f32;
                while t > 1e-3 {
                    t -= 0.002;
                    let v = d.apply(t);
                    assert!(
                        v >= prev - 1e-6,
                        "non-monotone at t={t}: {prev} -> {v} ({d:?})"
                    );
                    assert!((0.0..=1.0).contains(&v), "out of range at t={t}: {v}");
                    prev = v;
                }
            }
        }
    }

    #[test]
    fn slider_directions_are_right_to_increase() {
        // Work in density so "high density = dark positive" is unambiguous.
        let base = Develop::with_anchors(0.0, 2.4);
        let d_mid = 1.2_f32; // middle of the [0, 2.4] window (x = 0.5)
        let d_dark = 0.6_f32; // dark positive (x = 0.25)
        let d_bright = 2.0_f32; // bright positive (x = 0.83)

        // Exposure right ⇒ brighter: positive output rises at every
        // transmission (exposure is a density offset applied in `apply`).
        let brighter = Develop {
            exposure_ev: 1.0,
            ..base
        };
        let t_sample = 10.0_f32.powf(-d_mid);
        assert!(brighter.apply(t_sample) > base.apply(t_sample), "exposure");

        // Contrast right ⇒ more contrast: darks fall, brights rise.
        let more = Develop {
            contrast: 2.0,
            ..base
        };
        assert!(more.apply_density(d_dark) < base.apply_density(d_dark), "contrast dark");
        assert!(more.apply_density(d_bright) > base.apply_density(d_bright), "contrast bright");

        // White point: moving the white anchor DOWN (toward the data) brightens
        // (the same density now sits closer to white). The UI maps "White right
        // = brighten" onto this.
        let white_in = Develop {
            white: 1.8,
            ..base
        };
        assert!(white_in.apply_density(d_mid) > base.apply_density(d_mid), "white");

        // Black point: moving the black anchor UP (toward the data) darkens.
        // The UI maps "Black right = darken/clip darks" onto this.
        let black_up = Develop {
            black: 0.4,
            ..base
        };
        assert!(black_up.apply_density(d_mid) < base.apply_density(d_mid), "black");

        // Pivot shifts WHERE the contrast acts: the curve always passes through
        // the pivot point (pivot_density -> pivot_fraction), and moving the
        // pivot moves that fixed point. With contrast != 1, different pivots
        // give different curves.
        let contrast = 2.0_f32;
        for offset in [-0.3_f32, 0.0, 0.2, 0.4] {
            let d = Develop {
                contrast,
                pivot_offset: offset,
                ..base
            };
            let pivot_frac = 0.5 + offset;
            let pivot_density = base.black + pivot_frac * (base.white - base.black);
            let out = d.apply_density(pivot_density);
            assert!(
                (out - pivot_frac).abs() < 1e-5,
                "pivot {offset}: {out} vs {pivot_frac}"
            );
        }
        // The pivot actually changes the curve shape elsewhere.
        let low = Develop { contrast, pivot_offset: -0.3, ..base };
        let high = Develop { contrast, pivot_offset: 0.3, ..base };
        assert!(
            (low.apply_density(d_bright) - high.apply_density(d_bright)).abs() > 1e-3,
            "pivot did not change the curve"
        );
    }
}
