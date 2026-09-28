// SPDX-License-Identifier: GPL-3.0-or-later

//! Film-negative develop: density-domain tone mapping and clear-film base
//! measurement. (Legacy stock/preset machinery is being retired; see
//! `docs/raw-pipeline-rewrite.md`.)

/// Samples at or below this transmission encode maximum density. Shared by the
/// base measurement and the develop function.
pub const MIN_TRANSMISSION_DEV: f32 = 1e-6;

/// Fallback clear-film transmission when no calibration frame has measured one
/// yet. Only a starting point: [`crate::edit_manifest::RollManifest`] resolves
/// the per-frame `base` from the designated calibration frame.
pub const DEFAULT_FILM_BASE: f32 = 0.8;

/// Default usable density range above the clear-film base (the positive's white
/// point) until a frame records its own. The value mirrors the old HP5+
/// `d_max` used for visual calibration.
pub const DEFAULT_D_MAX: f32 = 2.4;

/// The pointwise develop function for a film negative: a monotone map from a
/// sensor-linear transmission to a display positive, evaluated entirely in
/// optical **density**.
///
/// Pipeline (all pointwise):
/// 1. Exposure is a density offset: a `2^-EV` gain on the transmission.
/// 2. `density = log10(base / value)` — clear film (`value == base`) is
///    density 0.
/// 3. Black/White normalize the density window: `x = (density - black) /
///    (white - black)`, clamped to `[0,1]`.
/// 4. Contrast bends `x` about the midtone pivot with the two-sided power
///    (`x/p)^k` / `1 - (1-p)·((1-x)/(1-p))^k`, identity at `k == 1` for every
///    pivot. The pivot is a signed offset from the window midpoint (default 0).
///
/// Every control is independent and monotone; each slider's rightward direction
/// is pinned by the tests (`slider_directions_*`). The identity (`exposure=0`,
/// `contrast=1`, `pivot=0`, and `black=0`/`white=DEFAULT_D_MAX` at the measured
/// `base`) is exactly the plain normalized density, so untouched values render
/// unchanged.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Develop {
    /// Exposure in EV; a density offset (positive brightens the positive).
    pub exposure_ev: f32,
    /// Clear-film transmission anchor: `density = log10(base / value)`. Pinned
    /// by the roll's calibration frame; a per-frame trim is allowed.
    pub base: f32,
    /// Density mapped to output black (`density == black` ⇒ 0).
    pub black: f32,
    /// Density mapped to output white (`density == white` ⇒ 1).
    pub white: f32,
    /// Contrast power about [`Self::pivot_offset`]; `1.0` = identity.
    pub contrast: f32,
    /// Signed offset from the `[black, white]` midpoint the contrast bends
    /// around. `0.0` = identity.
    pub pivot_offset: f32,
}

impl Develop {
    /// The identity develop anchored on `base`: no exposure, black at density 0,
    /// white at [`DEFAULT_D_MAX`], identity contrast and pivot.
    #[must_use]
    #[allow(dead_code)] // exercised by the develop unit tests
    pub fn with_base(base: f32) -> Self {
        Self {
            exposure_ev: 0.0,
            base,
            black: 0.0,
            white: DEFAULT_D_MAX,
            contrast: 1.0,
            pivot_offset: 0.0,
        }
    }

    /// Applies the develop to a sensor-linear transmission, returning a
    /// `[0,1]` positive (pre-sRGB).
    ///
    /// Mirrored exactly by the WGSL `develop()`; keep the two in lockstep (the
    /// `develop_matches_wgsl_transcription` parity test uses a Rust
    /// transcription of the shader's expression).
    #[must_use]
    pub fn apply(&self, transmission: f32) -> f32 {
        let gain = (-self.exposure_ev).exp2();
        let base = self.base.max(MIN_TRANSMISSION_DEV);
        let value = (transmission * gain).clamp(MIN_TRANSMISSION_DEV, base);
        let density = f32::log10(base / value);
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

/// Measured channel/base transmissions below this are implausible — they
/// indicate frames without any measurable clear film — and are rejected.
pub const MIN_PLAUSIBLE_BASE: f32 = 0.1;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_base_takes_a_high_percentile() {
        // A ramp with a few hot outliers: the 95th-percentile guard ignores
        // them and lands near the top of the bulk distribution.
        let mut samples: Vec<f32> = (0..=100).map(|v| v as f32 / 100.0).collect();
        samples.push(5.0);
        let base = measure_base(&samples).expect("finite samples");
        assert!((base - 0.95).abs() < 0.06, "base {base}");
    }

    #[test]
    fn measure_base_needs_finite_samples() {
        assert_eq!(measure_base(&[]), None);
        assert_eq!(measure_base(&[f32::NAN, f32::INFINITY]), None);
    }

    /// A Rust transcription of the WGSL `develop()` expression, so the parity
    /// test pins the shader and the CPU twin together by construction.
    fn wgsl_develop(d: &Develop, transmission: f32) -> f32 {
        let gain = (-d.exposure_ev).exp2();
        let value = (transmission * gain).clamp(1e-6_f32, d.base);
        let density = -f32::ln(value / d.base) / f32::ln(10.0);
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
        // Identity develop: no EV, identity contrast/pivot, default anchors.
        let d = Develop::with_base(0.82);
        for t in [0.02_f32, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let out = d.apply(t);
            // Compare against the plain normalized density (the reference).
            let density = f32::log10(d.base / t);
            let expected = (density / d.white).clamp(0.0, 1.0);
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
        for base in [0.5_f32, 0.82, 0.95] {
            for black in [-0.1_f32, 0.0, 0.2] {
                for white in [2.0_f32, 2.4, 3.0] {
                    for contrast in [0.5_f32, 1.0, 2.5] {
                        for pivot in [-0.3_f32, 0.0, 0.3] {
                            for ev in [-1.0_f32, 0.0, 1.5] {
                                let d = Develop {
                                    exposure_ev: ev,
                                    base,
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
    }

    #[test]
    fn develop_is_monotone_and_in_range() {
        for contrast in [0.25_f32, 1.0, 4.0] {
            for pivot in [-0.4_f32, 0.0, 0.4] {
                let d = Develop {
                    exposure_ev: 0.3,
                    base: 0.82,
                    black: 0.0,
                    white: 2.4,
                    contrast,
                    pivot_offset: pivot,
                };
                let mut prev = d.apply(d.base);
                let mut t = d.base;
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
        let base = Develop::with_base(0.82);
        let d_mid = 1.2_f32; // middle of the [0, 2.4] window (x = 0.5)
        let d_dark = 0.6_f32; // dark positive (x = 0.25)
        let d_bright = 2.0_f32; // bright positive (x = 0.83)

        // Exposure right ⇒ brighter: positive output rises at every
        // transmission (exposure is a density offset applied in `apply`).
        let brighter = Develop {
            exposure_ev: 1.0,
            ..base
        };
        let t_sample = base.base * 10.0_f32.powf(-d_mid);
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
