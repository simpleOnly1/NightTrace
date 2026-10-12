# Lightweight paired approximate imaging

Use `sample_parameters(seed, ranges)` once per image/replicate, then `degrade`
for `clean, L, N, G, LN, LG, NG, LNG`. Parameters are continuous uniforms defined
in `config/*.yaml`; integer source count is uniform inclusive. These defaults
are **uncalibrated experimental choices**, not quoted TDDC/DepthDark parameters.

The signal chain is inverse sRGB → exposure attenuation → additive optical
halo/streak → signal-dependent shot noise + read noise → clip → sRGB.
Already processed RGB cannot recover RAW irradiance or a calibrated camera.
Adding flare after background exposure is an independently controlled *sensor
plane* radiance intervention, not a claim that real lamps ignore shutter time.
Low exposure and fixed sensor noise naturally reduce signal-to-noise ratio;
noise and exposure parameters themselves are sampled independently for the
factorial experiment. Do not call this a fitted joint night distribution.

All factors share the same per-image/per-replicate parameters. Local torch RNG
streams make outputs independent of condition ordering and skipped conditions.
Shot counts depend on signal. Read noise uses a separate reproducible stream.
White flare only: channel complementarity is outside this experiment's scope.

Each source has continuous `(u,v,radius,amplitude,aspect,angle,streak_weight)`.
Radius is a fraction of the image short edge. It is **not distance in metres**.
The broad position range is a simple image-plane prior, not a vehicle/light
semantic detector. No claimed wet-road reflection, metric lamp placement, RAW
calibration, or Flare7K asset is bundled. Near-frame exterior sources are allowed.
The G-off conditions supply no-flare controls; G-on has 1–3 sources by default.

The compact G-only probe uses this same renderer with truncated support and
narrower ranges. It is a controlled local stimulus, not natural night synthesis.
Its support is recomputed from the actual model-preprocessed image difference.

Legacy `nighttime_degradation`, `apply_*` and `analysis.degradation.corrupt`
route into this backend. Legacy near/medium/far are deprecated radius categories;
legacy colored names in `corrupt` map to achromatic conditions. New diagnostics
use only the explicit factor labels above.
