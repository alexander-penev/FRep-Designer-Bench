# Parity verification (guards against "fast but wrong")

The render benchmark (frep4_bench_render) now cross-checks every render, so a
path that runs fast but produces the wrong image — or nothing — is caught.

## Visual parity (between render paths)
Each scene's first available path (cpu_ir) is the reference; every other path
(gpu_glsl, gpu_rtx, gpu_ir) is diffed against it. Reported on stderr per scene:
  ~~ gpu_glsl vs cpu_ir (s2_csg): max 0.0044, mean 0.0007 — identical
  ~~ gpu_rtx  vs cpu_ir (s2_csg): max 0.0029, mean 0.0000 — identical
Verdict: identical (max<0.02), close (AA/silhouette, mean<0.02), or DIVERGENT.
Sub-pixel silhouette differences are expected between a raymarcher and an
RT-BVH path; a DIVERGENT verdict means a real correctness bug.

## Black-screen / NaN guard
Every first render is scanned: if it contains NaN/inf pixels, or under 1% of the
frame is lit, the bench flags it:
  !! cpu_ir s2_csg  only 0.30% of frame lit — possible BLACK SCREEN
  !! cpu_ir gears   N NaN/inf pixels — render is CORRUPT
This is exactly what would have caught the converted gear scenes (100% NaN ->
black) instead of silently reporting a fast-but-empty render.

## Still TODO
- Cross-SYSTEM numeric parity: sample the same points through frep4 / libfive /
  hyperfun / mpr and confirm the SDF values agree (the systems share the same
  implicit function; the evaluators should too). Needs a --dump-values mode in
  each bench tool + a comparison script.

## Numeric parity (scalar vs SIMD) — implemented & verified
A standalone check (scalar compile_scene_sdf vs SIMD compile_scene_sdf_simd)
confirms the two evaluators agree:
  s2_csg:   scalar vs SIMD max diff 0.000000 over 28800 pts — PARITY OK
This caught a real asymmetry on the gear scenes: the scalar path is now
domain-safe (pow = copysign(|a|^b,a)) and produces no NaN, but the SIMD path's
transcendental approximations (asin/atan2 via sqrt(1-x^2)) still NaN on the
gears' out-of-domain inputs. So on pow/trig-pathological imported scenes the
scalar and SIMD paths currently disagree — flagged, not hidden. Normal scenes
(spheres, CSG, blends, gyroid, twist) have full scalar/SIMD parity.

## Status of the gear scenes (known limitation, now visible)
The involute-gear .let files use pow(x, -0.2) — NaN for x<0 in raw IEEE pow, and
in libfive's array evaluator too. libfive renders them via its *interval*
evaluator, which prunes the out-of-domain branch before evaluating; the flat
expr conversion lost that. Domain-safe pow fixes the scalar path (gears no longer
all-NaN), but:
  - the SIMD path still NaNs on them (trig approximations), and
  - even NaN-free the gear field has no clean inside region at coarse sampling.
The benchmark now FLAGS all of this (black-screen guard + scalar/SIMD parity)
rather than reporting a fast-but-empty render as if it were fine. Treat the gear
throughput/render numbers as unreliable until the converter emits domain-guarded
expressions (or the scenes are re-exported through libfive's interval path).
