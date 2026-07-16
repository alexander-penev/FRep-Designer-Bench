# Canonical scenes (identical math in all four systems)

Domain: [-1.6,1.6]^3. Sign convention: frep4/libfive/mpr use distance-like f<=0 inside;
HyperFun uses f>=0 inside -> HF scenes are the NEGATION of the formulas below. |X| = sqrt(x^2+y^2+z^2).

s1_sphere : f = |X| - 1
s2_csg    : f = max( max(|x|,|y|,|z|) - 0.9 , -( |X| - 1.1 ) )        # cube minus sphere
s3_blend  : a = |X-( 0.45,0,0)| - 0.7 ; b = |X-(-0.45,0,0)| - 0.7
            h = max(0.25 - |a-b|, 0) / 0.25
            f = min(a,b) - h*h*h * 0.25/6                              # C2 cubic smooth-union
s4_gyroid : g = sin(3x)cos(3y) + sin(3y)cos(3z) + sin(3z)cos(3x) - 0.2
            f = max(g, |X| - 1.4)
s5_twist  : w = 1.2 ; xt =  x cos(wz) + y sin(wz) ; yt = -x sin(wz) + y cos(wz)
            f = max(|xt| - 0.35, |yt| - 0.35, |z| - 1.1)               # twisted bar

All systems implement these via primitive ops (min/max/abs/sqrt/sin/cos) only - no native
blend/twist nodes - so every system evaluates the same expression tree.

# Complex scenes (larger DAGs, same cross-system contract)

Defined in libfive_bench/scenes_trees.hpp (complex_scenes()) and emitted to every
system through the same DAG->text export path as the canonical set. Built only from
min/max/abs/sqrt and constant rotations, so they are valid signed distance fields
(finite everywhere, f<=0 inside) that every backend evaluates bit-identically.

c1_gear      : disc body max(sqrt(x^2+y^2)-0.85, |z|-0.22) minus a central bore,
               union of 18 radial teeth (rotated boxes). ~440 nodes.
c2_colonnade : floor + roof slabs joined by a 5x5 grid of columns
               (cylinders max(sqrt((x-cx)^2+(y-cy)^2)-0.11, |z|-0.78)). ~410 nodes.

These REPLACE the original imported archives (architecture / hello_world /
involute_gear / prospero / bear). Those .frep files were not signed distance fields
- their root ops were atan2 and division, so they evaluate to NaN across the whole
domain and every evaluator, including libfive's own interval heightmap renderer,
draws them empty (0% surface). Any throughput measured on them was timing NaN
arithmetic, not geometry. Removed.
