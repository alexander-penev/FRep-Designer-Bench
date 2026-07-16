# Section 6 material — cross-system comparison (DRAFT for Penev to rewrite)

**What this is.** Ready-to-drop material for the Results section, in the paper's
own build markers (SUB / TABLE / FIG / PARA), plus the data and the honest
caveats. It is working material, not final prose — per the CIT AI policy the
wording must be rewritten by hand before submission. The *numbers*, *table*, and
*figures* are real and reproducible; treat them as the contribution and re-phrase
the sentences.

**Measurement provenance (must be stated wherever these numbers appear).**
- Hardware: dual-socket Intel Xeon E5-2690 v4 (Broadwell, AVX2, no AVX-512),
  single core pinned with `taskset -c 0`; median of 3 runs (each an internal
  median). This is a DIFFERENT machine from the Table 5/6 render figures — do not
  mix them into one table without a consistent-hardware note.
- Grid: 193^3 samples over [-1.6,1.6]^3, the identical grid all three systems use.
- Scenes: the five canonical (s1-s5) plus two valid complex SDFs (c1_gear,
  c2_colonnade). The previously-imported archives (gears/architecture/hello_world/
  prospero) are excluded — they are not signed distance fields (atan2/division
  roots) and evaluate to NaN in every system, libfive's own interval renderer
  included; timing them measures NaN arithmetic.
- **mpr is deliberately NOT in this table.** It is a GPU interval renderer with no
  scalar SDF entry point, so it cannot evaluate the shared grid; forcing it in
  would break the like-for-like contract. mpr belongs in the GPU-render discussion
  (its geometry parity is shown separately, image-only).

---

## Draft section (paper markers)

SUB:Cross-system: compiler, interval tree, and interpreter

The per-target results above characterize FRep Designer's own backends. To place
the compiler organization against the alternatives named in Section 2, Table 8
compares raw SDF-evaluation throughput on a shared 193^3 grid for three systems
that evaluate the same field point-wise: FRep Designer's CPU_IR (a JIT compiler),
libfive (an interval-arithmetic tree kernel), and HyperFun (a tree-walking
interpreter). The comparison is like-for-like: every system evaluates the same
signed distance field on the same grid, and the three agree to floating-point
epsilon (maximum per-point difference about 2-5 x 10^-7, with no sign
disagreement), so the throughput difference reflects only how each system
evaluates, not what it computes. mpr is excluded here: as a GPU interval renderer
it exposes no scalar evaluation of the field and so cannot run this grid.

TABLE:8:Grid SDF-evaluation throughput (Mvox/s; higher is faster) on a shared 193^3 grid. Fields agree across all three systems to about 2-5 x 10^-7. Single core pinned, Broadwell Xeon; setup excluded.
Scene|HyperFun (interpreter)|libfive (interval tree)|CPU_IR (compiler)|vs interpreter
Sphere (s1)|4.8|221|300|62x
CSG diff (s2)|2.4|229|270|111x
Smooth blend (s3)|1.7|167|108|65x
Gyroid (s4)|1.4|93|145|101x
Twisted bar (s5)|2.1|159|315|148x
Gear (c1, ~440 nodes)|0.11|22|89|798x
Colonnade (c2, ~410 nodes)|0.10|37|61|582x
ENDTABLE

PARA
Two facts stand out. First, compiling the whole evaluation is one to nearly three
orders of magnitude faster than interpreting it: CPU_IR outpaces HyperFun by 62x
on the cheapest scene and by roughly 600-800x on the complex ones, where the
interpreter re-walks a large expression tree at every one of the 193^3 points
while the compiler pays that traversal once, at JIT time. This is the
compiler-versus-interpreter thesis of Section 1 as a measured quantity rather than
an argument. Second, against libfive — the closer modern relative, which subdivides
the field over space rather than compiling the image-forming computation — CPU_IR
is competitive and usually ahead: faster on six of the seven scenes (from 1.2x on
CSG to 4.1x on the gear) and behind only on the smooth blend, where libfive's
array evaluator is well served by its cache. The advantage widens on the larger
scenes, exactly where whole-program compilation has more to amortize.

FIG:1_throughput_3systems.png:Grid SDF-evaluation throughput across the three point-wise evaluators (log scale). The three compute the same field to floating-point epsilon; the bars show only how fast each evaluates it. Compilation (CPU_IR) leads the interpreter by 62-798x and the interval-tree kernel on six of seven scenes.:7.4

PARA
The parity underlying the table is not assumed but measured: rendering each
system's field through one identical orthographic projection yields
pixel-indistinguishable images, with an all-black difference map. Fig. 2 shows this
for three scenes; it is the correctness counterpart to the throughput bars — the
systems are being timed on genuinely the same geometry.

FIG:4_parity_gear_3way.png:The gear scene evaluated by FRep Designer, libfive, and HyperFun through one shared projection, with the HyperFun-vs-libfive difference at right (black = identical). Same geometry, three independent evaluators.:7.4

---

## Draft section — second table (full render)

SUB:Full-frame render: native pipelines

The grid comparison isolates one thing — evaluating the distance function — which
is the only work all systems share. It therefore understates the compiler's reach:
FRep Designer compiles the whole image-forming computation (the field, its
gradient for normals, the shading model, and the sphere-tracing loop) into one
program, not just the field. Table 9 measures that whole pipeline: the
steady-state time to render one full frame at 512x512 for each system's native
renderer. This is deliberately not like-for-like — each system renders its own
way (FRep Designer sphere-traces on CPU and three GPU paths; mpr rasterizes by
GPU interval subdivision; libfive builds a CPU orthographic height map) — so the
numbers compare systems as built, not a fixed algorithm. The input geometry is the
same (mpr and libfive share the tree these scenes are exported from), so the
comparison is fair at the level of "render this solid, best effort," which is the
question a user actually asks.

TABLE:9:Steady-state full-frame render time at 512x512 (wall-clock ms; lower is faster). Native renderer of each system; setup excluded. mpr renders the canonical set only. Different algorithms — see text.
Scene|libfive height map (CPU)|CPU_IR (CPU)|mpr (GPU interval)|GPU_GLSL|GPU_IR
Sphere (s1)|5.5|9.4|3.0|7.4|4.6
CSG diff (s2)|7.0|6.6|2.9|5.9|4.5
Smooth blend (s3)|6.3|9.1|3.5|5.9|4.8
Gyroid (s4)|9.4|14.1|21.6|5.3|4.5
Twisted bar (s5)|6.2|9.9|22.2|4.4|4.4
Gear (c1)|42.1|29.1|--|6.1|5.4
Colonnade (c2)|27.3|47.4|--|6.1|6.2
ENDTABLE

PARA
The full-render picture differs from the grid one in a way that favors the
compiled pipeline. FRep Designer's GPU_IR path renders in 4.4-6.2 ms across every
scene, essentially independent of the geometry: the sphere-tracer and shading are
compiled once and the field is one call inside the loop, so a gyroid costs the same
as a sphere. mpr, by contrast, is fastest of all on the simple scenes (about 3 ms)
but degrades sharply on the gyroid and the twisted bar (to about 22 ms), because
its interval subdivision loses its pruning power on non-metric, trigonometric
fields — a scene-dependent cliff of roughly 7x. The compiled sphere-tracer has no
such cliff. On the CPU side the same holds against libfive's height-map renderer:
comparable on the simple scenes, and the two swap the lead on the complex ones,
with neither cliffing. The hardware ray-tracing path (GPU_RTX) is slower here
(about 30-38 ms across scenes) because it currently renders a full frame and crops
to the tile; its value is feature parity (Table 7), not raw speed at this
resolution. The reading is not "fastest everywhere" — mpr wins the simple scenes —
but "predictable everywhere": compiling the whole pipeline buys performance that
does not depend on whether the field happens to be metric.

FIG:5_mpr-vs-frep4_geometry.png:The same geometry rendered by mpr (GPU interval subdivision) and FRep Designer (sphere tracing). Different renderers, same solid — mpr shares the tree the scene is exported from. This is the geometry-parity basis for comparing the two GPU renderers in Table 9.:7.4

---

## Draft — optional block A: GPU_IR confirmed on real NVIDIA hardware

Short addition, folds into the parity discussion (Table 7). Data: measured on the
RTX 2080 via the CUDA Driver API.

PARA
The shared-IR pair is confirmed on real hardware, not only in software. GPU_IR
lowers the same LLVM IR the CPU path JITs to NVPTX and runs it through the CUDA
driver; on the RTX 2080 its output matches the CPU_IR reference to a mean per-pixel
difference of 7 x 10^-4 (maximum 0.002-0.003 across scenes, within the parity
tolerance) — the Table 7 figure, measured on the device rather than on a software
Vulkan/CPU fallback. It is also the fastest of the four render paths (Table 9,
about 4.5-6.2 ms per frame at 512x512), and the only one whose render time is
essentially independent of the scene, since the field is a single call inside a
compiled sphere-tracer. Reaching this required lowering llvm.pow/exp/log and the
trigonometric intrinsics to NVPTX (the backend selects only a few transcendentals
natively); once lowered, the shared-IR guarantee holds end to end from CPU JIT to
GPU PTX with no separate GPU codegen.

---

## Draft — optional block B: API shape as an architectural advantage (FREP_GRID_HOIST)

Short addition to the grid discussion. Makes concrete that the evaluator's calling
convention, not just its code, has performance consequences.

SUB:The evaluation interface as a performance lever

The grid figures of Table 8 use a calling convention chosen to match libfive's:
each point's three coordinates are written before every evaluation, as libfive's
ArrayEvaluator::set(p,i) requires. FRep Designer's SIMD entry point instead takes
three separate coordinate arrays, which lets the harness hoist the invariant x
(constant along a scan row) and y stores out of the innermost loop — an
optimization libfive's interface cannot express. Enabling it leaves the results
bit-identical and speeds the grid evaluation by the factors in Table 10, measured
on the same Broadwell host. The gain is largest on the cheapest scenes, where the
coordinate broadcast is a larger share of the work, and shrinks as per-point
arithmetic grows; it should widen again at 16-lane width, where the removed
broadcast is twice as large. The point is not the speedup itself but its source:
the shape of the evaluation interface — three arrays rather than one point at a
time — is a compiler-design choice with a measurable cost, of a piece with the
paper's framing of the model as a program.

TABLE:10:Grid-evaluation speed-up from hoisting invariant coordinate stores (bit-identical results). AVX2 width 8, Broadwell, grid 193^3.
Scene|Speed-up
Sphere (s1)|2.13x
Smooth blend (s3)|1.36x
Gyroid (s4)|1.40x
Gear (c1)|1.14x
Colonnade (c2)|1.15x
ENDTABLE

---

## Notes for integration

- Table/figure numbers (8, and the two figs) are placeholders — renumber into the
  existing sequence (Tables 5-7, the fig series).
- Figure files are in results/apples_to_apples/ (1_throughput_3systems.png,
  2_parity_s2csg_3way.png, 3_parity_gyroid_3way.png, 4_parity_gear_3way.png).
  The throughput chart is currently an SVG->PNG; for the paper it should be redrawn
  in the TikZ chart style used by the other figures (data in throughput_data.csv).
- The scene names are mapped to the paper's descriptive style (Sphere/CSG/…); the
  raw ids are s1_sphere etc.
- Related Work (Section 2) already sets up HyperFun and libfive as the comparison
  points but the CIT draft has no empirical comparison against them — this closes
  that loop. Consider one back-reference sentence there pointing to Table 8.
- If GPU_IR-on-real-hardware and the FREP_GRID_HOIST API-advantage result are also
  wanted, they are separate short additions (both measured); ask and I'll draft them.
