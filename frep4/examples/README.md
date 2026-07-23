# Example scenes

Small `.json` scene files demonstrating common patterns. Each one
is a valid input to `frep_gpu_render --scene`. To render the whole
set:

```bash
mkdir -p out
for f in examples/*.json; do
    name=$(basename "$f" .json)
    ./build/frep_gpu_render "out/$name.ppm" 800 500 --scene "$f"
done
```

| File | Demonstrates |
|---|---|
| `01_csg_basic.json` | Three CSG ops (Difference, SmoothUnion, Intersection) side by side |
| `02_twisted_column.json` | TwistY deformation on a thin box |
| `03_smooth_union_blob.json` | Two spheres blended via SmoothUnion |
| `04_patterned_spheres.json` | Solid, Checker, and Stripes pattern materials |
| `05_carved_sphere.json` | Sphere with a cubic Difference cut-out |
| `06_textured_objects.json` | Wood-textured sphere + marble-textured cube (loaded from `textures/`) |

The `textures/` subdirectory contains the BMP texture assets referenced
by the textured scenes — these are procedurally generated (256×256
each), bundled so the examples render out-of-the-box without
external downloads.

These files use only built-in node types — no plugin registry needed
to load them. They're also good starting points for hand-editing: open
any `.json` in a text editor to see the schema documented in
`docs/USER_GUIDE.md`.

To regenerate this set from C++:

```bash
# See the build process in tools/build_examples.cpp (not committed —
# scene files are committed instead so they're language-independent).
```

The scenes are also exercised programmatically by `frep_gallery`
(`tools/gallery.cpp`), which renders richer variants with multiple
lights and SSAA.

## Benchmark scenes (`benchmark_*.json`)

A set of ready-made scenes for manual performance testing, so you don't
have to hand-build complex scenes to compare render modes or check the
adaptive spatial-guard behaviour. Two families:

**Simple** — cheap per-object SDFs (~2 nodes each). The spatial-guard
heuristic leaves these on the inlined path (guarding bare primitives
doesn't help; the vectorised `min()` wins):

- `benchmark_simple_spheres_27.json` — 27 spheres
- `benchmark_simple_boxes_64.json` — 64 boxes
- `benchmark_simple_spheres_125.json` — 125 spheres (count stress)

**Heavy** — expensive per-object SDFs (twist + smooth-union, or CSG;
~4–6 nodes each). Once the host is calibrated, the heuristic switches
these to the guarded path, where measured speedups are ~3–6.5×:

- `benchmark_heavy_twist_27.json` — twisted box + smooth-union sphere
- `benchmark_heavy_twist_64.json`
- `benchmark_heavy_twist_125.json` — large; inline baseline is slow
- `benchmark_heavy_csg_48.json` — box-minus-sphere difference per object
- `benchmark_mixed_48.json` — half cheap / half heavy (tests averaging)

Load any in the app (File → Open) and toggle **Render → Adaptive spatial
guards** to compare, or render headless. The first heavy scene compiled
triggers a one-time guard calibration (~2 s); simple scenes never do.

To regenerate the benchmark set:

```bash
# builds the scenes through the SceneGraph API → schema-valid JSON
clang++ -std=c++23 tools/gen_benchmark_scenes.cpp build/libfrep_core.a ... -o gen
./gen examples
```

