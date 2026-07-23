# FRep Designer 4.0.0 — Release Notes

**Release date**: 2026-05-23

> **Note:** these are the notes for the 4.0.0 release. The system has since
> grown a fourth retargeting path (GPU_RTX, Vulkan ray tracing), energy and
> distributed-render tooling, and a reworked GUI. For the current state see
> `README.md` and `CHANGELOG.md`.

The first stable release of FRep Designer — a complete F-Rep
geometric modeling system with three execution back-ends (CPU JIT,
GPU compute, AST interpreter) sharing a unified node graph and
scene representation.

---

## Highlights

- **Three rendering paths from one scene**: CPU JIT via LLVM 20
  with full Cook-Torrance PBR; GPU compute via Vulkan 1.3
  (4-6× faster); AST interpreter for picker, marching cubes, and
  BVH voxelization.
- **Runtime analytic expressions**: type `sin(x)*cos(y) +
  sin(y)*cos(z) + sin(z)*cos(x)` and get a gyroid. One parser
  feeds three back-ends — adding a new math function is a 4-line
  patch.
- **Dynamic plugin system**: write a custom F-Rep primitive in
  ~150 lines of C++, drop the `.so` in `plugins/`, and it appears
  in the GUI with full CPU/GPU/mesh-export support.
- **Two-way scene ↔ node graph sync**: edits in the inspector,
  toolbar, expression editor, or node graph all stay in sync;
  no destructive overwrites.
- **Marching-cubes export**: any scene → STL/OBJ at user-selectable
  resolution (32–256), async with live progress.
- **Comprehensive testing**: 211 GoogleTest tests + 1 headless
  graph logic test, all passing on CI. Performance regression
  detection with committed baseline.

---

## Getting started

```bash
git clone <repo-url> frep-designer-4.0
cd frep-designer-4.0
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/frep_designer --empty
```

See [Tutorial](docs/TUTORIAL.md) for a step-by-step walkthrough.

---

## Performance

At 800×600 the GPU compute pipeline is 4-6× faster than the CPU JIT
across all benchmark scenes:

| Scene | CPU @ 800×600 | GPU @ 800×600 | Speedup |
|---|---|---|---|
| Simple (1 sphere) | 260 ms | 64 ms | 4.1× |
| Moderate (CSG diff) | 307 ms | 67 ms | 4.6× |
| Complex (5 objs, deforms) | 1460 ms | 252 ms | 5.8× |
| Heavy (MeshSDF + CSG) | 590 ms | 188 ms | 3.1× |
| CustomExpr (gyroid) | 576 ms | 114 ms | 5.1× |

Run `./build/frep_bench` for numbers on your hardware.

---

## Documentation

- [Tutorial](docs/TUTORIAL.md) — build your first scene
- [User guide](docs/USER_GUIDE.md) — comprehensive reference
- [Architecture overview](docs/ARCHITECTURE.md) — internals
- [Plugin authoring](docs/PLUGIN_AUTHORING.md) — write your own
  F-Rep primitive
- [Performance tuning](docs/PERFORMANCE_TUNING.md) — choosing the
  right execution path
- [Examples gallery](docs/GALLERY.md) — 11 showcase scenes
- [Benchmarks](docs/BENCHMARKS.md) — measured performance
- API reference — `cmake --build build --target docs`

---

## Requirements

| Dependency | Version | Required? |
|---|---|---|
| C++ compiler | clang++-20 or gcc-14 (C++26 features) | yes |
| CMake | 3.20+ | yes |
| LLVM | 20.x (also tested with 21.x and 22.x) | yes |
| Qt | 6.4+ | for `frep_designer` GUI only |
| Vulkan SDK | 1.3+ | for GPU compute path; CPU works without |
| glslangValidator | matching Vulkan SDK | for GPU compute path |
| libpng | 1.6+ | optional — PNG texture loading |
| GoogleTest | 1.14+ | tests |
| doxygen + graphviz | any recent | optional — API docs generation |

On Ubuntu 24:

```bash
sudo apt install \
    clang-20 cmake build-essential \
    llvm-20-dev libllvm20 \
    qt6-base-dev qt6-base-private-dev \
    libvulkan-dev mesa-vulkan-drivers vulkan-tools \
    glslang-tools \
    libpng-dev \
    libgtest-dev libgmock-dev \
    doxygen graphviz
```

---

## Known limitations

- **Animation timeline** is not yet implemented. The architecture
  supports it (all parameters are numeric and addressable by id),
  but no UI exists. Planned for a future release.
- **Real-time GPU viewport** (QVulkanWindow + swapchain at 60 FPS)
  is not implemented. The current viewport renders on demand
  (per-edit), which is fast enough at low resolutions but not
  truly interactive for complex scenes.
- **Windows and macOS** builds are not tested in CI, though the
  code is cross-platform-friendly. Reports welcome.
- **Plugins are loaded with full process privileges**. Only load
  plugins from trusted sources.

---

## License

MIT. See [LICENSE](LICENSE) for the full text. The project links
against LLVM (Apache-2.0 with LLVM Exceptions), Qt6 (LGPL v3 or
commercial), Vulkan (Apache-2.0/MIT), libpng (zlib/libpng), and
GoogleTest (BSD-3-Clause).

---

## Acknowledgements

Originally developed at the Faculty of Mathematics and Informatics,
Plovdiv University. Research funding from FP17-FMI-008.

The shared-AST refactor was inspired by the Frostbite engine's
"Moving Frostbite to PBR" approach to baking constants into runtime
lighting values. Marching cubes uses the classic Lorensen & Cline
1987 lookup tables. The sparse octree compression scheme is adapted
from "Efficient Sparse Voxel Octrees" (Laine & Karras 2010).

Special thanks to the LLVM, Qt, and Vulkan communities for the
foundations this project rests on.
