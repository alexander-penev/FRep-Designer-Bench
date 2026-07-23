# Computation paths and stages

This document records the conceptual model behind the multi-path execution
system (`core/exec/`). It is a *direction-setting* document: not everything
here is built yet, but new code should fit this model or leave a clear path
toward it. The guiding principle is correctness of the model over speed of
arrival — we would rather reach the right abstraction across several
increments than lock in a narrow one early.

## The core idea: a model is a program

A geometric model — the scene graph of F-Rep nodes — is not data to be
drawn. It *is* a computation, written in a visual, domain-specific
programming language whose semantics are those of signed distance fields,
CSG, deformations, shading, and rendering. The system is therefore a
**compiler**: it translates that computation from the visual language into
the language of some executor (CPU, GPU, …), honouring that executor's
architecture and semantics, directly or — more usually — through an
intermediate representation (LLVM IR, GLSL, PTX, SPIR-V).

Crucially, *everything* needed to produce the final image is part of the
compiled computation, not just the geometry: the implicit SDFs, the
shaders, the raymarching algorithm, the acceleration structures, and the
post-processing (SSAA, denoising, depth of field, motion blur, …). A
post-process is not a separate category bolted on at the end — it is simply
another algorithm on the path from model to image, and it too is compiled
to and run on an executor.

## A path is the whole chain, not just the executor

It is tempting to call "CPU-IR", "GPU-GLSL", "GPU-IR" the three *paths*. But
that names only the executor. A path is the **entire chain** from the shared
model to the pixels on screen:

```
Model → emit/retarget → executor → result return → post-process → presentation
```

Two chains that share an executor can still produce different final pixels
because they differ in a *later* stage — most concretely, SSAA done as a
CPU box-downsample vs a GPU bilinear blit. That difference belongs to the
path, and must be explicit, not incidental. When we compare or stitch paths
(see `MULTIPATH.md`), a divergence should mean "the executors computed
something different", not "the paths happened to post-process differently".

So a path is named by its full chain. At this stage we spell it out
literally, e.g.

```
Model → IR → CUDA → GPU raymarch → SSAA(box) → offscreen readback → image
Model → GLSL → Vulkan → GPU raymarch → SSAA(bilinear) → swapchain → screen
Model → IR → JIT → CPU raymarch → SSAA(box) → offscreen → image
```

Short human-friendly names may be added later, but the full chain remains
available as the precise description (and as a hint in the GUI).

## Stages

The chain is built from **stages**. A `Stage` is one step of the
computation: it takes an input, runs on some executor, and produces an
output for the next stage. Stages are the composable unit; the concrete
kinds we foresee:

- **EmitStage** — translate the model to an executor's language
  (Model → IR / GLSL / PTX / SPIR-V).
- **RenderStage** — an executor runs the raymarch, producing pixels (for a
  tile or the whole frame).
- **PostProcessStage** — an algorithm over pixels: SSAA, denoise, DOF,
  motion blur, tone-mapping, filters. Runs on an executor too — possibly
  the same one that rendered, possibly a different one. **Implemented** as
  of v4.21.0 (`core/postprocess/post_process.hpp`): an abstract
  `PostProcessStage` (frame → frame, may resize), `BoxDownsampleSSAA`, and a
  `PostProcessPipeline` that composes stages. `frep_multipath --ssaa N`
  applies it to the *stitched* frame, so a split supersamples correctly
  across the seam (verified bit-identical to a whole-frame supersample).
  This is the first formal stage to land; EmitStage/RenderStage are still
  implicit in the executors.
- **PresentStage** — how the result reaches the screen: offscreen readback
  into a CPU image, or direct swapchain presentation. (Often uniform across
  paths and so elided from the name.)

A path is, for now, a linear sequence of stages. This is deliberately the
simple case. The model is designed so the sequence can later become a
**graph** (see below) without invalidating the stage abstraction.

### Design intent for stages

- **Abstract base + concrete subclasses**, like the decompose/dispatch/merge
  strategies already in `multipath.hpp`. Built-in stages cover the common
  cases; users can subclass to add their own (custom post-process, a new
  executor's emit, …).
- **Composition over a fixed pipeline.** Stages compose; the pipeline is not
  hard-coded into an executor. In particular, **post-process is its own
  stage**, never baked into a RenderStage — that is what keeps paths
  honestly comparable and frame-splits seamless.
- **A stage knows its executor**, so the same post-process can be scheduled
  on whichever compute resource is appropriate, not tied to where the
  render ran.

## The data-code spectrum (compile policy)

A model parameter can live anywhere on a spectrum between pure data and
compiled code, and the right place depends on how the model is being used.

- **Classical editor (data extreme).** A small fixed visualisation algorithm
  takes the whole scene as one big data structure and interprets it. Almost
  everything is data; the code is small, static, scene-independent. Flexible
  to edit (no recompilation), but the interpreter cannot specialise for a
  particular scene.

- **Everything-is-code (code extreme).** Every value (geometry, shaders, CSG,
  the raymarch and its settings, camera, post-process) is baked into the
  generated program as a constant, so the optimiser can fold constants,
  eliminate dead code, drop x0 terms, and specialise globally. One compile,
  one run, one image. Ideal for a final render of a finished scene; unsuited
  to interactive editing, since any change means a full recompile.

- **Balanced (where FRep Designer lives).** Each parameter is either a
  constant (rarely changed -> specialise around it) or a runtime input
  (often changed -> avoid recompiling). The balance is per parameter, chosen
  by how often it changes and what the visualisation is for. A dragged value
  (a colour, a position) is best a runtime parameter; a value chosen once
  (the SSAA factor) is best a constant; the camera is a judgement call (maybe
  decided by statistics later). Where incremental recompilation is possible,
  only the affected function (e.g. one object's material) need be regenerated
  rather than the whole scene.

This is the classic compiler trade-off (specialisation vs. recompilation,
constexpr vs. variable) carried into the geometric domain. It is the direct
consequence of "a model is a program".

### CompilePolicy

`core/compiler/compile_policy.hpp` encodes this choice. A `CompilePolicy`
decides, per parameter, a `ParamPlacement` of `Constant` or `Runtime`:

- **AllConstantPolicy** -- the code extreme; everything baked. For one-shot
  final renders.
- **ByParamClassPolicy** -- runtime for a chosen set of parameter *classes*
  (Geometry, Material, Deform, Render, Other), constant for the rest.
  `ByParamClassPolicy::interactive()` is the editing default: geometry,
  material, and deform are runtime; render and observer settings stay
  constant.

Today the decision is by class. The interface is intentionally able to grow
toward per-parameter decisions, and later toward frequency- or
statistics-driven *promotion* (a value that starts constant and, once
dragged often, is promoted to a runtime input -- recompiled once, then
cheap), without changing the call sites that ask for a parameter value.

**Current status.** The policy governs geometry and deform parameters via
`FRepNode::param_value`, which now carries the parameter's class. Camera and
lighting/render settings stay off the runtime path for now -- camera is
already a runtime kernel argument (so orbiting is smooth regardless), and
render/lighting changes are rare. Material parameters are classified
`Material` and the interactive policy would place them at runtime, but the
material emitter still bakes them as constants (its values come from the BVH
build, not the node-param path); migrating materials onto the runtime buffer
is a separate, deliberate step, for when material editing needs to be
recompile-free.

## Distribution and scheduling (future)

Multi-path execution distributes the whole computation across the available
compute resources — CPUs, GPUs, and combinations — optimising for some
criterion (today: total speed; later possibly others). The current
`MultiPathExecutor` does the simple, fixed, linear case: decompose the frame
into tiles, dispatch tiles to executors, merge the results.

The longer-term picture, which this model is meant to grow into:

- **Non-fixed chains.** A render need not be a fixed sequence; it can be an
  arbitrary composition of stages and sub-algorithms that interact to reach
  the image.
- **Non-linear paths.** CPU and GPU may render parts in parallel and/or in
  sequence; the parts are combined on one or more executors (say, on the
  CPU); then a further post-process runs over the whole result (e.g. SSAA,
  DOF, a filter) — possibly on the same executors, possibly on others not
  used until then. Whether a post-process can run on the still-tiled pieces
  or only on the assembled whole is itself a scheduling question.
- **Non-trivial, possibly dynamic scheduler.** Choosing the optimal
  assignment of stages to resources is not "line up known steps and run
  them". In a network, where which resources exist and how capable they are
  may be unknown ahead of time, the scheduler cannot make a static optimal
  plan — it must be dynamic.

None of this is built now, and we deliberately constrain the current goals.
But the stage model is chosen so that today's linear `decompose → dispatch
→ merge` is recognisably a special case of a future stage graph, and so
that adding post-process as a stage, then non-linear composition, then a
dynamic scheduler, are incremental steps rather than rewrites.

## What exists today vs. this model

| Concept | Today | This model |
|---|---|---|
| Executor | `IExecutor` (CpuIr, GpuGlsl, GpuIr) | RenderStage's executor |
| Decompose/dispatch/merge | strategies in `multipath.hpp` | stages / scheduling of a linear graph |
| Post-process (SSAA) | baked into each executor/viewport, **inconsistently** | its own `PostProcessStage`, explicit and shared |
| Presentation | offscreen-readback (Viewport) vs swapchain (VulkanViewport) | PresentStage |
| Path name | executor shorthand (CPU_IR…) | full chain |

The consequence carried through the design: multi-path executors render
**without** baked-in post-process, so a compare measures executors cleanly; when
frame-split needs post-process, it becomes an explicit shared stage applied after
merge, not inside the executors. In the GUI the old per-mode picker is now a
single checkable path selector — tick one path (which uses its most efficient
backend) or several (which composite) — mirrored in the Render menu.
