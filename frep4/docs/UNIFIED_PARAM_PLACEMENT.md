# Unified parameter placement across all backends

Goal: bring every SDF-emitting backend (CPU_IR, GPU_IR, GPU_GLSL, GPU_RTX) as
close as possible to the reference CPU_IR incremental behaviour — a parameter
edit refreshes a shared runtime buffer instead of regenerating the shader/kernel
— and support a continuum of intermediate tiers between *all-constant* and
*all-runtime*, driven by a per-parameter statistic or strategy.

## What changed (new, backend-agnostic, LLVM-free, unit-tested)

* `core/compiler/compile_policy.hpp`
  * `AllRuntimePolicy` — the all-variable extreme (symmetric to `AllConstantPolicy`).
  * `ParamEditStats` — per-`(node,param)` edit heat with time decay.
  * `PerParamPolicy` — promotes a parameter to **Runtime** once its heat crosses
    a threshold, else defers to a base policy. This is the continuum: at any
    instant the runtime set is exactly the currently-hot parameters, so a scene
    sits anywhere between all-constant and all-runtime and migrates as editing
    focus shifts — without touching any emit call site.

* `core/compiler/param_binding_table.hpp`
  * `ParamBindingTable` — the **single authority** for slot assignment. One
    deterministic schema walk + a `CompilePolicy` produce `slot_of(node,param)`,
    the seed buffer, and a `placement_hash()`. Every backend reads the SAME slot
    for the SAME parameter, so the runtime buffer layout is identical across
    paths. `node_param_schema()` / `classify_param()` centralise the per-kind
    parameter order and class (previously the scattered `param_class` argument).

The placement logic is verified by `tests/standalone/test_placement_poc.cpp`
(builds with plain `g++ -std=c++23`, no LLVM/Vulkan): spectrum, statistics
promotion/demotion, determinism, cross-consumer slot identity, and
placement-hash stability under a runtime value edit.

## Integration contract for every backend

1. Build one table per scene/compile and share it across whatever backends
   render that scene:
   ```cpp
   ParamBindingTable bt = ParamBindingTable::build(to_view(scene), policy);
   ```
   `to_view` is a ~8-line recursive adapter from `FRepNode` to
   `ParamBindingTable::NodeView` (kind→int, id, &params, children).

2. Emit each parameter through one choke point: **Runtime ⇒ buffer read,
   Constant ⇒ baked literal.** CPU already does this in
   `FRepNode::param_value`; point its `slot_for_param` at `bt.slot_of` instead of
   the codegen-local lazy allocator so CPU shares the canonical layout.

3. Cache the generated artifact on `structure_hash ^ placement_hash`, **not** on
   the value-bearing source/IR. `SceneGraph::structure_hash()` already excludes
   parameter *values* (topology only); XOR-ing `bt.placement_hash()` folds in
   *which* params are runtime. Then: runtime-value edit ⇒ key unchanged ⇒
   re-upload the buffer, no regeneration; placement or constant-value or topology
   change ⇒ key changes ⇒ regenerate.

## Per-backend wiring

### GPU_GLSL — `core/gpu/glsl_emitter.cpp`, `core/exec/gpu_executor.hpp`
Add `const ParamBindingTable* bindings` to the emitter `Ctx` and a choke point:
```cpp
std::string GlslEmitter::pval(Ctx& c, const FRepNode& n, const char* name) {
    float def = n.params.at(name);
    int slot  = c.bindings ? c.bindings->slot_of(n.id, name) : -1;
    return slot < 0 ? flit(def) : ("P.v[" + std::to_string(slot) + "]");
}
```
Replace every `flit(n.params.at("x"))` with `pval(c, n, "x")`. For nodes that do
host-side arithmetic on a parameter (scale `1/s`, taper, twist/bend constants),
emit the arithmetic in GLSL when the operand is Runtime, e.g.
`"(1.0/max(1.0e-6, abs(" + pval(c,n,"s") + ")))"`. Add the buffer to the preamble
when the table is non-empty:
```glsl
layout(std430, set = 0, binding = 3) readonly buffer Params { float v[]; } P;
```
Populate `GlslEmitResult.param_bindings` from `bt.slots()`. In the executor,
upload the seed buffer once, overwrite edited slots in place, and rebuild only
when `structure_hash ^ placement_hash` changes (the GLSL source is now identical
across runtime-value edits, so this falls out naturally).

### GPU_IR — `core/exec/gpu_ir_executor.hpp`
The kernel ABI already carries `global float* params`. Set
`cfg.incremental_params = true`, `cg.set_compile_policy(&policy)`, build the
shared buffer from `bt`, and pass it to `cuda_ctx_->render(args, params, full)`
instead of the empty vector. Key the cached CUDA context on
`structure_hash ^ placement_hash`. Result: a runtime-param edit re-uploads
`params` with no PTX rebuild — matching CPU_IR.

### GPU_RTX — `core/gpu/rtx_shaders.cpp`, `core/gpu/rtx_pipeline.*`, `core/exec/rtx_executor.hpp`
Heaviest, because of descriptor/SBT wiring. Add a binding-3 std430 SSBO to the
intersection/closest-hit shaders, emit `P.v[slot]` for Runtime params there, and
bind the buffer in the descriptor set. Keep `RtPushConstants` for camera/lights;
remove the ad-hoc `sphere_radius` field once the general buffer is in place. Key
the pipeline/SBT cache on `structure_hash ^ placement_hash`. The acceleration
structure still rebuilds when a Runtime geometry edit changes an AABB — that is
inherent to RT broad-phase, not a code-gen cost, and is the one residual asymmetry
versus the compute paths (worth stating honestly in the paper).

## Status

Implemented (in this change set):

* **Shared layer** — `compile_policy.hpp` (AllRuntime / ParamEditStats /
  PerParamPolicy), `param_binding_table.hpp`, `scene_bindings.hpp`. LLVM-free,
  unit-tested (`tests/standalone/test_placement_poc.cpp`, plain `g++`).
* **GPU_GLSL** — `glsl_emitter.{hpp,cpp}` routes every geometry/deform parameter
  (both the SDF and the dual-AD normal path) through `pval()`; runtime params
  become `P.v[slot]` reads from a binding-3 std430 buffer, constants still fold.
  `vulkan_ctx.{hpp,cpp}` gained the binding-3 SSBO + `update_params()` (a cheap
  memcpy into a persistently-mapped buffer). `gpu_executor.hpp` builds the table
  and calls `update_params()` each frame.
* **GPU_IR** — `gpu_ir_executor.hpp` enables incremental params and feeds the
  kernel's existing `float* params` buffer; the cached PTX is now invariant to
  runtime-value edits.

All three GPU paths are **opt-in**: with no policy installed (the default), the
binding table is empty, no binding-3 buffer is emitted, and the output is
bit-identical to before — so the change is safe to merge and is exercised only
when a policy (`ByParamClassPolicy::interactive()`, `PerParamPolicy`, …) is set
on the executor via `set_compile_policy()`.

Validated on a software Vulkan rasterizer (Mesa lavapipe, LLVM 20):

* `frep_parity_check --paths cpu_ir,gpu_glsl` — 17/17 scenes within tolerance
  (mean|Δ| ~7e-4), including the rewritten twist/bend/taper/rotate/scale/
  smooth_union, confirming the `pval`-routed emitter is correct with no policy.
* `frep_validate_runtime_params` (new tool) — with an interactive policy a
  runtime-parameter edit leaves the shader byte-identical (compile 124 ms first
  frame → 0.10 ms after the edit) and the buffer-path image is bit-identical to
  a full recompile (max|Δ| = 0.0), while the edit still changes the frame.

Still requires real hardware (not the software path): the CUDA PTX path
(GPU_IR) and the RT path (GPU_RTX).

Remaining (next increment, pattern now established):

* **GPU_RTX** — apply the same `pval`-style choke point in `rtx_shaders.cpp`,
  add a binding-3 SSBO to the intersection/closest-hit shaders and the descriptor
  set, and route the executor through `update_params`. The acceleration structure
  still rebuilds when a runtime *geometry* edit moves an AABB — inherent to RT
  broad-phase, the one residual asymmetry versus the compute paths.
* **CPU canonical layout** — point `SceneCodegen::acquire_param_slot` at a shared
  `ParamBindingTable` so the CPU JIT buffer uses the identical slot order as the
  GPU buffers (today each backend's layout is self-consistent; this makes them
  byte-identical, which matters only if a single buffer is shared across paths).
