# Changelog

## [4.53.0] — GPU_RTX: instancing Level 2 + interval cull (cross-path consistency complete)
- Instancing Level 2 now works on the RTX path. Root cause of it being broken before: the RT intersection/closest-hit shaders lift the shared GLSL region (scene_sdf + helpers) from the compute emitter, but the lift started at "float scene_sdf" — and the shared _inst_fn_N / _inst_grad_fn_N subprograms are emitted *before* scene_sdf and called by it. So an instanced scene produced an intersection shader that called _inst_fn_0 without defining it (wouldn't compile). Fixed by starting the lift at the earliest instance function (added _inst_fn_ / _inst_grad_fn_ to the same earliest-helper scan already used for mesh/texture helpers). Verified: the intersection shader now includes the _inst_fn_0 definition before its first use and compiles to SPIR-V (glslang, vulkan1.2 RT stages); a non-instanced scene is unaffected (no instance functions lifted, still compiles).
- Interval tile-cull analog on RTX: an interval pre-skip in the intersection shader. The hardware BVH already does object-level broad-phase (a ray only runs the intersection shader for AABBs it hits), so the per-tile slab cull has no direct RTX analog — but within a hit AABB the sphere-trace still crawls empty space. The pre-skip bounds the field over the AABB of the ray's [tmin,tmax] segment via the lifted sdf_ival() and returns immediately when the interval can't straddle 0 (surface not in this segment). Emitted only when interval culling is on and sdf_ival was lifted. Compiles to SPIR-V.
- Verified by REAL rendering, not just shader validity: the lavapipe software Vulkan device exposes VK_KHR_ray_tracing_pipeline, so the RTX path runs on CPU. RTX interval cull on-vs-off on a twisted box is byte-identical (max diff 0.0000); RTX Level-2 vs Level-1 on a 4-instance scene is byte-identical (max diff 0.0000); non-instanced no-cull RTX still renders. Performance on real NVIDIA RT hardware is for the user to confirm; the software RT device establishes correctness + shader validity.
- Cross-path consistency is now complete: instancing (L1 + L2) and tile cull (Lipschitz + interval) run on all four paths — CpuIr, GpuIr, GpuGlsl, GpuRtx. FEATURE_MATRIX updated to match.

## [4.52.0] — interval tile cull in the IR paths (the LLVM-IR interval emitter)
- Added the third interval implementation: an LLVM-IR interval emitter (core/compiler/node_interval_ir.hpp), alongside the existing CPU evaluator (node_interval.hpp) and GLSL string emitter (glsl_node_interval.hpp). Each interval is a pair of llvm::Value* (lo, hi) and every operation (mul, add, sub, sqrt, abs, div, min, max, sin, cos with the crosses-extremum saturation) emits LLVM instructions. Same node kinds and rules as the other two — verified endpoint-identical to the CPU evaluator (max error 0.000000 over 2000 random boxes per node type across sphere/box/csg/smooth/rotate/scale/twist/taper).
- This unblocks interval tile culling on the IR paths (CpuIr/GpuIr). The per-tile cull now emits a scene_ival(lo3,hi3)->{lo,hi} function (union of visible objects, min of their interval bounds) once and calls it per slab; a slab is occupied when its field interval spans 0. Interval culling is sound for non-metric trees (twist/bend/taper/CSG), which Lipschitz L=1 is not — so the IR paths can now cull those correctly. Auto resolves to Lipschitz on metric trees (cheapest, exact) and interval otherwise, matching the GLSL Auto resolver.
- Verified on CpuIr (identical IR to GpuIr): interval-cull IR verifies (NVPTX-ready) and emits scene_ival; a full-frame render of a twisted box with interval cull on is byte-identical to cull off (max diff 0.0000, 0 differing components); Auto correctly emits scene_ival for a non-metric scene and omits it (uses Lipschitz) for a metric one, both byte-identical to no-cull; a miss-tile on the twisted scene is ~93x faster (sphere tracing crawls through the non-metric field otherwise, and the interval cull turns the whole empty tile into an instant miss). GLSL path unaffected (0.0000).
- Tile cull is now consistent across CpuIr, GpuIr, and GpuGlsl for both methods. FEATURE_MATRIX updated. The only remaining cull gap is GPU_RTX; instancing Level 2 on GPU_RTX is also still pending. Those two are the last cross-path items.

## [4.51.0] — per-tile Lipschitz cull in the IR paths (CpuIr/GpuIr)
- The IR raymarch paths now support a per-tile Lipschitz occupancy cull, the IR analog of the GLSL tile cull. Once per tile (before the pixel loops), [near, max_dist] is split into cull_slabs depth slabs; for each slab an AABB is built enclosing the tile frustum's four corners at the slab's near and far depth, and the slab is marked occupied when the surface can pass through it (|scene_sdf(center)| <= L * halfdiagonal). The occupied slabs' depth span then clamps every ray's march in the tile. This is emitted in LLVM IR (unrolled over the slabs, reusing the existing scene_sdf function), so it lowers to NVPTX for GpuIr as well.
- Scope: emitted only when cull_slabs > 0 and the method resolves to Lipschitz (or Auto on a metric tree, where L=1 is exact and sound). Interval cull stays GPU_GLSL-only — it would need an LLVM-IR interval emitter (the CPU and GLSL interval evaluators exist, but there's no IR interval codegen yet); on the IR paths an Interval selection simply falls back to no tile cull rather than culling incorrectly. Default is cull_slabs = 0 (off), so nothing changes unless enabled; the existing Render-tab cull controls drive it.
- Unlike the whole-scene ray-box clip (4.48.0, which only trims empty space outside the scene box), this can skip whole tiles that miss the geometry. Verified on CpuIr (identical IR to GpuIr): a tile that entirely misses the geometry becomes an instant miss (~3.4x faster on that tile); tiles over the surface are unchanged (correct — there's real marching to do there); a full-frame render with cull on is byte-identical to cull off (max diff 0.0000, 0 differing components); the cull IR verifies (NVPTX-ready); the GLSL path is unaffected (0.0000).
- FEATURE_MATRIX updated: Lipschitz tile cull is now CpuIr/GpuIr/GpuGlsl; interval tile cull remains GLSL-only; instancing Level 2 is now marked done on the IR paths too (from 4.50.0). Remaining for full cross-path consistency: interval tile cull on the IR paths (needs the IR interval emitter), and both tile cull + instancing Level 2 on GPU_RTX.

## [4.50.0] — instancing Level 2 (shared subprograms) in the IR paths
- The IR raymarch paths (CpuIr and GpuIr, which share the same emitted LLVM IR) now support instancing Level 2: an instanced target's geometry is emitted ONCE as a shared LLVM function and called, instead of being inlined into every instance. Previously the IR paths always inlined instances (Level 1) — correct, but the code grew with the instance count. This is the IR twin of the GLSL emitter's _inst_fn_N / _inst_grad_fn_N sharing, so instancing is now consistent across CpuIr, GpuIr, and GpuGlsl (all three share; GPU_RTX still pending).
- Both the scalar SDF path and the dual-number AD (normal) path share: an instance emits a call to inst_geom_N (scalar) and inst_grad_N (returns a {value, derivative} struct seeded with the incoming duals). Memoised by target pointer, registered before the body is emitted so a self/cyclic reference resolves to a call rather than recursing.
- Mechanism: CgCtx gained instance_call / instance_grad_call callbacks; InstanceNode::codegen / codegen_grad emit a shared call when the context provides them, and fall back to inlining otherwise (so paths that don't set the callback are unchanged). SceneCodegen owns the memo tables and binds the callbacks in make_cgctx, gated on the existing instance_shared_subprograms flag (the same one that drives GLSL Level 2). The shared functions thread the params buffer through so incremental-mode runtime params still work.
- Verified on CpuIr (identical IR to GpuIr): 8 instances of a twisted box emit exactly 1 inst_geom + 1 inst_grad definition (called 8x each) instead of 8 inline copies; a full render — SDF and normals, so both shared functions exercised — is byte-identical to the Level 1 inline path (max diff 0.0000); the IR verifies, so it lowers to NVPTX for GpuIr; non-instanced scenes and the GLSL path are unaffected (0.0000). IR size shrinks 24% (N=8) → 36% (N=64); the smaller-than-GLSL figure is because the IR carries more fixed boilerplate (the whole march loop, shading, AABB guards) that isn't shared.
- Remaining for full cross-path consistency: instancing Level 2 on GPU_RTX, and per-tile Lipschitz/Interval cull on the IR paths (the ray-box clip from 4.48.0 covers empty space; per-tile occupancy inside the bounds is the remaining piece) and on GPU_RTX.

## [4.49.0] — Node Graph editor: reflect the new node types
- The Node Graph catalog (node_types.hpp) was stuck at the original set (Sphere/Box/Plane, the CSG ops, Translate/uniform-Scale/RotateY). It now includes everything added since: RotateX and RotateZ, non-uniform Scale (sx/sy/sz replacing the single s), and a new Deformation category with TwistY, BendXY, TaperY (teal), plus an Instance node type (pink).
- The graph→FRepNode factory (node_graph.cpp) builds all the new types, and Scale now constructs the non-uniform ScaleNode(sx,sy,sz). The right-click palette gained a "Deformations" submenu; Transforms lists RotateX/Y/Z.
- Instances are now visible in the graph: an InstanceNode renders as a pink node whose subtitle is "→ target_id". Its shared target subtree is deliberately NOT re-expanded (that would duplicate the whole target in the graph and loop forever on a cyclic reference), and it isn't offered in the create palette (an instance needs a target object — it's created from the Scene toolbar). NodeItem gained an optional subtitle for this.
- Verified: the catalog exposes all eight new/updated types with the right category, input count, and params (Scale = 3 params); the existing node-graph round-trip and cycle-detection test still passes; the GUI builds and starts clean.

## [4.48.0] — ray-box near/far clip for the IR paths (CpuIr/GpuIr)
- The IR raymarch (shared by CpuIr and GpuIr) now clips each camera ray to the scene bounding box before marching: a slab-method ray-box intersection sets the march's start t (box entry) and far bound (box exit), so rays skip the empty space before and after the scene instead of sphere-tracing to max_dist. This is the IR-path analog of the GLSL tile cull. Emitted in LLVM IR (min/max intrinsics for the slab envelope); the IR verifies, so it lowers to NVPTX for GpuIr too.
- Scene AABB is the union of visible objects' AABBs, plus a 5% margin so grazing rays and the surface standoff aren't clipped. If any visible object is unbounded (a Plane, or any node whose aabb() is infinite), the clip auto-disables — an infinite box can't narrow anything — and the march falls back to [near, max_dist]. TracerConfig::bbox_clip (default on) toggles it; a "BBox clip (IR)" checkbox sits by the cull controls in the Render tab.
- Measured on CpuIr (renders identically — a 2-pixel silhouette FP difference, i.e. the geometry is unchanged, only sub-pixel grazing edges shift): small sphere in a large view 6.1 ms → 2.4 ms (2.5x); a twisted box−sphere that otherwise marched a non-metric field to max_dist 396 ms → 3.7 ms (~100x). A scene with a floor plane correctly disables the clip (0.0000 diff). The GLSL path is byte-unaffected (the clip is IR-only; GLSL keeps its tile cull, which bounds the march more tightly than a whole-scene box).
- FEATURE_MATRIX updated: the IR-path gap is now "no per-tile occupancy cull inside the bounds" rather than "no cull at all" — the empty-space part is closed.

## [4.47.1] — real GPU frame time in the metrics HUD
- Correction to the v4.45.0 note: the real-time Vulkan path DID already measure GPU frame time — a VkQueryPool timestamp pair wraps the compute dispatch and the result was sampled ~10 Hz and emitted as render_time_sampled — it simply was not plumbed into the metrics HUD. Now VulkanViewport captures the sampled value and metrics_text() reports actual GPU ms + fps alongside the cull method. No new GPU instrumentation was needed; it was a wiring gap, not a missing measurement.

## [4.47.0] — 3-axis rotation gizmo
- The Scene inspector's single "Rotation Y" spinbox is replaced by three (X/Y/Z). The scene gizmo now keeps a canonical RotateX→RotateY→RotateZ chain between the Translate and Scale slots, so the three axes are independent: setting one inserts/updates/removes its node in place without disturbing the others. New SceneGraph::set_rotation_axis/get_rotation_axis (axis 0/1/2) and an undoable SetRotationAxisCommand; a spinbox edit commits only the axis that actually changed.
- The gizmo walks (set/get scale, both uniform and per-axis) now skip the whole rotation run rather than a single RotateY, so Scale is still found and editable beneath any combination of rotations. gizmo_tag recognises RotateX/RotateY/RotateZ.
- Verified: setting X/Y/Z independently, updating one while the others persist, removing an axis (angle 0) while the rest stay, scale still working below the chain, the SDF compiling, and a full save/load round-trip of a triple-rotated + non-uniformly-scaled object.
- The legacy set_rotation_y/get_rotation_y and SetRotationCommand remain for compatibility (they drive the Y axis of the same chain).

## [4.46.0] — Scene-tab property grid
- The Scene tab now has a "Properties" grid: a tree of the selected object's geometry nodes and their numeric parameters, with an editable Value column. Each node shows as "type (id)" with one row per parameter; editing a value commits an undoable SetParamCommand (parameter-only, so it rides the incremental recompile path — no full rebuild for a slider-like tweak). Rebuilt on selection change and scene edits; empty for multi/no selection.
- Core support: SceneGraph::set_node_param / get_node_param locate a node by id inside an object's geometry tree (new find_node_by_id helper) and mutate one parameter, bumping the scene revision. Verified: editing a nested sphere's radius and a twist rate through the API updates the compiled SDF; a missing node id fails gracefully.

## [4.45.1] — GUI tidy: Plugins tab, Render-tab cleanup
- Moved "Registered plugins" out of the Render tab into its own dedicated "Plugins" tab (build_plugins_panel), sitting between Lights and LAN.
- Removed the "Controls:" help blurb (orbit/zoom/add-primitive) from the Render tab — it duplicated behaviour that is discoverable and cluttered the panel.
- Housekeeping: removed stray node*.ppm files (cull benchmark render outputs from gpu_cull_bench) that had accumulated in the project root. They were already covered by .gitignore (*.ppm), so this only removes the on-disk clutter, not anything tracked.

## [4.45.0] — Batch D: docs + gallery refresh (task 8) and metrics HUD (task 2)
- Documentation brought current with this session's work:
  * FEATURE_MATRIX.md — the geometry table now lists RotateX/RotateY/RotateZ and non-uniform Scale, and adds Instance in two rows: Level 1 (shared-reference semantics) works on all four paths via the virtual codegen; Level 2 (shared subprogram, the memory win) is GPU_GLSL-only. A new "Acceleration & debug" table records that tile cull and the debug views (step heatmap, cull span) are GPU_GLSL-only by construction. The honest-gap summary is updated: the real remaining gap is that tile cull and instancing Level 2 don't exist on the IR paths (not a regression — GLSL-emitter features — but real if the IR paths are to match GLSL's memory/traversal efficiency).
  * USER_GUIDE.md — node-type table updated for non-uniform Scale (sx/sy/sz), RotateX/RotateZ, and Instance (target_id).
- Gallery regenerated (all images were stale — several shading/geometry fixes since the last render) and a new showcase added: scene 11 "Instancing, non-uniform scale, multi-axis rotation" (tools/gallery.cpp scene_instances()). It shows one twisted box authored once and reused as five live instances, three ellipsoids each from a non-uniform Scale on a different axis, and a cube tilted with stacked RotateX/Y/Z. GALLERY.md documents it; the intro count is now eleven scenes.
- Metrics HUD (task 2) now actually draws: a translucent monospace label pinned to the top-left of the viewport, refreshed ~4x/sec from the active viewport's new IViewport::metrics_text(). The executor viewport reports per-path ms + fps (and "ERROR" for a failing path); the real-time GLSL viewport reports the tile-cull method in use. Honest limitation: the real-time Vulkan path still lacks a per-frame ms readout — that needs a GPU timestamp query around the compute submit, which isn't wired yet — so on that path the HUD shows the cull configuration rather than a frame time. The toggle, overlay, and cull-method display work; the GPU-timer piece is the remaining part of task 2.

## [4.44.0] — Batch C (task 5): RotateX / RotateZ + non-uniform Scale
- New node kinds RotateX and RotateZ (transforms.hpp), mirroring RotateY: RotateX rotates (y,z) about the X axis, RotateZ rotates (x,y) about Z. Wired through every path — scalar codegen, dual-number AD (operations_ad.cpp), AABB (aabb.cpp), scalar GLSL emit, dual GLSL emit, CPU node-interval and GLSL node-interval, the unit-Lipschitz predicate (rotations are metric), and serialization. Verified: RotateZ 90° maps a long-X box onto Y (f(0,0.8,0)=-0.100, f(0.8,0,0)=+0.600); RotateX 90° maps Y onto Z; the CPU interval encloses the field with 0 violations over 50k random boxes; all three axes render on the GPU.
- Fixed a latent bug found while doing this: the RotateY interval (both CPU node_interval and GLSL node_interval) read the angle from a param named "angle", but the node stores it as "a" — so interval culling always saw a 0° rotation for RotateY. Now reads "a" everywhere.
- ScaleNode extended to non-uniform: it now stores per-axis sx/sy/sz. Two constructors — the old ScaleNode(child, s) still exists and sets sx=sy=sz=s (fully backward-compatible), plus a new ScaleNode(child, sx, sy, sz). Non-uniform scale is NOT distance-preserving, so the SDF is multiplied by the SMALLEST axis factor to stay a conservative (never-overshooting) Lipschitz field. Updated across all paths (codegen, AD, AABB, GLSL, CPU+GLSL interval, the Scale<1 metric predicate now checks every axis, serialization). Verified: uniform ×2 still gives a sphere of double radius; non-uniform (2,1,1) stretches only X into an ellipsoid; interval enclosure holds with 0 violations for non-uniform factors; save/load round-trips; both render on the GPU. Existing uniform-scale scenes render byte-identical after the s→sx/sy/sz change.
- GUI: the Scene inspector's single "Scale" spinbox is replaced by three (X/Y/Z) for non-uniform scaling, committed via a new undoable SetScaleXYZCommand; the scene gizmo gained set_scale_xyz/get_scale_xyz. Backward-compatible uniform SetScaleCommand/set_scale still exist (they set all three axes equal).
- Scoping note: the inspector's rotation gizmo is still the single Y control. The fixed Translate→RotateY→Scale gizmo slot only holds one rotation; stacking three Euler rotations in that gizmo is deferred. RotateX/RotateZ are fully supported by the engine and can be applied via the node graph / API today; the dedicated inspector X/Z rotation spinboxes are the remaining GUI piece.

## [4.43.0] — Batch B: render-path audit (task 9) + failure surfacing (task 6)
- Task 9 audit — which features live in which of the four render paths (CpuIr, GpuIr, GpuGlsl, GpuRtx):
  * Shadows (enable_shadows): gated once in the SHARED codegen (codegen.cpp: shadow_fn = enable_shadows ? emit_shadow() : nullptr), so CpuIr and GpuIr behave identically; GpuGlsl has its own equivalent. Verified that toggling enable_shadows changes the emitted LLVM IR (shadow fn present/absent), so the GpuIr PTX cache — keyed on IR text — correctly rebuilds on the toggle.
  * Instancing Level 1 (shared-pointer reference semantics): works in ALL paths automatically, because codegen.cpp emits geometry through the virtual FRepNode::codegen() and InstanceNode::codegen() delegates to its shared target. Verified an instanced scene renders correctly through the CpuIr JIT.
  * Tile cull and instancing Level 2 (shared GLSL subprograms) are GpuGlsl-only by construction — they are GLSL-emitter features. The IR paths (CpuIr/GpuIr) don't have a tile cull; this is a known gap, not a regression, and is noted for future parity work if the IR paths need it.
  * Camera: both families pass scene.camera() as per-frame runtime args (GpuIr: cam_pos/fwd/right/up + fov_scale; GpuGlsl: push constants), so a camera move is a re-upload, not a recompile.
- Task 6 (shadows/camera "not reflected" on GPU_IR): the config path was verified correct end-to-end — GUI toggle -> render_config_ -> set_tracer_config -> executors cleared and rebuilt with the new cfg -> codegen honours it -> IR changes -> PTX cache busts. The most likely real cause is a path that FAILS silently: the executor viewport did `if (!r.ok) continue;`, keeping the previous frame's pixels, which looks exactly like "the setting did nothing". Fixed by capturing the first render error per path and showing it in the status text ("GPU_IR: ERROR (…)") instead of silently displaying stale pixels — so an unavailable/failing backend is now visible rather than masquerading as a working render that ignores settings.
- No engine behaviour change for working paths; diagnostics + audit only.

## [4.42.0] — instancing Level 2: shared subprograms (the memory win)
- Instanced geometry is now emitted ONCE as a shared GLSL function and called, instead of inlining a copy at every instance — so the emitted code (and its recompile cost) grows with the number of *distinct* shapes, not the total instance count. This is the point of instancing for large repetitive models such as a detector geometry.
- All three per-object geometry bodies are functionalised: the SDF (float _inst_fn_N(x,y,z)), the dual-number AD normal (Dual _inst_grad_fn_N(Dual x,y,z) — takes incoming duals so AD stays correct through an outer transform), and the albedo distance test (reuses the SDF function). Dedup is by the shared node pointer: an InstanceNode's children[0] is the same pointer as the target object's geometry root, so the original object and all its instances map to one function. A pre-pass over the scene marks which roots are instance targets; only those become functions (a plain object with no instances stays inlined).
- Measured emitted-shader size, copy vs instance, for a twisted box−sphere replicated N times: N=8 38% smaller, N=16 49%, N=32 56%, N=64 60%. The saving grows with N toward the fixed-boilerplate asymptote. Different materials per instance still work (geometry shared, colour per-object).
- Correctness: Level 2 (functions) renders pixel-identical (0.0000) to Level 1 (inline) of the same instanced scene, on the GPU path. Non-instanced scenes are byte-for-byte unaffected (the pre-pass finds no targets). Fully-functionalised instanced shaders validate to SPIR-V.
- TracerConfig::instance_shared_subprograms (default on) toggles Level 2 vs forced inlining, exposed as a "Share instances" checkbox in the Render tab. Off is useful to trade code size for avoiding the function-call overhead or the sub-pixel silhouette FP differences that inline-vs-call can produce.
- Interactive/live editing preserved from Level 1: instances share the target pointer, so editing the target updates every instance; a parameter edit flows through the runtime buffer with no recompile, and instances follow for free.

## [4.41.0] — true instancing, Level 1 (task 4, corrected)
- Replaced the earlier mislabeled "Instance" (which deep-cloned = a copy) with real instancing. New NodeKind::Instance / InstanceNode (core/frep/instance.hpp): an instance references another object's geometry by id and *shares* its subtree pointer rather than copying it. In FRep terms the target's SDF is a function already in the scene and an instance is a call to it. Semantics (ii): the instance references the target's bare geometry and applies its own transform in full (the usual Translate/Rotate/Scale wrap it).
- Live editing (required — this is an interactive modeller): because instances share the target pointer, editing the target's nodes is seen immediately by every instance. MainWindow::on_scene_changed() calls resolve_instances() after every edit so that even a structural swap (SetGeometryCommand replaces a root pointer) rebinds instances to the new geometry and the link holds. Verified: changing the target sphere's radius updates the instance's rendered radius.
- Reference integrity: resolve_instances() runs cycle detection over the instance reference graph (id -> targets) and leaves any cyclic (A<->B or self) or dangling instance unresolved so it renders empty instead of recursing forever in codegen. Deleting a target cascades to its instances via find_dependent_instances(), with a GUI confirmation listing the affected instances before removal.
- Serialization: an Instance saves only its target_id (never an inline copy of the shared subtree, which would defeat instancing and break sharing); deserialize_scene() rebinds pointers via resolve_instances() after all objects are parsed. Round-trip verified.
- All emit paths delegate through the shared child: scalar GLSL, dual-number AD GLSL, CPU node-interval and GLSL node-interval, plus node_is_unit_lipschitz. Verified: an instanced scene renders pixel-identical (0.0000) to the equivalent copied scene on the GPU path, and matches the CPU JIT.
- NOT yet done — Level 2 (the memory win): codegen still inlines the shared subtree at each instance, so emitted-code size does not yet shrink with instance count. Level 2 will emit the target once as a GLSL subprogram and call it; the shared children[0] pointer is the dedup identity for that. Tracked as the next step.

## [4.40.0] — GUI batch A (tasks 1,3,4,7; 2 partial)
- Task 7 (deformations from the GUI): the Scene toolbar now has Twist/Bend/Taper buttons that wrap the selected object's geometry in the corresponding deformation node via an undoable SetGeometryCommand. Previously the deformation node kinds existed but could not be applied from the UI at all.
- Task 4 (instancing): a Scene-toolbar "Instance" button deep-clones the selected object(s) (io::clone_node, handles plugin/mesh/expr subtrees) under fresh ids, offset so the copy is visible, added as independent objects. Added SceneGraph::find_object() accessor.
- Task 1 (cull control from the GUI): the Render tab gained a "GPU tile cull" group — method (Auto/Lipschitz/Interval/Off), slab count, and Lipschitz L — all live through render_config_ -> apply_render_config(), so the effect shows while orbiting.
- Task 3 (debug visualisation): TracerConfig::DebugView = {Off, StepHeatmap, CullSpan}, emitted in the GLSL march. StepHeatmap colours each pixel by march-iteration count (blue few -> red many); CullSpan colours by the fraction of the ray marched after culling (green mostly-culled -> red little). Exposed via the Render-tab "Debug view" combo. Verified on llvmpipe: both views render correctly (silhouette shows hot in the step map; empty regions show green in the cull-span map).
- Task 2 (on-screen metrics) — PARTIAL: the toggle, IViewport::set_metrics_overlay() hook, and render_config plumbing are in, but the actual on-viewport HUD text drawing in the real-time viewport is not yet implemented. ms/frame is already measured; cull-rate readback from the shader is still to be added.
- Build note: frep_designer requires -DFREP_BUILD_GPU=ON (the GUI's real-time viewport links GlslEmitter/VulkanCtx from core); added explicit Vulkan::Vulkan to the GUI link so the viewport's direct Vulkan calls resolve under --whole-archive.

## [4.39.2]
- Added interval cull support for the two remaining deformations, BendXY and TaperY, in both the CPU node_interval.hpp and the GLSL NodeIntervalEmitter (previously they returned an infinite interval — correct but pruning nothing). TaperY: is=1/max(1+clamp((y+h/2)/h,0,1)*(t-1),1e-3) is an interval, x'=x*is / z'=z*is are interval products (tight). BendXY: th=ks*x and r=1/ks+y are intervals, x'=r*sin(th) / y'=r*cos(th)-1/ks use interval trig (sound, loose like the twist), with the /max(1,|ks*r|) Lipschitz correction as an interval divide. CPU and GLSL are rule-for-rule mirrors; a random-box dense-sampling enclosure check on BendXY showed 0 violations.
- Widened the cull safety margin from 2 to 4 depth-slabs (still absolute, 4*range/32). BendXY bends the geometry sharply enough that grazing rays needed more than a 2-slab pad at fine slab counts (diff 0.30 on node:bend at 64 slabs); 4 slabs fixes it with no measurable cull-rate loss on the other scenes. Verified all scenes — csg/twist/blend/bend/taper and the five canonical — pixel-identical (0.0000) to cull-off under Auto at 64 slabs, and BendXY sound through 128 slabs.
- All node deformations (Twist/Bend/Taper) now have a working interval cull; node trees have full method coverage.

## [4.39.1]
- Reverted Auto to topology-based selection: metric tree -> Lipschitz L=1 (sound, tightest, and cheapest per-box test), non-metric -> Interval (sound without an L). The v4.39.0 cull-rate probe was removed from the decision because cell count is the wrong proxy — it ignores the ~2x per-box cost of the interval test and so mis-selected Interval on the metric CSG scene, where measurements showed Lipschitz is actually faster end-to-end (1.08x vs 1.01x on the user's iGPU). Topology gives the correct call for the intended clean-SDF workloads: csg/blend -> Lipschitz, twist -> Interval, all matching the faster measured method.
- The probe helpers (node_interval.hpp, octree_leaves_node_interval) are kept and documented as raw material for a future *timed* selector (render a few frames each way, keep the faster) — the correct-but-heavier approach. A TracerConfig::cull_auto_timed_probe opt-in flag is wired through the executor as the enable point; it currently falls back to topology (inert, not wrong) until scene data shows the timed probe earns its per-scene cost.
- Verified pixel-identical (0.0000) to cull-off under Auto for node:csg/twist/blend and all five canonical scenes at 64 slabs.

## [4.39.0]
- Auto cull method is now probe-based for metric trees, not topology-based. New core/compiler/node_interval.hpp is the CPU twin of the GLSL NodeIntervalEmitter (same interval rules for every node kind), so resolve_cull_method() can measure a node tree's interval cull rate without a GPU. For a metric tree Auto now octree-probes both bounds on a coarse grid (octree_leaves_lipschitz vs octree_leaves_node_interval) and picks Interval only when it leaves >=20% fewer cells to march, else the cheaper Lipschitz L=1. Non-metric trees still go to Interval (sound without an L).
- Effect: the choice is per scene by measured cull rate, not assumed from node types. Probed cull cells (49^3 grid): csg 28672 interval vs 45824 Lipschitz -> Auto picks Interval; blend 38400 vs 17920 -> Lipschitz (the SmoothUnion interval bound is loose); sphere 23552 vs 26624 -> Lipschitz (within the 20% margin, cheaper wins).
- Verified on llvmpipe: node:csg / node:twist / node:blend and all five canonical scenes render pixel-identical (0.0000) to cull-off under Auto at 32 and 64 slabs.

## [4.38.1]
- gpu_cull_bench accepts built-in node-tree scenes as "node:<name>" (csg | twist | blend), which have no JSON form (JSON scenes are single CustomExpr) and exercise the per-node interval cull. Output filenames sanitise the colon.
- Measured on llvmpipe (node:twist, 64 slabs, both pixel-identical to cull-off): interval cull 2.92x vs Lipschitz L=1 0.83x — a twisted (non-metric) field gives a very loose Lipschitz box, so it prunes little, while the interval bound prunes aggressively. Confirms the per-node interval path is the right cull for non-metric node trees, not just a correctness backstop.

## [4.38.0]
- Interval tile cull now works for ANY node tree, not just a single CustomExpr. New core/gpu/glsl_node_interval.hpp (NodeIntervalEmitter) mirrors GlslEmitter::emit_node with interval arithmetic: metric primitives (Sphere/Box/Plane) and rigid transforms (Translate/RotateY/Scale) are exact; booleans/blend compose by min/max/negate; non-metric deformations (TwistY) use interval trig on the rotated coordinates (sound, loose); CustomExpr sub-nodes defer to the expression interval emitter with the coordinate box substituted. The GLSL emitter emits scene_sdf_ival by combining the visible objects with min(), exactly as scene_sdf.
- Both cull methods are now available for every scene: TracerConfig::CullMethod::{Auto,Lipschitz,Interval,Off} all apply to node trees. resolve_cull_method's interval_ok is no longer restricted to single-CustomExpr scenes; Auto picks Lipschitz L=1 for a metric tree (exact, cheapest) and Interval otherwise (sound without an L, and unlike a Lipschitz estimate it never under-bounds the gradient).
- Verified on llvmpipe: node-tree CSG (box - sphere, translated) and a non-metric twisted box render pixel-identical (0.0000) to cull-off under Auto/Interval/Lipschitz at 32 and 64 slabs; all five canonical CustomExpr scenes remain 0.0000 after the widening. SPIR-V validated (glslangValidator) for the node-tree interval shader.

## [4.37.2]
- gpu_cull_bench now interleaves cull-off / cull-on frame by frame (A/B) instead of timing all-off then all-on, and reports the median of per-frame-pair ratios. On a noisy GPU the frame-time variance between two sequential phases (clock boost/thermal) was larger than the cull's effect, so the ratio swung with baseline drift rather than slab count (the cull-off baseline moved 3.6-4.7 ms across processes although slab count cannot affect a disabled cull). Interleaving makes the drift common-mode to both configs; the per-pair ratio cancels it.
- No engine change; measurement methodology only. Confirms the physical expectation: the tile cull is a real win on sparse scenes (s2_csg ~1.14x) and net-neutral on the dense gyroid (~1.0x), independent of method — there is simply little empty depth to skip when the surface fills the view.

## [4.37.1]
- GpuGlslExecutor caches the resolved Auto cull method per scene pointer instead of re-running jit::resolve_cull_method() every frame. The choice depends on scene topology, not edited parameters, so it is recomputed only when the scene changes. Removes a per-frame probe from the render path (visible on fast GPUs, hidden under a software rasteriser). Verified Auto and forced Interval now render within noise of each other on the gyroid (llvmpipe), both pixel-identical to cull-off.

## [4.37.0]
- Tile cull now has both box bounds selectable per scene, not one hard-wired: TracerConfig::CullMethod = Auto | Lipschitz | Interval | Off. Lipschitz (f(box) in [f(c)-L*r, f(c)+L*r]) works for any node tree; Interval (interval arithmetic, sound without an L) is emitted for a single-CustomExpr scene. Auto picks per scene via node_is_unit_lipschitz(): a metric tree -> Lipschitz L=1 (exact, cheapest); a non-metric single CustomExpr -> Interval (sound by construction). The GLSL emitter emits sdf_ival from the CustomExpr AST when Interval is chosen; jit::resolve_cull_method() resolves Auto with the JIT SDF (kept out of the emitter) and the GPU executor calls it before emitting.
- Auto deliberately prefers Interval over a Lipschitz bound with an *estimated* L for non-metric fields: the estimate is not a guaranteed upper bound on |grad f| and silently culls real surface (the gyroid at fine slab counts). Interval needs no L.
- Fixed the cull margin: it was expressed in slabs (2*st), so at high slab counts the absolute depth padding shrank below what a grazing gyroid ray needs and the cull removed real surface (diff 0.74 from ~48 slabs up). The margin is now a fixed depth (2 * range / 32) independent of slab count.
- gpu_cull_bench gained a method argument (auto|lipschitz|interval|off) to force a path for comparison. Verified on llvmpipe: cull-on vs cull-off pixel-identical (0.0000) for all five canonical scenes under Auto at 32 and 64 slabs, and interval gyroid sound through 96 slabs. GPU wall-clock still measured on the user's host.

## [4.36.2]
- Fixed the tile cull rendering holes on the silhouette (max pixel diff 0.72 on s2_csg at L=1, a true SDF that should have been sound). Two fixes, both verified on a software Vulkan device (llvmpipe): (1) the kept depth span is padded by two slabs on each side, so a grazing ray whose hit lands in the boundary slab is not culled; (2) the per-slab box tests are now spread across the 64 workgroup lanes (lane s does slabs s, s+64, ...) with a shared occupancy array reduced by lane 0, instead of lane 0 doing all of them behind the barrier — which had made the pass net-negative (0.67-0.76x) on cheap scenes.
- Verified cull-on vs cull-off pixel-identical (max diff 0.0000) for s1_sphere, s2_csg, s3_blend, s5_twist at L=1 across 16/32 slabs; frame time now break-even to ~1.05x on llvmpipe (a software rasteriser where the 64 lanes are not truly concurrent, so this is a floor — a real GPU should net-gain on sparse scenes).
- Confirmed the Lipschitz caveat empirically: the gyroid (|grad f| ~ 6) is culled correctly at L=1 only while slabs are thick enough for the 2-slab margin to absorb the gradient error; at 64 slabs L=1 breaks (diff 0.73) and the true L~6.5 is required (diff 0.0000). Node trees stay sound at L=1. gpu_cull_bench prints this guidance when the images diverge.

## [4.36.1]
- tools/gpu_cull_bench.cpp (built under FREP_BUILD_GPU): renders a scene on the Vulkan path with the tile cull off and then on, reports median ms/frame and the speedup, and writes <scene>_cull_off.ppm / _cull_on.ppm plus the max pixel difference as a correctness signal. Args: scene.json [res] [slabs] [lipschitz] [frames].

## [4.36.0]
- GPU tile cull wired into the emitted render shader (TracerConfig::cull_slabs, cull_lipschitz; 0 slabs = off, unchanged output). Each 8x8 workgroup bounds its own frustum depth slabs with the Lipschitz box rule f(box) subset [f(c)-L*r, f(c)+L*r] using the scalar scene_sdf, publishes the surviving depth span in shared memory, and the whole workgroup marches only that span; a fully culled workgroup falls through to the existing miss path. No second pipeline, buffer, or host-side barrier.
- The barrier is emitted before the bounds-check early-returns, so it sits in uniform control flow (a barrier reached by only part of a workgroup is undefined and hangs some drivers).
- tools/bench_tile_cull.cpp gained a lipschitz mode that runs the shader's exact rule on the CPU, so the cull rate can be measured without a GPU.
- Measured (256x256, 8px workgroups, 32 slabs): skipped (tile x slab) volume 95.8% s1_sphere, 93.1% s2_csg, 96.7% s3_blend, 97.4% s5_twist at L=1, with zero wrongly culled tiles. The gyroid implicit is not 1-Lipschitz (|grad f| ~ 6): at L=1 it wrongly culls 8 tiles and puts 33 hits outside the kept span; at L=6.5 it is sound again but only skips 56%. For such raw implicits the interval shader in glsl_interval.hpp is the better cull (87% skipped, sound), while node trees are better served by the in-shader Lipschitz pass.
- GPU wall-clock still unmeasured here: shaders are generated and validated to SPIR-V (glslangValidator), including a node-tree CSG scene, but no CUDA/Vulkan device is available in this environment.

## [4.35.0]
- core/gpu/glsl_interval.hpp: GLSL interval arithmetic (the same rules as the LLVM interval twin, DAG-shared via vec2 temporaries) plus a generated tile-cull compute shader. Per screen tile it walks depth slabs and writes the active span vec2(t0,t1); a fully rejected tile gets an empty span and the render kernel skips it. A single slab per tile bounds the whole view ray and never culls, so the depth subdivision is what makes the pass work.
- tools/bench_tile_cull.cpp: emits the shader (--dump) and runs the identical slab logic on the CPU against the JIT'd interval SDF, reporting skipped volume, empty tiles, kept span, and a correctness check against a scalar march.
- Measured (256x256, 8px tiles, 32 slabs, 1 core): rejected (tile x slab) volume 96.5% s1_sphere, 95.5% s2_csg, 97.1% s3_blend, 87.0% s4_gyroid, 91.5% s5_twist; mean kept depth span 0.09-0.17 of [near,far]; zero wrongly culled tiles and zero hits outside the kept span on every scene.
- All five shaders compile to SPIR-V (glslangValidator). GPU wall-clock numbers still need a CUDA/Vulkan host: the shader is generated and validated here, not dispatched.

## [4.34.1]
- Interval tan: detect a pole (pi/2 + k*pi) inside the argument range and widen to the full range instead of returning the endpoint values, which were wrong across a pole.
- Interval atan2: replaced the blanket [-pi,pi] for x<=0 with the box-corner extremes, widening to the full circle only when the box actually contains the origin or the negative-x branch cut. Mean bound width on random boxes drops from 2*pi to ~1.3.
- tests/test_interval_enclosure.cpp: randomized enclosure check (random boxes, dense sampling) for sin/cos/tan/atan2/asin and x*x-y*y. All sound.

## [4.34.0]
- core/render/cpu_trace.hpp: host-side CPU sphere tracer with three march strategies — scalar, SIMD-8 packet (over the general-tree scene_sdf_simd of 4.33.0), and SIMD-8 with a per-region step taken from a coarse Lipschitz grid instead of the global safety_factor. LipGrid::worth_using() reports whether a scene's field actually needs it.
- tools/bench_trace.cpp: measures all three (wall time, SDF evaluations, hit parity).
- Measured, 192x192, 1 core: SIMD packet is 2.6x (CSG), 2.4x (blend), 3.6x (gyroid) over scalar with identical hits. The Lipschitz step is a correctness feature, not a speed one: on a true SDF (max_L~1.15) it only adds lookup cost, while on the gyroid implicit (max_L~6.0) the global 0.85 step overshoots and loses 498 of 19092 reference hits, which the per-region step recovers all but 59 of, at ~10x less time than marching the whole image at the conservative step.

## [4.33.0]
- General-tree SIMD: CgCtx gains `width`; fc()/param_value() splat in vector mode, so every FRepNode's existing scalar codegen emits <W x float> unchanged. compile_scene_sdf_simd() now vectorizes arbitrary node trees (union of visible objects), not just single-CustomExpr scenes. CustomExprNode inlines its vector AST (gen_vec_inline) instead of calling the scalar helper. Verified bit-exact against the scalar path for Sphere/Box/CSG/SmoothUnion/Translate/Scale/RotateY/TwistY/Negate; CustomExpr differs only by the poly-approx error (~5e-7).
- Lipschitz octree pruning (octree_leaves_lipschitz): prunes any node tree using only the scalar SDF and the L=1 SDF invariant, with an optional L for non-distance implicits plus estimate_lipschitz() (heuristic). Interval arithmetic stays the tighter option for CustomExpr scenes.

## [4.32.9]
- Interval trig (sin/cos/tan/asin/acos/atan/atan2) + x*x square rule; octree children overlap by one grid index so no sign-crossing cell falls in the inter-sample gap (soundness verified: 0 uncovered surface cells on all canonical scenes). Pruned grid eval 2-8x over per-cell SIMD, 2-6x over libfive.

## [4.32.8]
- Interval arithmetic + octree pruning: compile_interval() emits an {lo,hi} twin of a CustomExpr scene (add/sub/mul/div/neg/abs/sqrt/min/max/pow); compile_scene_sdf_interval() + octree_leaves()/octree_classify() drop fully inside/outside regions so only surface cells are evaluated. Trig interval rules pending. Grid eval on the pruned leaf set is ~7-29x over per-cell SIMD on the canonical scenes.

## [4.32.7]
- SIMD poly: vectorizable inline polynomial approximations for sin/cos/tan/asin/acos/atan/atan2/mod in the vector path (replaces llvm.sin.vNf32, which scalarized, and the per-lane libm fallback). Range-reduced minimax, f32 err ~6e-7 (sincos) / ~1e-5 (atan/asin). No libm dependency in the SIMD leaf; lowers to real SIMD on any target.

## [4.32.6]
- CustomExpr SIMD: compile_vec() emits a W-lane (<8 x float>) vector twin of the scalar fn from the same DAG; vector intrinsics for sqrt/sin/cos/min/max/etc., per-lane scalar libm for asin/acos/atan/atan2/mod. compile_scene_sdf_simd() JITs scene_sdf_simd for single-CustomExprNode scenes.

## [4.32.5]
- CustomExpr: let-bindings ("name = expr; ... result") with shared-subtree (DAG) evaluation; fold and LLVM codegen memoize by node identity, so structured imports keep common subexpressions (fewer JIT ops).

## [4.32.4]
- CustomExpr: add atan, asin, acos, atan2, mod to all expression back-ends (LLVM lowers them to libm calls — CPU paths only; the NVPTX path keeps its self-contained-transcendentals constraint).
- core/compiler/compile_sdf.hpp: JIT the whole-scene SDF to a plain function pointer (used by frep-designer-bench).
- CMake: guard perf-check targets when frep_bench is absent; per-target AUTOMOC for repro_cpu_ir_black.
- benchmarks/ extracted to the separate frep-designer-bench project.

## [4.32.3]
- Add benchmarks/: cross-system performance suite (frep4, HyperFun, libfive, MPR) — canonical + imported scenes, grid-eval and render metrics, RAPL/nvidia-smi energy (kWh/Mpix).

## [4.32.2] — 2026-06-27

### Fixed — horizon "holes" at object silhouettes (raised default max_steps)

`TracerConfig.max_steps` 192 → 512. Reproduced from the user's own scene
(`horizon.json`, camera grazing the floor) rendered through the CPU-IR path and
compared against a hole-free max_steps=4096 reference: at 192 there were ~540
hole pixels — a bright line all along the horizon (the floor seen at grazing
angle) plus the silhouettes of the objects against it, exactly the reported
artifact. Those are grazing rays that exhaust the step cap and escape to the sky
instead of converging. 384 cleared every one of them; 512 is the shipped default
for margin. The cost falls only on grazing rays (interior rays still converge in
a handful of steps), and all four paths read `cfg.max_steps` (codegen for cpu_ir
/ gpu_ir, the GLSL and RTX emitters), so the fix is uniform. Parity unchanged
(cpu_ir vs gpu_glsl 17/17). This is the empirical replacement for the
over-relaxation attempt below, which over-stepped the same grazing floor and made
the holes worse.

### Reverted — over-relaxation (enhanced sphere tracing) disabled by default

An over-relaxation march (Keinert over-stepping with an overshoot guard) was
added but is now **disabled**: `TracerConfig.over_relax = 1.0` = classic sphere
tracing. On real floor-heavy scenes it made the horizon **worse**, not better:
omega>1 over-steps the ground plane seen at grazing angle (the horizon itself),
the overshoot guard backtracks, and the wasted iterations make grazing horizon
rays exhaust `max_steps` *more* — widening the "holes" exactly where the horizon
meets object silhouettes. The standalone step-count sim (sphere over floor,
near-head-on) had shown a net win, but it never modelled the grazing floor, which
is the dominant case at the horizon. The mechanism stays in the code as a knob
(all four paths still consult `over_relax`) but ships at 1.0; the horizon will be
addressed separately with max_steps / the grazing-rescue, no over-stepping.
Parity with over_relax=1.0 is preserved: cpu_ir vs gpu_glsl 17/17 within
tolerance.

### Changed — multi-view viewport preserves the previous frame

`ExecutorViewport::render_worker` now seeds each pass's frame from the last
completed frame (guarded by matching resolution / layout / path set) instead of
clearing to the dark fill colour. A strip whose path doesn't finish the current
generation — e.g. a slow CPU path pre-empted by rapid camera moves — keeps its
previous content rather than flashing through dark grey. (GUI code; now builds
against Qt6 in the headless `offscreen` configuration.)

### Fixed — gpu_ir strip freezes permanently after a camera move (multi-view)

Root cause: a **CUDA context thread-affinity** bug. `ExecutorViewport::start_render`
spawns a fresh `std::thread` for every frame (joining the prior one), so each
render generation runs on a different OS thread. `CudaCtx::create` calls
`cuCtxCreate`, which makes the new context current **only on the creating
thread** (the first frame's worker thread), and `CudaCtx::render` assumed the
context was already current. On the next camera move the worker is a new thread
where that context is not current, so `cuLaunchKernel` / `cuCtxSynchronize` fail
("invalid context"), `render()` returns an error, the strip isn't blitted, and
gpu_ir stays frozen on its last good frame — permanently, since every subsequent
worker thread has the same problem. (A material edit recompiles and recreates the
context on the current thread, which is why it would unfreeze for exactly one
frame.) gpu_glsl is unaffected because Vulkan contexts aren't thread-bound this
way.

Fix: `CudaCtx::render` now calls `cuCtxSetCurrent(context)` at entry, making the
context current on whichever thread runs the render. Verified the edit compiles
(`-fsyntax-only -DFREP_HAS_CUDA`); it can't be exercised in the sandbox (no CUDA
device), so confirm on a CUDA GPU. The diagnosis was reached without CUDA by
proving the gpu_ir IR is camera-invariant (so no per-frame recompile) and that
the camera is uploaded fresh in the kernel args every render — leaving the
context-current step as the only gap. gpu_rtx (also CUDA) likely needs the same
guard.

### Changed — multi-view renders the per-path strips in parallel

The worker now renders each path's strip on its own thread instead of
sequentially in path order, so a fast path (gpu_glsl) no longer waits behind a
slow one ahead of it (cpu_ir) — whichever finishes first publishes first, which
is what the user saw lag (the quicker GPU strip appearing after the slower CPU
strip). Each path owns disjoint image regions and disjoint slots in the
accounting vectors, so the threads don't synchronise on those; only the shared
frame and the image_ publish are serialised (image_mu_). Executors are
one-per-path (no sharing) and the scene is read-only during render, so the
parallel renders don't contend. Verified with the headless cpu_ir + gpu_glsl
harness: correct output (no races or corruption) and both strips complete
concurrently (≈286 ms each in the sandbox), where sequentially the second strip
could only appear after the first finished (≈2×).

### Note — analytic normals are already universal

Confirmed that **all** render paths already compute the surface normal
analytically (cpu_ir / gpu_ir via forward-mode AD in `emit_scene_normal`;
gpu_glsl via dual-number arithmetic matching it; gpu_rtx via the same
`scene_normal()`). The "finite-diff, noisy near the surface" comment in
`codegen.cpp` describes the approach being *avoided*, not used. No change needed —
the F-Rep analytic-gradient normal is already the one in effect everywhere, which
is why CPU/GPU profiles match to ±1/255. (Central differences remain only for
*texture* bump normals, not geometry.)

---

## [4.32.1] — 2026-06-25

### Fixed — blank GPU renders from the test camera (`build_push_simple`)

`build_push_simple` (the explicit camera+light helper used by the GPU render
tests via `sane_push`/`make_push`) set the light *position* but never the light
*colour*, so `light_colors[0]` stayed zero-initialised. The emitted GLSL shader
multiplies every light's contribution by `pc.light_colors[i].xyz`, so a zero
colour makes all shading black — the rendered frame is blank regardless of the
device (reproduced on both Mesa lavapipe and a discrete GPU). `build_push_from_scene`
already seeds a white default; this variant now matches it.

This is a pre-existing bug (present since coloured lights were added), unrelated
to the 4.32.0 unified-placement change, surfaced only when a Vulkan device is
available so the render tests actually run. It fixes the six emitter-based GPU
tests (`GlslEmitter.Renders*OnGpu`, `GpuPatterns.CheckerRenderProducesTwoColours`,
`GpuMeshSdf.RendersOnGpu`, `GpuTextures.RendersOnGpu`). Three render tests that
use the legacy hand-written `sphere_trace.comp` (`Vulkan.RendersNonBlankImage`,
`Vulkan.SphereAlbedoVisible`) or a shadow-artifact check (`GpuShadowRegression.NoHorizonHoles`)
are a separate pre-existing matter and are not addressed here.

---

## [4.32.0] — 2026-06-25

### Added — unified parameter placement: opt-in incremental runtime params across backends

Brought the GPU backends toward the CPU_IR reference incremental model. A new
backend-agnostic placement layer decides, per parameter, whether a value is a
baked **Constant** (folded into the generated code) or a **Runtime** slot read
from a shared parameter buffer — a continuum between all-constant and
all-variable, driven by a policy/statistic, not a fixed three tiers.

New (LLVM-free, unit-tested via `tests/standalone/test_placement_poc.cpp`, builds
with plain g++):
  - `core/compiler/param_binding_table.hpp` — the single authority for slot
    assignment. One deterministic scene walk + a CompilePolicy yield
    `slot_of(node,param)`, a seed buffer, and a `placement_hash()`. Every backend
    reads the same slot for the same parameter, so the runtime buffer layout is
    identical across paths.
  - `core/compiler/compile_policy.hpp` — added `AllRuntimePolicy`,
    `ParamEditStats` (per-parameter edit heat with decay), and `PerParamPolicy`
    (promotes frequently-edited parameters to Runtime, demotes idle ones).
  - `core/compiler/scene_bindings.hpp` — FRepNode → table adapter.
  - `tools/validate_runtime_params.cpp` (`frep_validate_runtime_params`) — checks
    that a runtime-parameter edit leaves the GPU-GLSL shader byte-identical and
    that the buffer-path image matches a full recompile.

Backend wiring (**opt-in**; with no policy installed the output is bit-identical
to 4.31.65 — the binding table is empty and no runtime buffer is emitted):
  - **GPU_GLSL** — `glsl_emitter` routes every geometry/deform parameter through
    a `pval()` choke point in both the SDF and the dual-AD normal path; Runtime
    params become `P.v[slot]` reads from a new std430 buffer at binding 3,
    Constants still fold at shader-compile time. `vulkan_ctx` gained the binding-3
    SSBO and `update_params()` (a memcpy into a persistently-mapped buffer);
    `gpu_executor` pushes the current values each frame. A Runtime-param edit is
    now a buffer re-upload — no re-emit, no SPIR-V recompile, no pipeline rebuild.
  - **GPU_IR** — `gpu_ir_executor` enables incremental params and feeds the
    kernel's existing `float* params`; the cached PTX is invariant to
    Runtime-value edits.

Enable via `set_compile_policy(&policy)` on the executor (e.g.
`ByParamClassPolicy::interactive()` or a `PerParamPolicy`). See
`docs/UNIFIED_PARAM_PLACEMENT.md`.

Not yet done — GPU_RTX (same choke point + a binding-3 SSBO in the
intersection/closest-hit shaders; note the acceleration structure still rebuilds
when a Runtime *geometry* edit moves an AABB, inherent to RT broad-phase), and
pointing the CPU codegen's slot allocator at the shared table so all four buffers
are byte-identical (today each backend's layout is self-consistent). Hardware
validation pending for the GLSL SPIR-V/Vulkan and CUDA PTX paths and for
cross-path parity under an interactive policy; the placement layer is unit-tested.

---


## [4.31.65] — 2026-06-11

### Changed — calibrated cubic smin blend width (was over-inflated)

The cubic smin from 4.31.64 fixed the edges but blended much wider/puffier than
the old quadratic at the same `k` (its internal scale was IQ's 6, giving ~16×
the influence width and ~8× the bulge depth). Recalibrated the internal scale to
2.0, which keeps the blend smooth (curvature jump ~0.03 at k=0.4, no edges) while
cutting the bulge ~3× (centre-bulge height 0.39 vs 0.75 at scale 6) — so `k`
behaves much closer to its old meaning.

  kk = k*2;  h = max(kk − |a−b|, 0)/kk;  smin = min(a,b) − h³·kk/6

Applied across all four paths (CPU eval, CPU IR, forward-AD, GLSL scalar + dual)
for parity. Note: as with any smin, smoothness scales with k — very small k still
gives a tighter, sharper transition (expected: small k ≈ "almost no blend").
330 tests pass.

---


## [4.31.64] — 2026-06-11

### Fixed — visible "edges" at smooth-blend boundaries (quadratic smin → cubic)

The smooth union (smin) showed faint creases right where the blend meets a clean
primitive — most visible on the two-sphere yellow blob. Root cause: the quadratic
polynomial smin clamps its interpolant `h` to [0,1], which makes the function
only C1-continuous; its gradient (the surface normal, computed analytically via
forward-mode AD) has a kink exactly at the blend boundary where h hits 0 or 1.
Numerically the curvature jumps ~0.37 there — a visible normal discontinuity.

Replaced it with Inigo Quilez's **cubic polynomial smin**, which is
C2-continuous, so the normal varies smoothly across the boundary. Measured
curvature jump drops from ~0.37 to ~0.013 (~28× smoother); the kink — and the
visible edges — are gone. Chosen over exponential smin because: (a) it removes
the edges just as well, (b) it actually improves distance-field fidelity
(max |∇|−1 of 0.22 vs the quadratic's 0.63), and (c) it's pure polynomial with no
transcendentals — so it JITs identically on every path including NVPTX, where
exp2/log2 wouldn't resolve at module-load time (the same constraint that forced
the inlined sin/cos minimax earlier).

Updated in all four code paths for cross-path parity: CPU eval, CPU IR codegen,
forward-AD gradient (for normals), and the GLSL emitter (both the scalar SDF and
the dual-number gradient `d_smin`). Formula:
  kk = k*6;  h = max(kk − |a−b|, 0)/kk;  smin = min(a,b) − h³·kk/6
330 tests pass, including cross-path parity and the SmoothUnion Lipschitz
invariant. Visual confirmation on GPU hardware recommended.

---


## [4.31.63] — 2026-06-11

### Fixed — real crash on layout change / removing lan (executor freed under render thread)

The earlier fix wasn't the whole story. Driving the exact GUI sequence under
AddressSanitizer (select cpu_ir+lan, change layout / remove lan — including
*while the retry timer is pending and no worker is connected yet*) pinned a SEGV
on the render worker thread:

- `ExecutorViewport::set_remote_config()` did `executors_.clear()` **without
  stopping the render worker**. apply_multipath_selection calls it whenever the
  lan path is active, so changing the layout (or any re-apply) could free the
  RemoteExecutor while the worker thread was inside `ex->render()` on it →
  use-after-free. Now it cancels + joins the worker before clearing, like the
  other mutators.
- `install_backend` tore the old viewport down with `deleteLater` (async). With
  the lan path's pending retry `QTimer`, the timer could fire `start_render()` on
  a half-torn-down ExecutorViewport before deletion ran. Now the viewport is
  deleted **synchronously**, so `~ExecutorViewport` (cancel + join worker, drop
  posted events) runs atomically; the executor also owns its canvas explicitly.
- `set_remote_config` is no longer called on every re-apply — only when the
  endpoint actually changes — so an unrelated re-apply no longer rebuilds the
  master (re-bind + re-accept) or drops a connected worker.

Verified under AddressSanitizer: selecting cpu_ir+lan then changing layout and
removing lan — both with no worker (retry pending) and with a worker connected —
is clean (no SEGV/UAF). 330 tests pass.

### Changed — aggregate throughput banner no longer overlaps per-path labels

With vertical strips, a per-path label chip (top-left of each strip) could sit
near the horizontal center where the "Σ Mpix/s" banner was drawn. The aggregate
banner is now offset one row below the per-path chips so they never overlap.

---


## [4.31.62] — 2026-06-11

### Fixed — lan strip not refreshing on worker connect; crash on layout/path change

Two issues with cpu_ir + lan composited rendering:

**lan strip stayed black until the camera moved.** The frame renders once when
paths are set; at that point the lan path has no worker yet, so its strip is
skipped. Nothing re-rendered when a worker connected later — only a camera/scene
change did. Now the render worker flags a retry whenever a path reports itself
unavailable (`available() == false`), and the viewport re-renders ~400 ms later,
repeating until the path comes up. So the lan strip fills in on its own once a
worker connects — no camera move needed.

**Crash when changing the Multi-view layout (split→tiles) or removing lan while
a worker was connected.** `set_active_paths` and `set_layout` mutated the shared
`paths_` / `layout_` directly and only then called `invalidate()`. If the render
worker was mid-`render()` — e.g. a RemoteExecutor blocked on a TCP round-trip —
mutating that state under it was a use-after-free. Both setters now cancel and
join the render worker *before* touching shared state, then restart the render.

Verified in-sandbox: with cpu_ir + lan and a late-connecting worker, the lan
strip auto-fills (bottom half lit) with no invalidate; switching split→tiles and
removing lan mid-session (worker connected) complete without a crash. 330 tests
pass.

---


## [4.31.61] — 2026-06-11

### Fixed — lan path crash/hang on path switching; clarified lan usage

Two problems with the new lan path:

**Crash / hang when switching path combinations.** `RemoteExecutor` opened its
master with a *blocking, uninterruptible* `accept()` on the viewport's render
thread (the master's cancel token was null). While it sat waiting for a worker,
switching paths made the ExecutorViewport try to join that render thread — which
was stuck in accept() — hanging the GUI, or crashing if the executor was
destroyed underneath the blocked call.

Fix: `RemoteExecutor` now opens the master on its own background thread with a
cancel token, so the render thread never blocks. `available()` reports false
until the master is up with a worker connected (the viewport skips the lan strip
until then), and `render()` returns a non-ok tile rather than blocking if called
early. The destructor sets the cancel flag and joins the open thread, so
teardown is immediate even while still waiting for workers. Verified under
AddressSanitizer: create+destroy with no worker returns in ~0 ms (was an
indefinite hang), and rapidly switching across seven path combinations
(including lan alone, lan+cpu_ir, lan+gpu_ir, lan+cpu_ir+gpu_rtx) is clean.

**Confusing lan usage.** It wasn't clear how to drive the lan path. Clarified:
- Selecting the **lan path** auto-starts a master on the LAN tab's port (alone,
  or composited with local paths). The LAN tab now says so, disables its
  standalone "Start master" button while lan is active (so the two can't fight
  over the port), and prompts you to start worker(s).
- The LAN tab's **Start render** button remains the separate standalone one-shot
  mode. The tab's note now explains both ways explicitly.

330 tests pass; the RemoteExecutor test waits for availability before rendering.

---


## [4.31.60] — 2026-06-11

### Changed — unified "lan" as a RemoteExecutor (IExecutor), not a special backend

Following the architecture review, the "lan" path is no longer a bespoke backend
with its own interactive viewport loop. It is now an ordinary `IExecutor` —
`RemoteExecutor` — that forwards each tile to a worker over TCP and returns the
pixels exactly like cpu_ir / gpu_glsl / etc. The compositing ExecutorViewport
drives it like any other path, so:

- **lan composites** alongside local paths (e.g. cpu_ir + lan in a split layout —
  half the frame local, half remote), directly demonstrating heterogeneous
  aggregate throughput across the network. Previously lan owned the whole frame.
- **scene-change re-render is free** — the viewport already re-invokes render()
  on every change, so moves / adds / deletes / material edits re-render with no
  special loop.

New / changed:
- `RemoteExecutor : IExecutor` (`core/exec/remote_executor.hpp`) — owns a
  `PersistentMaster` for one endpoint; `render(scene,W,H,tile)` re-serializes the
  scene, sends it only when its hash changes (one Scene push per frame, not per
  tile — no delta optimization, per the PoC scope), forwards the tile, returns
  the result. `available()` is true once the master is open with ≥1 worker.
- `PersistentMaster::render_tile()` — a per-tile primitive the executor drives
  (the viewport already does tile decomposition + scheduling, so the master just
  serves one tile per call). `worker_count()` exposes connectivity.
- `ExecutorViewport::set_remote_config(port, n_workers)` — builds the lan path's
  RemoteExecutor against the LAN tab's endpoint. lan now flows through the normal
  Executor backend; `Backend::Lan`, the interactive DistViewport loop, and the
  `start/stop_lan_interactive` special cases are gone.

Removed the bespoke interactive path and its now-dead DistViewport interactive
loop; the one-shot LAN tab (master/worker buttons, progressive display, worker
tile-list) is unchanged. Multi-master is documented in DISTRIBUTED.md and
remote_executor.hpp for future work.

**Documented for future work (in remote_executor.hpp): multi-master.** Because a
RemoteExecutor *is* an executor wrapping one master endpoint, multiple masters
fall out for free — `{cpu_ir, remote@:5900, remote@:5901}` is local CPU plus two
independent LAN clusters, all composited by the existing viewport, with the
weighted layout auto-balancing by measured throughput. Explicitly out of scope
for the PoC (each a substantial design): when/how to distribute the scene
(on-demand vs. ahead-of-time vs. hybrid) and incremental scene updates; dynamic
resource-aware job distribution; fault tolerance when a worker or master vanishes
mid-frame; back-pressure / per-round-trip batching / latency hiding.

Verified in-sandbox over real TCP: RemoteExecutor renders tiles as a plain
IExecutor across scene changes with a persistent worker; PersistentMaster renders
multiple frames keeping the worker connected. 330 tests pass (2 new); GUI builds
+ starts clean.

---


## [4.31.59] — 2026-06-11

### Added — "lan" as a selectable path (interactive distributed render)

A fifth path, **lan**, now appears in the path picker (and Render menu) alongside
cpu_ir / gpu_glsl / gpu_ir / gpu_rtx. Unlike the others it has no local
retarget target; selecting it starts the LAN master (if not already running) and
renders the frame across connected workers, streaming tiles into the viewport
progressively — then keeps the workers connected and **re-renders on every scene
change** (move, add, delete, material/visibility edits, …). The whole scene is
re-sent each frame (no delta optimization yet, by design).

**Protocol (core/dist):**
- New `MsgType::EndFrame` — master → worker: this frame's tiles are drained, but
  stay connected; a new Scene for the next frame is coming.
- `Worker::run` restructured into a frame loop: it now receives a Scene at the
  start of every frame (rebuilding the executor each time), serves tiles, and on
  EndFrame loops back for the next Scene; NoMoreWork or a closed socket ends the
  session. One-shot masters still work (they send one Scene then NoMoreWork).
- New `PersistentMaster`: `open()` binds + accepts workers once; `render_frame()`
  sends the scene, distributes tiles via a fresh scheduler, collects results
  (firing the progressive on_tile), and sends EndFrame so workers persist;
  `close()` ends the session. Cancellable like the one-shot master.

**GUI:**
- `PathKind::Lan` + `MainWindow::picker_paths()` (local paths + lan); the picker
  and Render menu now build from it. `desired_backend_for` maps any selection
  containing lan to a new `Backend::Lan`.
- `DistViewport` gains an interactive mode: `invalidate()` snapshots the scene
  and requests a frame; an `interactive_loop()` opens a PersistentMaster once and
  re-renders the latest scene whenever a frame is requested. Scene changes already
  funnel through `on_scene_changed → invalidate()`, so edits re-render live.
- `start_lan_interactive` / `stop_lan_interactive` install/tear down the
  persistent DistViewport using the same synchronous, ASan-clean teardown as the
  one-shot LAN path.

Verified in-sandbox over real TCP: a PersistentMaster + one worker render three
frames of changing scenes while the worker stays connected (27 tiles across 3
frames); the interactive DistViewport re-renders on each invalidate (3 frames,
worker persistent). 328 tests pass; GUI builds + starts clean.

---


## [4.31.58] — 2026-06-10

### Fixed — crash on starting a second LAN worker

Starting a worker a second time (master → worker → master → worker) crashed.
Root cause: `start_lan_worker` did `lan_worker_thread_ = std::thread(...)`, but
the previous worker's `std::thread` object was still joinable — the first run had
finished, yet the thread was never joined or detached. Assigning to a joinable
`std::thread` calls `std::terminate()`.

Fix: reap the previous worker thread before launching a new one — `if
(lan_worker_thread_.joinable()) lan_worker_thread_.join();` just before the
assignment. The guard at the top of `start_lan_worker` (`if (lan_worker_running_)
return;`) guarantees the previous run has already completed, so the join returns
immediately.

Verified under AddressSanitizer with three full master+worker render cycles
back to back (master→worker repeated): no std::terminate, no use-after-free, all
renders complete. 328 tests pass.

---


## [4.31.57] — 2026-06-10

### Fixed — LAN master crash on Stop / second Start + a latent node-graph UAF

Two reported crashes (Stop right after Start; second Start then Worker), plus a
latent use-after-free that AddressSanitizer surfaced once the LAN flows ran.

Driving the real MainWindow start/stop/restart/worker sequence under ASan (via a
temporary env-gated self-test, since removed) showed the LAN teardown itself was
clean after the lifecycle hardening below, but exposed a separate crash at
shutdown in the node-graph.

**LAN master lifecycle (the two reported crashes):**
- Interruptible `accept` (from 4.31.x) + `MasterConfig::cancel` let a master
  waiting for workers be stopped instead of blocking `join()` forever.
- The DistViewport is now stopped + deleted **synchronously** (not
  `deleteLater`) in both `start_lan_master` (restart path) and
  `stop_lan_master`; a synchronous `delete` runs `~QObject`, atomically dropping
  any posted repaint / `render_completed` events that would otherwise fire on a
  half-destroyed object. Before deleting, the status poll `QTimer` is stopped and
  the `render_completed` connection is `disconnect`ed.
- The DistViewport no longer takes a QObject parent and owns its canvas solely
  (reparented to null before delete) — removing a double-delete window.
- `on_tile` / completion callbacks are guarded by the cancel flag so nothing is
  marshalled into a dying viewport.

**Latent node-graph use-after-free (surfaced at shutdown):**
- `~EdgeItem` dereferences its input `PortItem` (`to_->edge`), which lives inside
  a `NodeItem`. Qt's automatic `~QGraphicsScene` deletes items in an unspecified
  order, so a NodeItem could be freed before its EdgeItem → UAF. Added a
  `~NodeGraphScene` that deletes all edges first, while their endpoint ports are
  still alive. This was always a latent hazard; the LAN backend swaps made the
  ordering manifest.

Also fixed the stale window title ("F-Rep Designer 4.0" → "F-Rep Designer").

Verified: the full start→stop→restart→worker→render→stop sequence plus app
shutdown runs clean under AddressSanitizer (no UAF, no SEGV). 328 tests pass;
the node-graph logic test passes.

---


## [4.31.56] — 2026-06-10

### Fixed — LAN master crash on Stop and on second Start (use-after-free)

Stopping the master, or starting it a second time, crashed. AddressSanitizer
pinned it to `~DistViewport` running from Qt's `sendPostedEvents`: the teardown
deleted the DistViewport with `deleteLater()`, but the master thread had already
posted `invokeMethod` repaint / `render_completed` events targeting it. Those
queued events fired on a half-destroyed object → SEGV. A second teardown path
(canvas deleted both by `deleteLater` as the old widget and again by the
destructor) compounded it.

Fix — deterministic teardown:
- The DistViewport is now **stopped then deleted synchronously** (not
  `deleteLater`) in both `start_lan_master` (when replacing an existing master)
  and `stop_lan_master`. A synchronous `delete` runs `~QObject`, which removes
  any posted events targeting the object atomically, so nothing fires on freed
  memory. `stop()` joins the master thread first, so no new events are posted
  after deletion begins.
- `DistViewport` now owns its canvas explicitly: the destructor reparents it to
  null and deletes it once, removing the double-delete against Qt's parent
  ownership.
- Normal (non-LAN) path viewports keep the existing async `deleteLater`
  teardown — only the DistViewport needs the synchronous path.

Verified under AddressSanitizer with a real Qt event loop: start→stop (no
worker), start→start (second master), and start→stop→start→worker→render→stop
all complete with no SEGV and a correct 12-tile render. 328 tests pass.

---


## [4.31.55] — 2026-06-10

### Changed — worker debug preview is now a chronological tile list

The worker preview (debug) showed only the single most-recent tile. It's now a
scrollable list of every tile the worker has rendered this run, in order, each a
64×64 thumbnail labelled with its frame-space position (e.g. "(128,64)"), newest
scrolled into view. The list is cleared on a new worker run and whenever the
checkbox is toggled, so it always reflects the current run. Still off by default
and purely diagnostic — no throughput implication.

Implementation: the single preview QLabel became a `QListWidget` in IconMode;
the worker's `on_tile` hook appends a `QListWidgetItem` per tile (marshalled to
the GUI thread) instead of replacing one pixmap.

Verified in-sandbox: the list accumulates one item per tile chronologically
(first/last labels correct) and clears to empty on both a new run and a toggle.
GUI builds + starts clean; 328 tests pass.

---


## [4.31.54] — 2026-06-10

### Fixed — LAN master hang on Stop + crash on second Start

Both bugs had the same root cause: `Master::run()` blocks in `accept()` waiting
for workers, and nothing could interrupt it.

- **Stop froze the GUI**: pressing Stop (or starting again) tore down the
  DistViewport, whose destructor joined the master thread — but that thread was
  stuck in `accept()` for a worker that never came, so `join()` blocked forever
  on the GUI thread.
- **Second Start crashed**: the old DistViewport was deleted with `deleteLater`
  (asynchronous), so its still-blocked master thread outlived the start of a new
  master on the same port — two masters racing, then a use-after-free as the old
  object was destroyed underneath its running thread.

Fixes:
- **Interruptible accept** — `TcpListener::accept(const std::atomic<bool>*
  cancel)` now polls with a 200 ms tick and returns early when the cancel flag
  is set, instead of blocking indefinitely. `MasterConfig::cancel` threads the
  flag through `Master::run()` (both the accept loop and the serving loop check
  it). Default null → CLI behavior unchanged.
- **DistViewport** owns a `cancel_` flag set by `stop_render()` before the join,
  so tearing it down interrupts a master waiting for workers; a public `stop()`
  lets the LAN tab join synchronously.
- **LAN tab** deletes the old DistViewport synchronously (not `deleteLater`)
  before binding a new master, and calls `stop()` before teardown — so the
  master thread is always joined before the port is reused. `SO_REUSEADDR` (already
  set) lets the immediate rebind succeed.

Verified in-sandbox: with no worker connected, `stop()` returns in ~0 ms (was an
indefinite hang); start → stop → start on the same port succeeds without a
crash. 328 tests pass; 10 dist tests still green.

---


## [4.31.53] — 2026-06-10

### Fixed — LAN viewport colors were BGR (red/blue swapped)

The DistViewport (master display) and the worker debug preview built their
QImage with `Format_RGBA8888` but filled it with `qRgba(r,g,b,a)`. `qRgba`
returns a 0xAARRGGBB integer; written through a `QRgb*` on a little-endian host
the bytes land as B,G,R,A — but `Format_RGBA8888` interprets memory as R,G,B,A,
so red and blue came out swapped. Switched both to `Format_ARGB32`, which packs
as 0xAARRGGBB and therefore matches `qRgba` (the same convention the local
ExecutorViewport uses with `qRgb` + `Format_RGB32`).

Verified with a direct format probe: `Format_RGBA8888`+`qRgba` reads back a
(200,50,20) red as (20,50,200) — swapped — while `Format_ARGB32`+`qRgba` reads
it back as (200,50,20), identical to the ExecutorViewport path. GUI builds +
starts clean; 328 tests pass (unchanged — this is GUI-only).

---


## [4.31.52] — 2026-06-10

### Added — optional worker tile preview (debug)

The LAN worker renders tiles for a master; it has no frame of its own to show.
For diagnostics, the worker group in the LAN tab now has a "Show last rendered
tile (debug)" checkbox — off by default — that displays the worker's most recent
rendered tile beneath its status line. It's purely diagnostic; the worker does
not progressively composite a frame (only the master does), and the preview
carries no cost when the box is unchecked.

- **`WorkerConfig::on_tile`** — a new optional per-tile hook on `dist::Worker`,
  fired (thread-safe, on the worker thread) after each tile is rendered and
  before it's sent. Empty by default; the CLI worker is unchanged.
- **LAN tab** — the worker group gains the checkbox + a small preview QLabel
  (hidden until checked). When on, `start_lan_worker` attaches the hook, which
  marshals each rendered tile into the label as a scaled QImage.

Verified in-sandbox over real TCP: a worker with the hook set renders 12 tiles
and the hook fires exactly 12 times with correct per-tile pixel counts. GUI
builds + starts clean; 328 tests pass.

---


## [4.31.51] — 2026-06-10

### Added — LAN distributed render in the GUI (progressive display)

Distributed render was CLI-only (`frep_dist_render`); now it's in the GUI with a
"LAN" tab, and the result paints into the main viewport tile-by-tile as workers
deliver it — the same progressive feel as the local modes.

- **`MasterConfig::on_tile`** — a new optional per-tile progress hook on the
  dist master, fired (thread-safe) as each tile arrives. The CLI path is
  unchanged (the hook defaults empty); the GUI uses it to display progressively.
- **`gui/dist_viewport`** — a new `DistViewport : IViewport` that runs a
  `dist::Master` on a background thread and blits each finished tile into a
  composited image, marshalling repaints onto the GUI thread. Installs as the
  active viewport while a master render is running.
- **LAN tab** (`build_lan_panel`) with two groups:
  - **Master** — port, workers-to-await, tile size, scheduler (pull/push),
    Start render / Stop. Starting installs a DistViewport into the main viewport;
    a status line shows live "N/M tiles | per-worker distribution".
  - **Worker** — master host + port, render-with path (cpu_ir / gpu_glsl /
    gpu_ir / gpu_rtx), Start / Stop. Runs a `dist::Worker` on its own thread.
  The frame size + scene come from the current viewport + scene.

Verified end-to-end in-sandbox over real TCP (127.0.0.1): a DistViewport master
plus a worker thread render a 128×96 frame as 12 tiles; all 12 arrive, the image
fills in progressively, and the status reports "done 12 tiles … w0=12". GUI
builds + starts clean; 328 tests pass (10 dist tests included).

---


## [4.31.50] — 2026-06-10

### Fixed — geometry-parameter undo: confirmed wired, locked with tests

An old comment in `undo_stack.hpp` claimed geometry-parameter edits (sphere
radius, box dimensions, transform values) were "not yet wired through commands"
and passed directly to the scene without undo. Tracing the actual paths shows
that's stale: node-graph parameter edits already reach the scene as one atomic
`SetGeometryCommand` via `sync_graph_to_scene`, and the translate/rotate/scale
gizmos already commit `SetTransformCommand` / `SetRotationCommand` /
`SetScaleCommand`. Undo also re-syncs the graph from the scene
(`on_scene_changed` → `sync_scene_to_graph`), so a reverted edit shows up in the
node-graph view too. The feature worked end-to-end; what was missing was
regression coverage and an accurate comment.

- Added `UndoCommand.SetGeometryRoundTrips` (a primitive-param edit — radius
  change — applied + undone + redone), `SetTransformRoundTrips`, and
  `SetRotationAndScaleRoundTrip` (two-step undo reverts scale then rotation).
- Replaced the stale "not yet wired" comment with an accurate list of the
  covered operations.

328 tests pass.

---


## [4.31.49] — 2026-06-10

### Changed — removed Phase/Milestone scaffolding from code comments

The doc refresh (4.31.47–48) cleared development-era "Phase 1/2/3/4" markers from
`docs/` and `README.md`, but the same historical scaffolding was still in code
comments across ~17 files (mostly the RTX subsystem) and in two tools' on-screen
output. Removed it, keeping the technical content:

- **RTX subsystem** (`rtx_executor`, `rtx_accel`, `rtx_pipeline`, `rtx_shaders`,
  `rtx_csg_groups`, `rtx_caps`, `rtx_ctx`, `bench_scenes`): the "Phase 0 skeleton
  / Phase 1 naive / Phase 2 validate / Phase 3 multi-BLAS" narration is gone;
  comments now describe what the code does (per-CSG-group BLAS broad-phase,
  hardware-confirmed parity) rather than the order it was built in.
- **Tools**: `frep_rtx_probe` no longer prints "(Phase 1a)" / "(Phase 1b)" in its
  section headers; `frep_advanced`'s banner is "F-Rep Designer — plugins + custom
  expressions + SPIR-V" instead of "F-Rep Designer 4.0 — Phase 4 demo";
  `rtx_bench` / `parity_check` header comments de-phased.

Comments and one banner string only; no behavioral change. Builds clean, 325
tests pass, and `frep_rtx_probe`'s output is verified phase-free.

---


## [4.31.48] — 2026-06-10

### Changed — README + root-doc refresh (de-phased, current state)

The earlier doc audit covered `docs/` but missed `README.md` and the root
release notes, which still described an old "phases" view of the project with
several long-since-finished items marked as not done. Brought them current:

- **Removed all "Phase 1/2/3/4" scaffolding** from README — the "Implemented
  components" sections are now organized by subsystem (core engine, plugins,
  GUI) rather than by development phase, and the architecture-flow / running /
  build sections no longer reference phases.
- **Replaced the stale "Done in 4.0" / "What remains" lists** with a current
  Capabilities section. The four retargeting paths (CPU_IR, GPU_IR, GPU_GLSL,
  GPU_RTX) are described as implemented — previously "Vulkan compute runtime",
  "CUDA backend", etc. were marked ❌ "not done" though all four have long been
  working and hardware-confirmed. "Possible future work" now lists only things
  genuinely not built (per-function incremental, scene-BVH GPU traversal, RTX
  swapchain, FPGA).
- **Intro + Stack table** rewritten to lead with the four-path heterogeneous
  retargeting story, energy (Mpix/kWh), and distributed render.
- **Build instructions** now lead with the tested clang-20 / LLVM 20 toolchain
  (was LLVM 22 primary with 20 as a "fallback"); added the Vulkan/GLSL/Qt deps.
- **Stale figures fixed**: "F-Rep Designer 4.0" title → version-less, "211/212
  tests" → durable phrasing, the octree "future GPU codegen" note clarified
  (GPU codegen exists; the in-kernel octree walk is the future bit).
- **RELEASE_NOTES.md** (a historical 4.0.0 document) got a note pointing readers
  to README/CHANGELOG for the current state, rather than rewriting the release
  record. CHANGELOG phase references are left intact as the historical log.

Docs-only; no source changes. 325 tests still pass.

---


## [4.31.47] — 2026-06-10

### Changed — CLI synchronization + documentation refresh

**CLI parity across the tools:**
- `--energy` added to `frep_multipath` — it was only in `frep_rtx_bench`, but
  the heterogeneous aggregate is exactly where Σ Mpix/kWh is the honest
  cross-device efficiency metric. Brackets the whole render with CPU (RAPL) +
  GPU (NVML) counters via the shared `core/power/energy_meter`; unavailable
  counters print a dash + how-to-enable hint, never a fabricated figure.
- `--help` / `-h` added to `frep_rtx_bench` (had none) and `frep_dist_render`
  (had usage text but no flag to trigger it). All six user-facing CLI tools now
  respond to `--help`.

**Documentation refresh** — brought every doc in line with the current system
and removed historical scaffolding that no longer informs a fresh reader:
- Stripped "Phase 1a/2/3", "Milestone", and similar development-era markers from
  MULTIPATH.md and RTX_VALIDATION.md (results are stated as the finished system;
  the RTX validation log keeps its chronology under descriptive headings).
- DISTRIBUTED.md: the push scheduler is documented as implemented + tested (was
  "planned, not built"); the compiled-artifact note reframed as a possible
  extension rather than a pending task.
- PERFORMANCE.md "remaining opportunities" updated to the real state (shadows
  are already a toggle; scene-BVH GPU upload has the exposed buffer + measured
  crossover but isn't wired in).
- ARCHITECTURE_PATHS.md: GUI description updated to the unified checkable path
  picker (was the old per-mode selector).
- Stale figures fixed: "172 tests" → durable "300+ tests" phrasing, dropped the
  hard-coded "v4.0.0" and "Last updated v4.31.26" stamps.
- The energy axis in MULTIPATH.md now points at both `--energy` tools.

No source-behavior changes beyond the two CLI flags; 325 tests pass and all
tools build + respond to --help.

---


## [4.31.46] — 2026-06-10

### Added — GPU-ready scene-BVH node export + crossover benchmark

The scene BVH (`core/accel/bvh.hpp`) had flattened nodes marked "for a future
GPU upload" but no actual GPU layout or evidence it would help. This exposes the
upload format and measures when it pays off, without the large emitter rewrite a
full GPU traversal would need (and which would regress the common single-object
scenes).

- **`Bvh::gpu_nodes()` / `gpu_node_floats()`** — the flattened hierarchy packed
  for an std430 storage buffer as a `GpuNode` array (three tightly-packed vec4,
  48-byte stride): box min/max in the xyz lanes, and left/right/obj child
  indices bit-cast into the w lanes so the whole thing is one float buffer a
  shader reads back with `floatBitsToInt`. A shader can walk it with an explicit
  stack, pruning by AABB distance — the iterative form of `Bvh::distance()`.
- **Crossover benchmark** (`frep_bench` §6) — flat `min()` over all objects vs.
  `Bvh::distance()` with AABB pruning, over a 32³ sample grid, across 1–128
  objects, reporting the GPU buffer size. On the CPU the BVH wins from ~2 objects
  up (at 32 objects: 10 ms vs 54 ms ≈ 5×; at 128: 36 ms vs 217 ms ≈ 6×), and the
  node buffer stays tiny (255 nodes ≈ 12 KiB at 128 objects). So a GPU BVH upload
  *would* help multi-object scenes — the reason it never showed before is that
  the parity/bench scenes are single-object, where the monolithic min() the GPU
  emits today is already optimal.

This makes the upload buffer ready (`gpu_nodes()`) and documents the crossover,
so wiring GPU traversal in later is an informed choice rather than a guess. A
full GPU traversal also needs per-object SDF functions in the GLSL emitter
(today it emits one monolithic `scene_sdf`), which is the larger follow-on.

Verified in-sandbox: `SceneBvh.GpuNodeLayoutRoundTrips` checks the packing
matches the CPU nodes (box floats equal, child/obj indices survive the bit-cast)
and `GpuNodeIs48Bytes` pins the std430 stride; the crossover benchmark runs.
325 tests pass.

---


## [4.31.45] — 2026-06-10

### Changed — RT pipeline cache wired into RtxExecutor + persistent executors

The RT pipeline cache from 4.31.41 is now actually used by the interactive path,
amortizing RT setup across frames (the safer half of the "more efficient RTX"
work, ahead of a dedicated realtime swapchain).

- **RtxExecutor** now persists its GPU state across `render()` calls: the RT
  device (`RtxCtx`, created once so the cache's objects stay valid), the
  acceleration structure, the compiled SPIR-V, and an `RtxPipelineCache`. A
  scene fingerprint (`scene_key` over geometry pointers + materials + frame
  size) decides when to rebuild shaders/SPIR-V/AS; a camera-only move reuses
  everything and `rtx_trace_cached` then makes the trace a warm-cache hit
  (pipeline_ms ≈ 0, only trace recurs). An explicit destructor releases the
  cache while the device is still alive.
- **ExecutorViewport** now holds persistent per-path executors (`executors_`,
  rebuilt by `ensure_executors()` only when the path set changes) instead of
  constructing fresh executors every frame — which had thrown away the RT cache
  each frame. `set_tracer_config` clears them (config changes the shaders); the
  destructor releases them after the worker thread is joined.

Net effect on real RT hardware: the first frame of a scene builds the pipeline,
subsequent frames while orbiting reuse it. Verified in-sandbox: builds clean,
RT tests pass (15, incl. RtxPipelineCache), and an ExecutorViewport smoke test
confirms executors persist across a same-path re-render and rebuild on a
path-set change without crashing. The warm-cache speedup itself needs a hardware
check (sandbox has no RT device). 323 tests.

---


## [4.31.44] — 2026-06-10

### Changed — single path uses its most efficient backend; multi composites

The path selection now picks the rendering backend, so a single path gets its
dedicated, most-efficient surface while multi-path keeps the compositing view:

- **1 path = gpu_glsl** → the real-time Vulkan **swapchain** (VulkanViewport),
  when hardware Vulkan is present (else offscreen executor as a fallback).
- **1 path = cpu_ir** → the offscreen Viewport in CPU mode.
- **1 path = gpu_ir** → the offscreen Viewport in CUDA mode (CPU fallback if
  CUDA is absent).
- **1 path = gpu_rtx** → the ExecutorViewport (RTX lives only there).
- **2+ paths** → the ExecutorViewport, splitting/compositing across paths with
  the chosen layout and the aggregate-throughput overlay.

`apply_multipath_selection` computes the desired backend (`desired_backend_for`)
and only tears down + rebuilds the viewport when the selection crosses a backend
boundary (single↔multi, or a different single path); otherwise it just re-drives
the existing one. The status bar reports the active backend (e.g. "GPU:
GLSL→Vulkan (realtime swapchain)" vs the multi-path Σ Mpix/s line).

This restores the efficient realtime swapchain for the common single-GLSL case
that 4.31.43 had routed through the offscreen executor, without losing the
unified checkable path picker.

Verified in-sandbox: GUI builds + starts headless clean; a MainWindow smoke
test constructs with the checkable combo and installs the default backend
without crashing. 323 tests pass (no core regressions).

---


## [4.31.43] — 2026-06-10

### Changed — unified path selection: checkable combo + mirrored menu

The Render-tab "Mode" dropdown and the Render menu are reworked from five
mutually-exclusive modes into one path-set model, since every path now runs
through the single ExecutorViewport:

- **Render tab** — "Paths:" is now a checkable-dropdown combo (new
  `CheckableComboBox`): tick one path to render it full-frame, tick several to
  split the frame across them. The closed box summarizes what's ticked; the
  popup stays open while toggling. A "Multi view:" combo picks the split
  (split / weighted / tiles).
- **Render menu** — "Render mode" lists the four paths as independently
  checkable actions, plus a "Multi view" submenu (Split / Weighted / Tiles,
  exclusive). Menu and tab combo stay in lock-step: toggling either updates the
  ExecutorViewport via `apply_multipath_selection()`, with the combo as the
  source of truth.

This removes the old OffscreenCPU / OffscreenGPU / Realtime / Rtx / Multipath
mode switching (and the now-vestigial `switch_render_mode`, kept as a no-op):
there's one viewport, and "which paths" is a selection, not a mode. RTX is
reachable like any other path — tick it.

Verified in-sandbox: GUI builds + starts headless clean; a `CheckableComboBox`
unit test confirms multi-select (tick rtx alongside cpu_ir → both checked;
untick cpu → only rtx), default cpu_ir checked. 323 tests pass (no core
regressions).

### Note
The legacy Viewport (CPU/CUDA) and VulkanViewport (realtime swapchain) backends
are no longer wired into the UI. They remain in the tree; if a dedicated
realtime-swapchain mode is wanted later it can be re-added as an explicit option
rather than overloading the path picker.

---


## [4.31.42] — 2026-06-10

### Fixed — GLSL mesh voxels duplicated across emit bodies

The GLSL emitter builds the SDF body, the dual-AD gradient body, the albedo
body and the PBR body, each by walking every scene node. A MeshSDFNode was
therefore visited multiple times, and `emit_mesh_sdf` appended its voxel grid to
the upload buffer on every visit — so a single mesh produced `mesh_count == 2`
and twice the `mesh_voxels` (e.g. 65536 floats for a 32³ grid instead of 32768),
wasting GPU memory and emit time, with a redundant second `sample_mesh_1`
function.

Fixed with a node-identity dedup map in `MeshAccum`: the first visit allocates
the slot and uploads the grid; later visits of the same node reuse its
`gpu_index`. The sample-function *call sites* are unchanged (the SDF and albedo
bodies still both call `sample_mesh_0`), so rendering is identical — only the
duplicate upload and the duplicate function definition are gone.

Verified in-sandbox: a single 16³ mesh now emits `mesh_count == 1`, exactly
RES³ voxels, one `sample_mesh_0` definition, no `sample_mesh_1`, and the call
appears in both the SDF and albedo bodies (semantics preserved). New test
`GpuMeshSdf.DeduplicatesVoxelsAcrossBodies`; 323 tests pass.

---


## [4.31.41] — 2026-06-10

### Added — RT pipeline cache (amortized setup for interactive RTX)

`rtx_trace` rebuilt the shader modules + ray-tracing pipeline + SBT every call.
For the benchmark that's already excluded from the reported number (it uses
trace_ms only), but for the interactive RTX viewport — where only the camera
moves and the shaders are unchanged — it's pure waste per frame.

New `RtxPipelineCache` + `rtx_trace_cached(ctx, accel, cache, …)`: the cache
holds the shader-dependent objects (pipeline, pipeline/descriptor layouts, the
four shader modules, the SBT), keyed by an FNV-1a hash of the four SPIR-V blobs
plus the texture/mesh binding shape. A matching key on the next frame is a hit:
the build is skipped and only the per-frame work runs (descriptor pool/set bound
to the TLAS + output image, the image itself, the trace, and readback). The
build/per-frame boundary is clean — the descriptor set is rebuilt every frame
regardless — so the cached objects are independent of the camera. On a shader
change the cache self-releases and rebuilds; `cache.release(device)` frees it.
`rtx_trace` is unchanged (forwards to the shared impl with a null cache, so the
one-shot path builds and tears down exactly as before).

Implemented and compiling; the non-cached path is unchanged and all RT tests
pass. The hit/miss reuse itself executes only on a real RT device (the sandbox
has no Vulkan), so the per-frame speedup wants a hardware check — wire
`rtx_trace_cached` into the RtxExecutor/ExecutorViewport with a persistent cache
and confirm pipeline_ms drops to ~0 on frames after the first while the image is
unchanged. A `RtxPipelineCache.FreshCacheIsInvalidAndEmpty` test covers the
value-type invariants that don't need a device. 322 tests.

---


## [4.31.40] — 2026-06-10

### Added — RAPL CPU energy without root (perf_event PMU)

The powercap sysfs counter (`intel-rapl:*/energy_uj`) is root-only on most
kernels since the PLATYPUS mitigation, which is why CPU energy read as
"unavailable" on the hardware run. Added a `perf_event` RAPL path that opens the
package-energy PMU (`/sys/devices/power`, event `energy-pkg`) via
`perf_event_open` — governed by `perf_event_paranoid`, commonly readable
without root. `make_cpu_energy_counter()` now tries perf RAPL first, then
powercap sysfs, then reports unavailable. When CPU energy is unavailable,
`frep_rtx_bench --energy` prints how to enable it (`perf_event_paranoid=0` or
`chmod a+r` on the powercap file). Scale comes from the PMU's `.scale` file, so
the raw counter converts to Joules correctly.

### Added — ExecutorViewport feature parity (SSAA, camera config, status)

The multipath viewport now matches the legacy viewports' controls:
- **SSAA** — renders at ssaa× the widget size (clamped 1–4) then box-downscales,
  with per-path overlay regions scaled back to output coordinates.
- **Camera control config** — orbit sensitivity, zoom step, distance clamps and
  pitch clamp are now honored (were hard-coded), wired from the Render tab.
- **Status bar** — Multipath shows per-path timings plus the aggregate
  Σ Mpix/s; RTX shows its own label. (Spatial-guards toggle stays a no-op here:
  it's a property of the legacy TileScheduler path, N/A to the executor view.)

### Fixed — stale documentation

- `DISTRIBUTED.md`: scheduler and master/worker were marked "planned" but are
  built and were run on a real two-machine LAN (130 tiles, 16/114 split across a
  local + remote worker); updated to "implemented" with that result.
- `FEATURE_MATRIX.md`: RT is 17/17 — mesh + texture on GPU_RTX are
  hardware-confirmed (mesh mean 0.00045, texture 0.0002 on the RTX 2080), no
  remaining feature gap.
- `MULTIPATH.md`: RT parity covers all 17 scenes (not "14, textures pending");
  the energy axis is measured (RAPL + NVML), not a named placeholder.

Verified in-sandbox: 321 tests pass; energy counters probe gracefully to
unavailable here; GUI builds + an ExecutorViewport smoke test confirms SSAA
output sizing, camera config, and per-path status text.

---


## [4.31.40] — 2026-06-10

### Added — RAPL CPU energy via perf_event (no root needed)

The powercap sysfs counter (intel-rapl:*/energy_uj) is root-only on most kernels
since the PLATYPUS mitigation, which is why CPU energy read "unavailable" on the
workstation. Added a `perf_event` RAPL path that opens the package-energy PMU
(`/sys/devices/power`, event `energy-pkg`, scaled by the PMU's `.scale`) — this
is governed by `perf_event_paranoid` and is readable without root on a typical
setup. `make_cpu_energy_counter()` now tries perf first, then powercap sysfs,
then reports unavailable. When unavailable, `frep_rtx_bench --energy` prints how
to enable it (`sysctl kernel.perf_event_paranoid=0`, or chmod the sysfs file).
This closes the CPU side of the energy axis, so pix/kWh can be reported for both
CPU and GPU.

### Added — ExecutorViewport feature parity (SSAA, camera config, status)

The multipath viewport now matches the legacy ones:
- **SSAA** renders at ssaa× the widget size (clamped 1–4) and box-downscales,
  with per-path overlay regions mapped back to output coordinates.
- **Camera-control config** is honored — orbit uses `mouse_sensitivity` and the
  pitch clamp; wheel zoom uses `zoom_step` and the min/max distance clamps
  (was hard-coded before).
- **Status bar** shows per-path timings and, in Multipath, the heterogeneous
  aggregate (`Σ N Mpix/s`); RTX single mode gets its own label.

### Docs — brought in line with what's built + hardware-confirmed

- DISTRIBUTED.md: scheduler and master/worker marked implemented (were
  "planned"), with the real two-machine LAN result (130 tiles, 16/114 split).
- FEATURE_MATRIX.md: RT is 17/17 — mesh + texture on GPU_RTX confirmed on the
  RTX 2080 (mesh mean 0.00045); the "real gap" line is now "none".
- MULTIPATH.md: full RT parity (17 scenes incl. texture/mesh) and the energy
  axis described as measured (RAPL + NVML), not a named placeholder.

321 tests pass. Note: link `frep_tests` with `-j1` in this environment — a
parallel link can exhaust memory and silently produce a 0-byte binary.

---


## [4.31.39] — 2026-06-10

### Fixed — GUI render-mode sync, single RTX, resizable split

Three fixes from using the multipath GUI:

- **Single RTX mode.** RTX was only reachable through Multi (one checkbox).
  Added a dedicated "RTX · Model→RT shaders→Vulkan ray tracing" mode that runs
  one GPU_RTX executor full-frame via the ExecutorViewport — so all four paths
  now appear as single modes (CPU, GPU/CUDA, GPU/Vulkan, RTX) plus Multi.
- **Menu ⇄ Render-tab sync.** The Render menu listed only three modes and no
  Multi; it now shares one entry table with the combobox, so both always show
  the same five modes (CPU, GPU·CUDA, GPU·Vulkan, RTX, Multi) in the same order.
- **Resizable viewport/settings split.** The viewport and the settings tabs are
  now in a horizontal QSplitter (was a fixed HBox). In Multipath mode the path
  panel widened the tabs column and squeezed the render; the splitter is
  user-draggable, non-collapsible, defaults to 900:460, so the render area can
  be reclaimed by hand.

Verified in-sandbox: GUI builds + starts headless clean; a MainWindow smoke
test confirms the render-mode combo has exactly five items in order with RTX and
Multi both present. (The earlier per-tile RT/energy/LAN hardware runs all passed
on the RTX 2080 — RT mesh at the floor → RT 17/17, NVML energy reporting real
Mpix/kWh, and a real two-machine LAN render with 16/114 tile split.)

---


## [4.31.38] — 2026-06-09

### Added — multipath overlay: per-path labels + aggregate throughput

Polish on the Multipath viewport. When more than one path is active, the canvas
now annotates the live render:

- **Per-region label chips** — each path's region (strip or tile-union) gets a
  faint border and a chip reading "path  NN.N Mpix/s", so it's clear which path
  drew which part of the frame and how fast.
- **Aggregate banner** (top-center) — "Σ NN.N Mpix/s (k paths)", the sum of the
  per-path throughputs: the heterogeneous number that's the whole point of the
  multipath study, shown live as the camera moves.

Throughput is computed per path from pixels / (ms × 1000) → Mpix/s; the
ExecutorViewport tracks each path's rendered region + pixel count + ok flag and
exposes them via `path_overlays()` / `aggregate_mpix_s()`. A single active path
shows no overlay (the status bar already has its timing).

Verified in-sandbox: GUI builds + starts headless clean; an overlay smoke test
with two strips on a 240px frame reports two 120×160 regions at +0 / +120 with
per-path Mpix/s and their correct sum as the aggregate — so region geometry,
per-path throughput, and the heterogeneous total all compute correctly.

---


## [4.31.37] — 2026-06-09

### Added — Multipath mode wired into the GUI (path checkboxes + layout)

The ExecutorViewport from 4.31.36 is now selectable in the app. A new "Multi"
render mode in the Mode dropdown installs the ExecutorViewport, and a panel
(shown only in that mode) gives:

- **A checkbox per local path** — cpu_ir, gpu_glsl, gpu_ir, gpu_rtx — toggled
  independently. Checking one renders that single path full-frame; checking
  several splits the frame across them. (RTX is now reachable from the GUI for
  the first time.)
- **A layout selector** — strips (equal vertical bands), weighted (bands sized
  by each path's measured throughput), tiles (interleaved 64px tiles).

Selection changes apply live via `apply_multipath_selection()`. The panel hides
itself in the three legacy modes, which are untouched.

Verified in-sandbox (Qt6 reinstalled): the whole GUI builds (`frep_designer`),
starts headless without crashing (`QT_QPA_PLATFORM=offscreen`), and an
ExecutorViewport smoke test renders both a single path and a 2-path split,
producing a 200×150 composited image with per-path timings — so the worker
thread, progressive compositing, and layout split all work end to end.

---


## [4.31.36] — 2026-06-09

### Added — ExecutorViewport (multipath-capable GUI viewport)

`gui/executor_viewport.{hpp,cpp}` — a new IViewport that renders through the
`exec::IExecutor` abstraction, so it drives ANY of the four paths (CPU_IR,
GPU_GLSL, GPU_IR, and the GPU_RTX path the old Viewport never exposed) and can
render with SEVERAL at once:

- **Single path**: the chosen executor renders the whole frame.
- **Multiple paths**: the frame is split among the active paths by layout —
  equal vertical strips, throughput-weighted strips (bands sized by each path's
  measured pixels/ms EMA), or interleaved 64px tiles round-robin'd to paths.
- Rendering runs on a worker thread; the canvas repaints **progressively** as
  tiles complete (queued to the GUI thread, never touching widgets off-thread).
- Mouse-orbit + wheel-zoom camera, letterbox-fit presentation, per-path timing
  for the status bar.

It's deliberately separate from the legacy Viewport so it can't regress the
existing CPU/CUDA/Vulkan modes. Not yet wired into MainWindow — that's the next
piece (the path-checkbox + layout panel).

Note on the sandbox: Qt6 was reinstalled (`apt-get install qt6-base-dev`), so
the GUI compiles and links here again — the earlier "can't build the GUI"
assumption was wrong, it just needed the package. The whole GUI builds clean
(`frep_designer`), including the new viewport (manual-MOC'd alongside the other
IViewport implementers).

---

## [4.31.35] — 2026-06-09

### Added — shared executor factory (groundwork for GUI path selection)

`core/exec/executor_factory.{hpp,cpp}` centralizes the "path name/kind →
IExecutor" mapping that parity_check and dist_render each had their own copy of.
`make_executor("gpu_rtx")` (and the three other paths) now lives in one place,
so the upcoming GUI path selection, the CLI tools, and the distributed worker
all build executors the same way and can't drift. `local_paths()` lists the four
locally-runnable paths in display order for a UI to enumerate.

parity_check now uses it (one less duplicated mapping). No behavioural change;
this is groundwork for exposing GPU_RTX and multipath rendering in the GUI.

Validated in-sandbox: builds, and parity_check still runs (cpu_ir self-parity at
the floor). The GUI itself can't be compiled here (no Qt6 in the sandbox), so
the viewport/multipath UI lands as separate, independently-buildable pieces.

---

## [4.31.34] — 2026-06-09

### Added — multi-machine LAN distributed run (hostname resolution + retry)

The distributed stack already worked over loopback; this makes a real two-machine
run practical without code changes beyond the worker's `--host`:

- `tcp_connect` now resolves DNS/hostnames via `getaddrinfo` (previously
  `inet_pton`, IP-only), so a worker can target the master by name.
- `tcp_connect` gained a retry window (`WorkerConfig::retry_secs`, `--retry N`
  on the worker tool), so a worker started before the master keeps trying for N
  seconds — the cross-machine startup race, where start times aren't
  synchronized.

`docs/DISTRIBUTED.md` gains a "Multi-machine LAN run" section: master binds
`INADDR_ANY` (accepts any interface), each worker runs one path pointing at the
master's host, and a mixed CPU-box + GPU-box cluster realizes the heterogeneous
aggregate directly (the pull scheduler hands faster devices more tiles).

### Verified — distribution is pixel-exact

Over TCP loopback (with the new hostname + retry path): whole-frame (1 tile),
1-worker/12-tile, and 2-worker/12-tile renders of the same scene are
**pixel-identical** (max byte diff 0). The pull scheduler's split varies with
worker speed (e.g. 5 vs 7 of 12 tiles) but the stitched frame is exact. So a LAN
run's only new variable is the network; the render math is unchanged. New test
`DistTransport.ResolvesLocalhostAndRetries` (binds the listener 300ms late, the
worker connects by name within the retry window). 321 tests.

### To run on two machines

```
# master (host bench-box / 192.168.1.10)
./build/frep_dist_render --master scene.json --port 53900 --workers 2 \
    --decompose grid:64x64 --width 800 --height 600 --out frame.ppm
# worker A (CPU box)
./build/frep_dist_render --worker --host bench-box --port 53900 \
    --paths cpu_ir --width 800 --height 600 --retry 30
# worker B (GPU box)
./build/frep_dist_render --worker --host 192.168.1.10 --port 53900 \
    --paths gpu_glsl --width 800 --height 600 --retry 30
```

---

## [4.31.33] — 2026-06-09

### Added — real energy measurement (RAPL + NVML) for the pix/kWh axis

`core/power/energy_meter.{hpp,cpp}` measures actual energy via hardware
performance counters, so the multipath study can report the efficiency axis
(pix/kWh) with real numbers instead of leaving it a named placeholder:

- **CPU**: Intel/AMD RAPL via `/sys/class/powercap/intel-rapl:0/energy_uj` — a
  monotonic microjoule counter, read before/after and differenced (handles the
  wrap at `max_energy_range_uj`).
- **GPU**: NVIDIA NVML `nvmlDeviceGetTotalEnergyConsumption`, with the library
  `dlopen`'d at runtime so there's no link-time dependency on the driver SDK.

Both are probed at runtime; where a counter isn't available (the sandbox,
non-Intel CPUs, no NVIDIA GPU) the meter reports `available() == false` and the
energy is omitted — **never invented**, consistent with the project's rule that
benchmark numbers come from real instrumentation.

`frep_rtx_bench --energy` wraps the CPU raymarch (RAPL) and the RT trace (NVML)
and adds `cpu Mpix/kWh` / `rtx Mpix/kWh` columns; a dash where a counter is
absent. A warmup render precedes the CPU energy window so JIT compile isn't
charged to energy. The footer explains this is the axis that decides *which*
devices to add to a datacenter — distinct from the throughput capacity axis.

Validated in-sandbox: counters probe gracefully to unavailable, `end()` returns
nullopt rather than a fabricated value, and the pix/kWh math checks out
(1e6 px at 10 J → 3.6e11 pix/kWh). Tests `EnergyMeter.*`. 320 tests.

### Validate on hardware (Intel CPU + NVIDIA GPU)

```
./build/frep_rtx_bench --energy --width 256 --height 256 --counts 1,16,64,256
```

Should print the RAPL/NVML domains and per-path Mpix/kWh. This makes the
energy-efficiency comparison concrete: e.g. if the GPU does more pix/kWh than
the CPU, that argues for growing the cluster with GPUs; if CPU+GPU together is
worse per kWh than GPU alone, that informs whether the heterogeneous sum is
energy-justified for a given deployment.

---

## [4.31.32] — 2026-06-09

### Added — RT mesh descriptor wiring (mesh now on GPU_RTX too)

With textures confirmed at the floor on the RTX 2080 (mean 0.0002), this wires
the last feature gap: MeshSDF on the RT path. `emit_rt_shaders` now accepts mesh
scenes, declaring the voxel buffer at RT binding 3 (2=texture, 1=output, 0=TLAS);
`rtx_trace` uploads the float voxel grid and binds it to the intersection +
closest-hit stages (the mesh SDF is sampled inside the sphere-trace, so the
intersection stage needs it, unlike textures which only the closest-hit shades).
The executor passes `RtShaderSet::mesh_voxels` through.

### Fixed — lifted shared region dropped sample_mesh helper

`lift_shared_region` started at the `float scene_sdf` forward-decl, but the
`sample_mesh_*` helper functions are emitted *before* it and scene_sdf calls
them — so the lifted RT region referenced an undefined `sample_mesh_0`
("no matching overloaded function"). Fixed: the lift now starts at the earliest
of scene_sdf / sample_mesh_* / _unpack_rgb / sample_texture, so all helpers the
region depends on come along. (Textures happened to work before because their
helpers sit after scene_sdf; mesh exposed the latent assumption.)

Validated in-sandbox: all four mesh RT stages compile to valid SPIR-V; voxels
populate (65536 floats for the 32³ scene); textures + analytic scenes still
compile (no regression). 318 tests, with the two obsolete "rejects mesh" tests
replaced by acceptance + SPIR-V-compile tests.

### Validate on hardware

```
./build/frep_parity_check --paths cpu_ir,gpu_rtx --only mesh --width 200 --height 150
```

Should now render and compare (previously a SKIP). At the floor → RT is 17/17,
closing the feature matrix: all four paths cover every node and material type.

---

## [4.31.31] — 2026-06-09

### Added — RT texture descriptor wiring (textures now end-to-end on GPU_RTX)

The shader side landed in 4.31.27 (binding 2 declared); this wires the actual
buffer. `rtx_trace` now takes `texture_pixels` and, when non-empty: adds a
binding-2 storage buffer to the descriptor set layout (closest-hit/intersection/
raygen stages), uploads the RGBA8 texels, and writes the descriptor. The
executor pulls the texels from the shader set (`RtShaderSet::texture_pixels`,
repacked from the GLSL emitter's RGBA8 bytes into uint32 texels) and passes them
through. Analytic scenes pass an empty vector and are unchanged.

This closes the texture gap on the RT path: `emit_rt_shaders` accepts the scene,
the buffer is bound, and the lifted `scene_albedo` samples `tex_data` exactly
like the compute path. The parity harness will now *compare* the texture scene
on gpu_rtx (previously a SKIP) instead of skipping it. Mesh remains a SKIP (its
own binding plan is the next step).

### Fixed — NVPTX transcendental pass crashed scanning function declarations

`lower_transcendentals` iterated every function including bodyless declarations;
scanning those (and not guarding a null `getCalledFunction()`) aborted with
"Illegal instruction" even on scenes with no sin/cos (e.g. a plain sphere).
Added `F.isDeclaration()` skip and an explicit null-callee guard. Caught by the
full test sweep before shipping.

Validated in-sandbox: texture RT shaders emit + the texels repack to 64 uints
for the 8×8 texture scene (0 for analytic); bend/twist/rotate PTX still
self-contained; 318 tests, no crash.

### Validate on hardware

```
./build/frep_parity_check --paths cpu_ir,gpu_rtx --only texture --width 200 --height 150
```

Should now render and compare (not SKIP). If it diverges, the texel repack or
the binding is the suspect; if it's at the floor, RT is 15/17 (mesh still skipped).

---

## [4.31.30] — 2026-06-09

### Fixed (properly) — NVPTX sin/cos via inline polynomial, not libdevice

v4.31.29's first attempt lowered sin/cos to libdevice externs (`__nv_sinf`),
but those aren't linked at JIT time, so `cuModuleLoadData` failed with "a PTX
JIT compilation failed" (unresolved `.extern .func __nv_sinf`). The PTX emitted
fine; the driver just couldn't resolve the symbol.

Fix: `lower_transcendentals` now inlines a self-contained polynomial sin/cos
directly in IR — no external symbols, so the PTX is fully resolvable at JIT.
Range-reduce x to [-π,π] via `k = round(x/2π)` (llvm.round → NVPTX cvt.rni),
then a Horner minimax polynomial (sin: odd, degree 7; cos = sin(x+π/2)).
Verified numerically against std::sin/cos: max error ~2e-4 over [-20,20], far
below the 0.0078 parity tolerance.

Validated in-sandbox: twist/bend/rotate PTX now contains **no** `.extern .func`
declarations (the thing that broke the JIT); sphere/customexpr unchanged. Test
renamed `NVPTX.LowersTranscendentalsInline`, now asserting the PTX is
self-contained. 318 tests.

This should make `cpu_ir,gpu_ir` succeed on twist/bend/rotate — the PTX no
longer references anything the CUDA driver can't resolve.

---

## [4.31.29] — 2026-06-09

### Fixed — GPU_IR (NVPTX) crashed on every sin/cos node (twist/bend/rotate)

Cross-path validation of the new `bend` scene surfaced a latent crash:
`cpu_ir,gpu_ir` on bend aborted with `LLVM ERROR: Cannot select: fsin`. The
NVPTX backend can select sqrt/fabs/min/max (hardware ops) but **not** the
transcendental intrinsics (`llvm.sin/cos/exp/log/pow`) — on CUDA those live in
libdevice. This affected every trig-using node (TwistY, BendXY, RotateY); twist
and rotate had simply never been run on the GPU_IR path before (Phase 2 parity
was cpu_ir↔gpu_rtx), so the bug was latent until `bend` was added and tested
against gpu_ir.

Fix: `retarget_nvptx` now runs `lower_transcendentals()` before PTX emit,
replacing each transcendental intrinsic call with a call to the corresponding
libdevice external (`__nv_sinf`, `__nv_cosf`, `__nv_expf`, `__nv_logf`,
`__nv_powf`), which the CUDA driver resolves when it JITs the PTX. No libdevice
bitcode link step needed. sqrt/fabs/min/max keep selecting natively.

Validated in-sandbox (PTX emission needs no GPU): twist/bend/rotate now emit PTX
with `__nv_sin`/`__nv_cos` calls instead of crashing; sphere/customexpr (no
transcendentals) still emit unchanged. New regression test
`NVPTX.LowersTranscendentalsToLibdevice`. 318 tests.

### Cross-path validation results (RTX 2080 workstation)

- `bend`: cpu_ir↔gpu_glsl at the floor (0.00074); cpu_ir↔gpu_ir was the crash,
  now fixed (awaits a re-run to confirm parity end-to-end on the GPU).
- `mesh`: cpu_ir↔gpu_glsl (0.00096) and cpu_ir↔gpu_ir (0.00074) both at the
  floor — confirms the MeshSDFNode kind fix holds on hardware (GPU_GLSL mesh no
  longer crashes), and mesh is genuinely equivalent across the IR-sharing pair.

---

## [4.31.28] — 2026-06-09

### Fixed — clearer error for >2 paths in parity_check

`--paths cpu_ir,gpu_glsl,gpu_ir` silently took "gpu_glsl,gpu_ir" as one path
name and reported "unknown path". parity_check compares exactly two paths (a
reference vs one target); it now detects a third comma and prints the three
pairwise commands to run instead. No behavioural change to valid two-path runs.

---

## [4.31.27] — 2026-06-09

### Added — RT texture shader support (shader side) + feature matrix doc

`emit_rt_shaders` now accepts textured scenes: it declares the texture storage
buffer at RT binding 2 (0=TLAS, 1=output image are taken) in every stage that
embeds the shared region, so the lifted `scene_albedo`/`shade` can sample
`tex_data`. Validated in-sandbox: all four RT stages for the `texture` scene
compile to valid SPIR-V. Mesh is still rejected (it needs its own binding plan).

The remaining piece for end-to-end RT textures is descriptor wiring in
`rtx_trace` — upload `texture_pixels`, add the binding-2 storage buffer to the
descriptor pool/layout, and write the descriptor. That's the next step; the
shaders are ready and pinned by `RtxShaders.AcceptsTextureSceneAndDeclaresBuffer`.

### Added — `docs/FEATURE_MATRIX.md`

An honest audit of which features each of the four paths supports. Key findings
from the audit:
- MeshSDF is on three paths (CPU_IR/GPU_IR via codegen, GPU_GLSL via the
  sample_mesh emitter), not one — the earlier worry was unfounded; only RT lacks
  it (same as textures).
- Distributed execution is **implemented and tested** (Master/Worker/schedulers/
  TCP, end-to-end loopback test), not a placeholder. `PathKind::Remote` is just
  an enum name; the dist stack coordinates via Master/Worker directly and runs
  any of the four paths through an ExecutorFactory.
- The one real feature gap is MeshSDF + Texture on GPU_RTX (descriptor plumbing,
  now half-done for textures).

### Tests

RT shader tests updated: texture scene accepted + declares the buffer, mesh
still rejected. 317 tests.

---

## [4.31.26] — 2026-06-09

### Fixed — MeshSDFNode left kind default-initialized (GPU_GLSL mesh crash)

`MeshSDFNode`'s constructor never set `kind`, so it stayed the default
`NodeKind::Sphere`. GLSL/codegen SDF dispatch detect mesh via `type_name()` and
were fine, but the GLSL dual-AD emitter (`emit_node_dual`) switches on `kind` —
it took the mesh node for a Sphere and read a nonexistent `"r"` param, throwing
`std::out_of_range` and crashing any GPU_GLSL render of a scene containing a
mesh. Fix: the ctor sets `kind = NodeKind::Plugin`, so kind-based dispatch
routes mesh through the virtual fallback (and dual-AD cleanly declines, falling
back to central-difference normals as for any non-analytic node). This bug was
latent until a mesh parity scene exercised the path.

### Added — `bend` and `mesh` parity scenes (close test-coverage gaps)

- `bend` — a BendXY-deformed box. BendXY existed as a node and has analytic AD
  + GLSL, but was never in the parity set, so cross-path equivalence for it was
  untested.
- `mesh` — a sphere polygonized via marching cubes, then sampled back as a
  MeshSDFNode (res 32). Exercises the voxel-grid mesh SDF on CPU_IR / GPU_IR
  (codegen) and GPU_GLSL (sample_mesh emitter). The RT path SKIPs it like
  textures (no descriptor plumbing yet).

Validated in-sandbox: full CPU run is 17/17 (incl. bend, mesh, texture); the
mesh GLSL shader compiles to valid SPIR-V; the RT path rejects mesh as a SKIP,
not a failure. Cross-path (gpu_glsl/gpu_ir) parity for bend/mesh awaits a GPU.

### Known issue (non-correctness)

A mesh node is emitted twice in GLSL (SDF body + albedo body), so it appears
twice in the mesh accumulator (`mesh_count=2` for one mesh) — duplicate voxel
data uploaded. Renders correctly; wasteful. To dedupe later.

---

## [4.31.25] — 2026-06-08

### Changed — benchmark reframed from "speedup" to throughput + aggregate

A speedup ratio between a CPU path and a GPU path is misleading: the cores
aren't commensurable and dividing by core counts doesn't fix it. `frep_rtx_bench`
now reports per-path throughput (Mpix/s) and their **sum** — because independent
devices add, and running the paths concurrently delivers `cpu + rtx` pixels/s,
which scaling a single device can't reach. That heterogeneous aggregate (and how
it grows as more paths/nodes join) is the system's actual contribution, not any
single ratio. The footer names the other axes — pix/kWh (energy), pix/$ (cost) —
as separate questions the tool deliberately doesn't conflate with capacity.

`docs/MULTIPATH.md` reframed to match: the scaling slopes (CPU ~linear in N, RT
sub-linear via BVH culling) plus the additive aggregate, with energy/cost noted
as named axes needing power instrumentation rather than invented figures.

No behavioural change to the renderers; this is honest measurement framing for
the paper.

---

## [4.31.24] — 2026-06-08

### Milestone — Phase 3 multi-BLAS works on RTX 2080; broad-phase pays off

The full multi-BLAS path (build_groups + rtx_trace_groups + per-group SBT) ran
end to end on the RTX 2080 on the first hardware attempt. The sphere-grid sweep
shows the broad-phase scaling the paper is after — RT trace grows far slower
than the CPU raymarch as object count rises:

```
N=1   cpu~46ms   rtx 0.3ms
N=4   cpu~66ms   rtx 0.3ms
N=16  cpu~137ms  rtx 0.9ms
N=64  cpu~368ms  rtx 2.7ms
N=256 cpu~2400ms rtx 6.5ms
```

CPU raymarch ~linear in N (min over all N spheres per step); RT trace ~22× over
a 256× object increase, because the BVH culls groups a ray misses.

### Fixed — fair benchmark columns (honest paper numbers)

The first benchmark compared CPU *total* (incl. JIT compile) against RT
*trace-only*, which flatters RT. Now `frep_rtx_bench` reports `cpu total`,
`cpu trace` (raymarch hot loop, from the executor's render_ms), and `rtx trace`,
with speedup = cpu trace / rtx trace — like-for-like hot loop, both excluding
one-time setup. A footer states exactly what each column includes and notes both
are full-frame and the GPU is massively parallel, so the scaling *slope* vs N is
the defensible result rather than any single absolute ratio.

`docs/MULTIPATH.md` gains the Phase-3 broad-phase result table with that caveat.

### Status

GpuRtx is complete through Phase 3: fourth path validated at parity on software
and hardware RT (Phases 1–2), and the CSG-aware multi-BLAS broad-phase
demonstrated with a scaling benchmark on the RTX 2080 (Phase 3). Deferred:
textures/mesh on the RT path; amortizing per-frame setup; non-grid benchmark
scenes.

---

## [4.31.23] — 2026-06-08

### Added — Phase 3: multi-BLAS trace (per-group SBT) + scaling benchmark

`rtx_trace_groups()` — the multi-BLAS counterpart of rtx_trace. The RT pipeline
gets one procedural hit group per CSG group (shared closest-hit + that group's
own intersection shader), and the SBT hit region holds one record per group, so
instance i (whose `instanceShaderBindingTableRecordOffset = i` from
build_groups) runs group i's intersection shader. A ray that enters group G's
box sphere-traces only G's sub-tree; groups it misses are never traced. raygen/
closest-hit/miss stay shared, and the timing breakdown (trace vs setup) is
reported as for the single-BLAS path.

`tools/rtx_bench.cpp` (`frep_rtx_bench`) — sweeps the sphere-grid scene over
object counts and times the O(N) flat-union CPU path against the RT multi-BLAS
path, printing the scaling curve (N, groups, cpu ms, rtx trace ms, speedup) with
the backend line so it's clear whether it ran on RT cores. Flags: `--width`,
`--height`, `--counts 1,4,16,64`.

Validated in-sandbox as far as possible without a device: the full multi-BLAS
shader set for a 4-sphere grid (rgen + closest-hit + miss + 4 per-group
intersection shaders) all compile to SPIR-V. The trace itself runs on the
workstation. 315 tests.

### Validate — the scaling curve (the paper's performance result)

```
./build/frep_rtx_bench --width 256 --height 256 --counts 1,4,16,64,256
```

Expect the CPU column to rise ~linearly with N (flat-union O(N) per step) while
the RT trace column rises far slower once N is large enough for broad-phase
culling to dominate the per-frame overhead — the crossover point and the
divergence of the two curves are the headline figure.

---

## [4.31.22] — 2026-06-08

### Added — Phase 3: multi-BLAS acceleration structure + benchmark scene

Two pieces toward the broad-phase scaling study:

`RtAccel::build_groups(ctx, group_boxes)` — builds one BLAS per CSG-group AABB
and a TLAS with one instance per group, each instance's
`instanceShaderBindingTableRecordOffset = i` so the SBT can route it to that
group's own intersection shader. This is the structure that lets the RT cores
cull groups a ray misses, instead of one BLAS covering everything. The move
assignment moves the new per-group vectors too (avoiding the lost-state bug
class we hit with RtxCtx).

`core/exec/bench_scenes.hpp` — `make_sphere_grid(n)` builds a k×k grid of
disjoint spheres as a flat union, framed by the camera. Validated in-sandbox:
n ∈ {1,4,9,16,64} each partitions into exactly n CSG groups, so the per-group
BLAS count tracks n cleanly for the scaling curve.

316 tests (`RtxBenchScenes.SphereGridYieldsNGroups`).

### Next — SBT with N hit records, then the benchmark sweep

`rtx_pipeline` needs an SBT whose hit region has one record per group (each the
group's intersection shader); then a benchmark tool sweeping n and timing the
RT multi-BLAS path vs the O(N) flat-union compute path → the scaling curve.

---

## [4.31.21] — 2026-06-08

### Added — Phase 3: per-group RT shader emission

`emit_rt_group_shaders(full_scene, group_scenes, cfg)` produces the shaders for
the multi-BLAS broad-phase: one shared raygen / closest-hit / miss (lifted from
the full scene, so sky/shade/normal match the other paths), plus one
intersection shader per CSG group, each lifted from that group's own scene so
it sphere-traces ONLY that group's sub-tree SDF.

That last point is the whole performance idea: when a ray enters group G's box,
the RT core runs G's intersection shader, which evaluates one sphere (say), not
the full N-object union. Groups a ray misses are never sphere-traced.

The single-BLAS intersection emit was refactored into a shared
`emit_intersection` helper so the single- and multi-BLAS paths can't drift.

Validated in-sandbox on a 3-sphere union (grid): partitions into 3 groups, emits
3 distinct intersection shaders (different translated SDFs — confirmed by
inequality and by hash), all 6 stages compile to SPIR-V with glslangValidator.
New test `RtxGroupShaders.PerGroupIntersectionsEvaluateOwnSubtree`. 316 tests.

### Next

The Vulkan assembly: build one BLAS per group AABB, a TLAS with one instance per
group (each `instanceShaderBindingTableRecordOffset` selecting its hit group),
and an SBT with the per-group intersection shaders. Then a many-object
benchmark scene to measure RT broad-phase vs the O(N) flat-union compute path.

---

## [4.31.20] — 2026-06-08

### Milestone — GpuRtx Phase 2: validated on RTX 2080 hardware

Ran on a real RTX 2080 (Turing RT cores). The backend line confirms it:
`[gpu_rtx backend] hardware ray tracing on "NVIDIA GeForce RTX 2080" (RT cores,
VK_KHR_ray_tracing_pipeline)`. All 14 analytic scenes are at the floor
(~0.0002 mean) at 400×300 — identical to software RT, confirming parity is
backend-independent (the shaders are the same; the only difference is the RT
cores doing the AABB broad-phase in hardware, which doesn't change the surface).

### Added — honest RT timing breakdown (separates trace from per-frame setup)

The hardware run also gave the first RTX-vs-CPU times, and they need honest
reporting: naive Phase-1 RTX is ~1.1–1.7 s/frame vs ~60–120 ms for the CPU JIT
— *slower*, as expected, because one BLAS over the whole scene means the RT
cores do a single trivial AABB test and all the work is the software
sphere-trace of the full O(N) scene_sdf in the intersection shader, plus
per-frame AS/pipeline/SBT build.

To avoid conflating "RT trace cost" with "our naive per-frame rebuild cost",
`rtx_trace` now reports a timing breakdown (`pipeline_ms` / `trace_ms` /
`readback_ms`); the executor maps trace→`render_ms`, setup→`compile_ms`; and
`parity_check --timing` prints `[gpu_rtx trace X + setup Y]`. A real renderer
amortizes setup across frames, so `trace_ms` is the recurring per-frame GPU
cost the paper should quote. 315 tests.

### Next — Phase 3 integration (the speed story)

Phase 3's per-CSG-group BLAS is what lets the RT cores actually cull, on
many-object scenes, where the O(N) flat-union compute path scales linearly and
RT broad-phase should win. Foundation (group partitioning) is done; remaining
is multi-BLAS build + per-group intersection shaders + the many-object
benchmark.

---

## [4.31.19] — 2026-06-08

### Added — GpuRtx Phase 3 foundation: CSG-aware group partitioning

The broad-phase split that turns the RT path from "one BLAS, full O(N)
scene_sdf per step" into "one BLAS per CSG-independent group, RT cores cull
groups a ray never enters."

`core/gpu/rtx_csg_groups.{hpp,cpp}` — `partition_csg_groups(root)` walks the
F-Rep tree and cuts at hard unions only:

- `Union` (min) is separable — a ray can test each side independently and keep
  the nearer hit, exactly what BVH broad-phase does — so the tree is cut there,
  one group per operand (n-ary / chained unions split fully).
- `SmoothUnion` (smin), `Intersection` (max), `Difference` (max(a,-b)) blend or
  need both operands at a point, so each such sub-tree stays whole in one group.
- A scene with no top-level union collapses to a single group — exactly the
  Phase-1 single-BLAS case.

Each group carries its sub-tree root + world-space AABB, ready to become a BLAS
(rtx_accel already builds N AABBs) with a per-group intersection shader.

This is pure CPU tree logic, so it's fully validated in-sandbox:
`RtxCsgGroups.SplitsOnlyAtHardUnions` checks all the cases (union→2,
smooth/intersection/difference→1, mixed→2, chain→3, lone primitive→1). 315
tests.

### Next (Phase 3 integration, needs hardware)

- Multi-BLAS build from the groups (one AABB each) + a TLAS over them.
- Per-group intersection shaders, each evaluating only its sub-tree's SDF.
- Benchmark: many-object scene, RTX broad-phase vs the O(N) flat union — the
  performance result for the paper.

---

## [4.31.18] — 2026-06-08

### Added — Phase 2 prep: RT backend reporting + render timing in parity_check

Tooling so the RTX 2080 hardware run is unambiguous and yields paper numbers:

- When `gpu_rtx` is one of the paths, `parity_check` prints a
  `[gpu_rtx backend]` line (hardware RT / software emulation / device name), so
  a result clearly states whether it came from real RT cores or llvmpipe.
- `--timing` adds per-scene render-time columns for both paths — the first
  RTX-vs-CPU timing data, and the seed of the Phase 3 performance comparison.
- `docs/RTX_VALIDATION.md` gains a Phase 2 section: probe should classify the
  2080 as Hardware, then `--paths cpu_ir,gpu_rtx --timing` at 400×300 should
  show the hardware backend line, 14/14 at the floor, and real RT timings.

This is the code/tooling side of Phase 2. The actual hardware run is yours to
do when the RTX 2080 is attached — parity is expected to hold identically
(shaders are the same), with the backend line confirming RT cores and the
timings far below llvmpipe's.

---

## [4.31.17] — 2026-06-08

### Milestone — GpuRtx Phase 1 complete: 14/14 analytic scenes at parity

`cpu_ir vs gpu_rtx`, 160×120 on software RT (llvmpipe): every analytic parity
scene is at the shared-SDF floor (~0.0002 mean), `texture` correctly SKIP.

```
sphere 0.000218  box 0.000223  union 0.000217  intersection 0.000217
difference 0.000212  smooth_union 0.000216  twist 0.000234  taper 0.000220
rotate 0.000223  scale 0.000216  customexpr 0.000272  checker 0.000237
stripes 0.000218  gradient 0.000217           texture SKIP   → 14/14 within tol
```

The fourth retargeting path is now validated across the full analytic feature
set: primitives, every CSG op, all transforms, both deformations, customexpr,
and the procedural material patterns — all bit-comparable with CPU_IR despite
running on a fundamentally different execution model (hardware-BVH broad-phase
+ SDF sphere-trace in a custom intersection shader, vs CPU JIT raymarch). Four
heterogeneous executors, one F-Rep scene, equivalent output.

`docs/MULTIPATH.md` updated with the GPU_RTX parity row.

### What's done / what's next

Phase 1 (naive single-BLAS RTX) is complete on software RT. Remaining:
- Phase 2: run the same parity on RTX 2080 hardware (real RT cores).
- Phase 3: CSG-aware per-group BLAS for true broad-phase, benchmarked against
  the O(N) flat union — the performance story for the paper.
- Textures on the RT path (descriptor plumbing) — currently SKIP.

---

## [4.31.16] — 2026-06-08

### Milestone — GpuRtx reaches parity on sphere (fourth path validated)

`cpu_ir vs gpu_rtx` on sphere: mean **0.000218**, max 0.0244 — at the
shared-SDF floor, even tighter than the other path pairs. The fourth
retargeting path (GPU ray tracing) now renders the same implicit F-Rep scene
bit-comparably with CPU_IR, on software RT (llvmpipe). SDF-on-RTX is
demonstrated end to end: hardware acceleration structure broad-phase + SDF
sphere-trace in a custom intersection shader, with the SDF/normal/shade/sky all
lifted from the compute emitter so the result matches by construction.

Also: `parity_check` now prints **SKIP** (not RENDER FAIL) when a path reports
a documented "not yet supported" feature, so a full-suite run over gpu_rtx
cleanly skips the texture scene instead of flagging a failure.

### Validate — widen to the analytic scenes

```
./build/frep_parity_check --paths cpu_ir,gpu_rtx --width 160 --height 120
```

Expect the analytic scenes (sphere, box, union, intersection, difference,
smooth_union, twist, taper, rotate, scale, customexpr, checker, stripes,
gradient) at the floor, and `texture` shown as SKIP. That would complete
Phase 1e — gpu_rtx equivalent across the analytic feature set on software RT.

---

## [4.31.15] — 2026-06-08

### Fixed — RTX background was black; geometry/shading already matched

PPM pixel comparison localized the divergence precisely: the lit sphere matched
cpu_ir *exactly* ((216,82,66) vs (216,82,66), etc.), but the **background**
differed — cpu_ir had the sky gradient (e.g. (102,128,179)) while gpu_rtx was
black (0,0,0). So the ray construction, march, normal, and shading from the
previous alignment were already correct; only the miss/background colour was
wrong (max |Δ| was at pixel (0,0), a corner).

The compute path computes the primary-ray sky as `sky_color_s(0.5 + 0.5*v)`
where `v` is the NDC vertical coordinate — not the ray direction's y. In the
RT split, the miss shader has no NDC. Fix: the raygen (which has `v`) seeds the
payload with that exact sky colour before `traceRayEXT`; the miss shader is now
a no-op so the seeded sky survives, and closest-hit overwrites it on a hit.
The shared region is now also embedded in the raygen so it can call
`sky_color_s`.

All four stages still compile to SPIR-V. 314 tests.

### Validate

```
./build/frep_parity_check --only sphere --paths cpu_ir,gpu_rtx \
    --width 160 --height 120
```

The background should now match; expect mean |Δ| to drop to the shared-SDF
floor (~0.0007), the same as the other paths.

---

## [4.31.14] — 2026-06-08

### GpuRtx renders! Phase 1d works end-to-end; aligning to parity

First real RT trace succeeded on llvmpipe — RT shader compile, pipeline, SBT,
`vkCmdTraceRaysKHR`, readback all ran, producing a sphere image (no crash, no
fail). Initial divergence vs cpu_ir was 0.0117 mean / 0.627 max — the classic
low-mean/high-max signature of a silhouette shift, not a structural problem.
Three RT-vs-compute mismatches found and aligned:

- **Primary ray**: the raygen added a `+0.5` pixel-center offset; the compute
  emitter uses the bare pixel index. Half-a-pixel shift moves the whole
  silhouette → edge fringe. Raygen now uses the compute formula verbatim,
  including the orthographic branch.
- **March loop**: the intersection shader now sphere-traces exactly like
  compute — t from ray-min, `max_dist` checked at the top, step `d*safety`, and
  the same grazing-ray rescue (`last_d < epsilon*80` counts as a hit), instead
  of an ad-hoc loop.
- **Normal**: closest-hit now calls the lifted `scene_normal(p)` (the emitter's
  own normal) rather than a hand-rolled central difference, so the shaded
  normal is identical.

All four stages still compile to SPIR-V; the shader test now checks the
`scene_normal` + `shade(p, n, ...)` usage. 314 tests.

### Validate

```
./build/frep_parity_check --only sphere --paths cpu_ir,gpu_rtx \
    --width 160 --height 120 --dump-images rtimg/
```

Expect the sphere divergence to drop toward the shared-SDF floor (~0.0007). If
a fringe remains, dump-images will show whether it's still silhouette
(geometry) or now surface (shading).

---

## [4.31.13] — 2026-06-08

### Validated — GpuRtx Phase 1a+1b confirmed on llvmpipe; ready for first trace

`frep_rtx_probe` on the workstation now runs the full acceleration-structure
build to completion:

```
RT device created OK on "llvmpipe ..." (software/CPU)
BLAS: getBuildSizes fn=0x... AS size=152 scratch=41144
BLAS: built OK
BLAS+TLAS built OK; TLAS device address = 0x7fffe578d000
```

So the RT foundation — device + feature chain + entry points (1a) and BLAS+TLAS
build (1b) — is validated on software RT. The diagnostic `[rtx_accel]` trace
lines have served their purpose and are removed; the build is quiet again.

Next: the first real trace (Phase 1d) via
`parity_check --paths cpu_ir,gpu_rtx`, which exercises RT shader compilation,
pipeline + SBT creation, `vkCmdTraceRaysKHR`, and readback for the first time.

### Validate

```
./build/frep_parity_check --only sphere --paths cpu_ir,gpu_rtx \
    --width 160 --height 120 --dump-images rtimg/
```

llvmpipe RT is slow — start at 160×120. Expect either a sphere image (compare
`rtimg/sphere_gpu_rtx.ppm` by eye, then the mean/max |Δ|) or a RENDER FAIL line
with the failing pipeline/SBT/trace step.

---

## [4.31.12] — 2026-06-08

### Fixed — Phase 1b segfault root cause: RtxCtx move dropped the RT api table

The parameter trace showed `fn=(nil)` for the build-sizes entry point — yet
`RtxCtx::create()` verified `api().complete()` (all pointers non-null) before
returning. The contradiction was a move bug: `RtxCtx::create()` returns the
context by value (move), and `RtxCtx`'s move assignment copied every member
*except* `api_`. So the loaded RT entry points were verified, then lost in the
move out of `create()`; by the time `RtAccel::build` ran, every RT function
pointer was null and the first call (`vkGetAccelerationStructureBuildSizesKHR`)
segfaulted.

Fix: the move assignment now copies `api_` too (the move constructor delegates
to it, so both are covered). This was the underlying cause of the Phase 1b
crash, not anything in the acceleration-structure code itself.

### Validate

`./build/frep_rtx_probe` — Phase 1b should now run `getBuildSizes` (fn no
longer nil) and continue to `BLAS: built OK` and a TLAS device address. The
trace lines remain for now; they'll be removed once the build is confirmed.

---

## [4.31.11] — 2026-06-08

### Debug — getBuildSizes segfault: param trace + optional validation layer

Phase 1b now gets past buffer device address (the core-symbol fix worked) and
faults inside `vkGetAccelerationStructureBuildSizesKHR`. This release:

- traces the call's parameters (fn pointer, geometryCount, primitiveCount,
  stride, AABB device address) right before it, to confirm none are bogus;
- adds an opt-in Khronos validation layer (`FREP_RTX_VALIDATE=1`) so the driver
  can report what's wrong instead of a bare segfault.

Run both and share the output:

```
./build/frep_rtx_probe
FREP_RTX_VALIDATE=1 ./build/frep_rtx_probe 2>&1 | tail -40
```

The first shows the parameter values at the crash; the second (if the
validation layer is installed) should print a specific message about the
build-sizes call.

---

## [4.31.10] — 2026-06-08

### Fixed — Phase 1b segfault: buffer device address via proc-addr on llvmpipe

The trace pinned the crash to the call right after the AABB upload:
`vkGetBufferDeviceAddress`. Root cause: it was called through a pointer loaded
with `vkGetDeviceProcAddr("vkGetBufferDeviceAddressKHR")`. That function was
promoted to core in Vulkan 1.2, and on llvmpipe the proc-addr lookup for a
promoted-to-core function returns a non-null but unusable trampoline that
faults when called (`complete()` saw it as "loaded", so creation succeeded,
then the call segfaulted).

Fix: call the core `vkGetBufferDeviceAddress` symbol directly (link-time) in
both `rtx_accel.cpp` and `rtx_pipeline.cpp`, instead of the loaded pointer. The
genuinely-extension RT entry points (build-sizes, create/build AS, trace rays)
keep using `vkGetDeviceProcAddr` — that's correct for them.

### Validate

`./build/frep_rtx_probe` — Phase 1b should now print through to
`[rtx_accel] BLAS: built OK` and a non-zero TLAS device address. If it now
faults at a later step, the trace shows which.

---

## [4.31.9] — 2026-06-08

### Debug — acceleration-structure build segfaults on llvmpipe (Phase 1b)

`frep_rtx_probe` on the workstation: Phase 1a (RT device) now succeeds on
llvmpipe, but Phase 1b (BLAS+TLAS build) segfaults. This release adds stderr
trace points through `RtAccel::build` (aabb upload → device address →
getBuildSizes → AS sizes → create handle → record build) so the exact failing
step is visible, rather than just "Segmentation fault".

No behavioural change; diagnostic only. Re-run `./build/frep_rtx_probe` and
share the last `[rtx_accel] ...` line printed before the crash — that pins the
Vulkan call that faults, which maps to a specific fix in `rtx_accel.cpp`.

---

## [4.31.8] — 2026-06-08

### Docs + guards — GpuRtx consolidation before hardware validation

The whole GpuRtx path (Phases 1a–1d) is written and compiles; this release
documents it and adds guards, ahead of the first hardware run.

- `RtxPipeline.PushConstantLayoutMatchesComputePath` — pins every field offset
  of `RtPushConstants` against `ShaderPush` (not just total size), so a future
  edit to either struct can't silently shift the RT path's camera/light inputs
  and break parity. Verified: all 15 fields at identical offsets, both 232 B.
- `docs/MULTIPATH.md` — documents `RtxExecutor` as the fourth path: same
  implicit SDF, hardware broad-phase + sphere-trace in a custom intersection
  shader, parity guaranteed by lifting scene_sdf/shade from the compute emitter
  and reusing its push builder.
- `docs/RTX_VALIDATION.md` — step-by-step hardware validation guide (probe →
  first render → widen to analytic scenes) with the likely first-trace issues
  mapped to specific RT modules.

313 tests.

### State

GpuRtx is code-complete for the naive Phase-1 strategy and validated as far as
the sandbox allows (compiles; all four RT stages produce SPIR-V; push layout
matches). The first real trace needs a Vulkan RT device — see
`docs/RTX_VALIDATION.md`. Remaining: Phase 1e (parity on llvmpipe), Phase 2
(15/15 on RTX 2080), Phase 3 (CSG-aware BLAS + benchmark).

---

## [4.31.7] — 2026-06-08

### Added — GpuRtx Phase 1d: full RT render pipeline (first real trace)

The GpuRtx path is now wired end to end. `RtxExecutor::render()` runs the whole
chain: emit RT shaders → compile each stage to SPIR-V → build the acceleration
structure over the scene AABB → create the RT pipeline + shader binding table →
`vkCmdTraceRaysKHR` → read the image back → crop to the requested tile.

- `core/gpu/rtx_pipeline.{hpp,cpp}` — `rtx_trace()` does the dense Vulkan work:
  shader modules, descriptor layout (TLAS at binding 0, rgba32f storage image
  at binding 1), RT pipeline with a raygen + miss + procedural hit group
  (intersection + closest-hit), SBT assembled from the queried shader-group
  handle/alignment sizes, output image, dispatch, and image→buffer readback to
  host floats.
- `compile_rt_stage_to_spv()` in `glsl_compile.hpp` compiles an RT stage with
  the right glslang stage suffix and `--target-env vulkan1.2`.
- Push constants reuse the compute path's `build_push_from_scene` —
  `RtPushConstants` is `static_assert`-ed layout-compatible with `ShaderPush`,
  so camera basis and lights are byte-identical to gpu_glsl, which is what lets
  the images match.
- `RtxExecutor` now takes a `TracerConfig` (epsilon / safety-factor / fd_eps
  flow into the intersection + closest-hit shaders) and computes the scene
  bounding box (with a small margin) for the BLAS.

Naive Phase-1 strategy throughout: one BLAS over the whole scene, full-frame
render cropped to the tile. Real broad-phase (per-CSG-group BLAS) is Phase 3.

This is the last heavy Vulkan step. The sandbox can't execute it (no device),
so the first real trace happens on llvmpipe / the RTX 2080. 312 tests; all four
RT stages still compile to SPIR-V in-build.

### Validate

`./build/frep_parity_check --only sphere --paths cpu_ir,gpu_rtx --width 200 --height 150`
— first end-to-end RT render on llvmpipe. Expect a sphere image; parity at the
shared-SDF floor is the goal (Phase 1e), though llvmpipe RT is slow, so start
small.

---

## [4.31.6] — 2026-06-07

### Added — GpuRtx Phase 1c: RT shader generation

`core/gpu/rtx_shaders.{hpp,cpp}` — `emit_rt_shaders()` produces the four RT
stages (.rgen/.rint/.rchit/.rmiss) for a scene. The parity-critical move: it
reuses `GlslEmitter` and **lifts the shared region** (everything from
`float scene_sdf` up to `void main()` — the scene_sdf/scene_albedo/shade
helpers) plus the exact push-constant block straight out of the compute
source, then embeds them in the intersection and closest-hit shaders. No second
SDF emitter, so the RT path evaluates byte-identical SDF/shading code:

- **raygen** builds a primary ray from the camera basis matching the compute
  emitter's perspective branch, traces into the TLAS.
- **intersection** sphere-traces `scene_sdf` inside the AABB with the same
  epsilon / safety-factor as the other paths, `reportIntersectionEXT` at the
  surface.
- **closest-hit** computes a central-difference normal (same `fd_eps`) and
  calls the lifted `shade(p, n, view_dir)` — the emitter's real 3-arg signature.
- **miss** returns the background colour.

Lifting the push-constant block from the compute source (rather than
re-declaring it) guarantees the RT stages read camera/light data at identical
offsets — a mismatch would silently break parity.

Phase 1c targets analytic SDF scenes; scenes needing mesh/texture storage
buffers are rejected with a clear message (that descriptor plumbing differs for
RT and lands later).

Validated in-sandbox without a GPU: all four stages compile to SPIR-V with
`glslangValidator --target-env vulkan1.2` for sphere, smooth_union, twist, and
customexpr. New tests `RtxShaders.GeneratesFourStagesForAnalyticScenes` and
`RtxShaders.RejectsTextureScenesForNow`. 312 tests.

### Next

- Phase 1d: compile the stages to SPIR-V at runtime, build the RT pipeline +
  shader binding table, `vkCmdTraceRaysKHR`, read back the image — first real
  RT render. Wire into `RtxExecutor::render()`.

---

## [4.31.5] — 2026-06-07

### Added — GpuRtx Phase 1b: acceleration structure build

`core/gpu/rtx_accel.{hpp,cpp}` — `RtAccel` builds the hardware acceleration
structure for the naive Phase-1 strategy:

- **BLAS** from an AABBs geometry (`VK_GEOMETRY_TYPE_AABBS_KHR`) — the box kind
  whose hits are resolved by a custom intersection shader, which is exactly
  what SDF sphere tracing needs. `build_whole_scene()` makes one box from the
  scene's bounding AABB; `build()` already accepts N boxes, so Phase 3's
  per-CSG-group split only changes how the boxes are computed.
- **TLAS** with a single identity-transform instance over the BLAS.
- Standard KHR flow throughout: describe geometry → `GetBuildSizes` → allocate
  AS storage + scratch (device-address buffers) → `CreateAccelerationStructure`
  → record `CmdBuildAccelerationStructures` in a one-shot command buffer →
  submit + fence wait. Exposes the TLAS handle + device address for the RT
  pipeline (Phase 1d) to bind.

Buffer/memory helpers (device-address allocation, host upload, one-shot
command submission) live in the same file for now; they may move to a shared
RT utility if Phase 1c/1d want them.

This is dense Vulkan the sandbox can't execute (no device here), so it's
validated on llvmpipe / hardware via `frep_rtx_probe`.

### Validate

`./build/frep_rtx_probe` — new "acceleration structure build (Phase 1b)"
section builds a BLAS+TLAS over a 3×3×3 box and prints the TLAS device address
on success, or the failing Vulkan step.

### Next

- Phase 1c: RT shaders (.rgen/.rint/.rchit/.rmiss); the intersection shader
  wraps the existing scene_sdf GLSL so the surface stays implicit and parity
  holds.
- Phase 1d: RT pipeline + shader binding table + `vkCmdTraceRaysKHR` + readback.

---

## [4.31.4] — 2026-06-07

### GpuRtx Phase 1a+: robust feature query + RT entry points loaded

Hardening before acceleration-structure work, and loading the function
pointers Phase 1b will need.

- `RtxCtx::create()` now **queries** the device's RT features with
  `vkGetPhysicalDeviceFeatures2` before enabling them, and fails with a precise
  message if `rayTracingPipeline` / `accelerationStructure` /
  `bufferDeviceAddress` is reported false. A software emulation can advertise
  the extension yet not every feature bit; requesting an unsupported feature
  fails device creation, so we check first.
- Loads the KHR RT entry points via `vkGetDeviceProcAddr` (they are not in the
  core loader): create/destroy/build-sizes/cmd-build/device-address for
  acceleration structures, buffer device address, RT-pipeline creation, shader
  group handles, and `vkCmdTraceRaysKHR`. `RtxCtx::api().complete()` verifies
  all are present; create() fails cleanly otherwise.

This keeps Phase 1b (acceleration structures) building on a verified, fully
loaded RT context rather than discovering missing entry points mid-build.

### Validate

`./build/frep_rtx_probe` — "RT device creation (Phase 1a)" should still report
OK on llvmpipe; if any RT feature bit or entry point is missing on this
emulation, the message now says which.

---

## [4.31.3] — 2026-06-07

### Added — GpuRtx Phase 1a: RT device + feature chain

`core/gpu/rtx_ctx.{hpp,cpp}` — `RtxCtx`, the ray-tracing counterpart to
`VulkanCtx`. `RtxCtx::create()`:

- picks an RT-capable physical device (prefers a non-CPU one, accepts a CPU
  emulation like llvmpipe so the pipeline runs without RT cores),
- creates the logical device with the RT extensions enabled
  (`VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`,
  `VK_KHR_buffer_device_address`, `VK_KHR_deferred_host_operations`) and the
  required feature chain wired through `pNext`
  (`bufferDeviceAddress`, `accelerationStructure`, `rayTracingPipeline`),
- fetches a compute-capable queue.

Getting the feature chain right is the usual first stumbling block, so Phase 1a
isolates it: stand up the device and confirm it, before any acceleration
structure or shader work.

`RtxExecutor::render()` now creates the real device (not just reads extension
lists) and reports it; the actual trace is still Phase 1b–1d.
`frep_rtx_probe` gained a device-creation step so the seam can be validated on
llvmpipe/hardware directly.

### Validate

`./build/frep_rtx_probe` — the new "RT device creation (Phase 1a)" section
should report the device created OK (software/CPU on llvmpipe, hardware on the
RTX 2080). If the feature chain is unsupported, it prints the failure with the
Vulkan code.

---

## [4.31.2] — 2026-06-07

### Fixed — software rasterizer misclassified as hardware RT

`frep_rtx_probe` on the workstation revealed the real picture and a bug:

```
device[0] "NVIDIA GeForce GTX 1050 Ti"  ray_tracing_pipeline=no   (Pascal: no RT cores — correct)
device[1] "llvmpipe (LLVM 20.1.2)"      ray_tracing_pipeline=yes  (software rasterizer emulating RT!)
```

The detector picked llvmpipe and labelled it Hardware, because it only checked
for the RT extensions — but llvmpipe is a CPU software rasterizer with no RT
cores. Advertising the extension is not the same as having the hardware.

Fix: classify by `VkPhysicalDeviceType`. A device is Hardware only if it
advertises both RT extensions **and** is not `VK_PHYSICAL_DEVICE_TYPE_CPU`. A
CPU device that advertises RT (llvmpipe/lavapipe) is Software with a working
Vulkan RT pipeline (useful — it can exercise the real RT shader path on the
CPU, just slowly); a device without the extensions falls back to our own BVH
walk. `RtxCaps::vulkan_rt_pipeline()` reports whether the Vulkan RT pipeline is
usable at all (hardware or emulated), distinct from `hardware()` (real RT
cores). The probe now prints each device's type.

This is genuinely useful: it means the Phase 1 Vulkan RT pipeline can be
validated on llvmpipe (software RT) before running on the RTX 2080 — the same
shader code path, no RT cores required.

New regression test `RtxPath.SoftwareRasterizerIsNotHardware` asserts a
hardware backend is never a recognized software-rasterizer name.

### Note on this machine

The workstation currently has the GTX 1050 Ti (no RT cores) plus llvmpipe. The
RTX 2080 (real RT cores) wasn't attached at probe time. With the fix, this
machine reports Software (llvmpipe RT pipeline) — enough to develop and test
Phase 1; the RTX 2080 will report Hardware when present.

---

## [4.31.1] — 2026-06-07

### Added — RT detection diagnostics (investigating RTX 2080 detection)

On the RTX 2080, `gpu_rtx` reported unavailable, so this release adds tools to
see exactly what Vulkan reports:

- `detect_rtx_caps_verbose(log)` traces the probe: instance creation result,
  device count, and per-device name + whether each RT extension is present.
- Detection now **scans every physical device** and prefers a hardware-RT one,
  instead of only inspecting device 0. On a multi-GPU or headless host the RT
  card may not be first (iGPU or a software rasterizer can be listed ahead of
  it), which alone could explain a false "unavailable".
- `frep_rtx_probe` — standalone tool that prints the verbose trace and the
  resolved backend. Run it on the RTX 2080 to localize the detection issue
  (loader/driver vs device ordering vs extension exposure).
- `parity_check` now prints a failing path's error message (previously it
  showed only "RENDER FAIL"), so the selected RT backend / failure reason is
  visible.

The sandbox probe reports `vkCreateInstance failed (code -9)` — no Vulkan
driver, as expected here.

---

## [4.31.0] — 2026-06-07

### Added — GpuRtx path, Phase 0 (infrastructure)

First step of the fourth retargeting path: GPU ray tracing. The design keeps
the implicit SDF — a hardware acceleration structure does broad-phase (which
object's AABB does a ray enter) and a custom intersection shader sphere-traces
the exact SDF inside, so parity with the other three paths is preserved. API
choice: Vulkan ray tracing (VK_KHR_ray_tracing_pipeline), to reuse the existing
Vulkan stack and degrade gracefully where RT cores are absent.

Phase 0 lays the seam without tracing yet:

- `PathKind::GpuRtx` ("gpu_rtx") added to the path enum and name table.
- `core/gpu/rtx_caps.{hpp,cpp}` — `detect_rtx_caps()` probes the first Vulkan
  device for `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure` by
  extension-name match (no RT SDK headers needed, so it builds on a stock
  Vulkan SDK and in the sandbox). Returns Hardware / Software / None.
- `core/exec/rtx_executor.hpp` — `RtxExecutor` implements `IExecutor`:
  `available()` reflects the detected backend, with an `allow_software`
  opt-out so a benchmark can require real RT cores. `using_hardware()` reports
  whether RT cores (Turing+) are driving it. Phase 0 `render()` reports the
  selected backend and fails cleanly (no trace yet).
- `gpu_rtx` wired into `parity_check` as a selectable path.
- `tests/test_rtx_path.cpp` — 5 tests pinning detection logic, backend
  selection, the software-fallback opt-out, and clean failure when no RT
  device is present. 310 tests.

Hardware: RT cores from Turing (the RTX 2080); a Pascal card (GTX 1050 Ti) or
a host with no Vulkan device selects the software-fallback backend (or
unavailable). The sandbox reports None — as expected, no live RT here.

### Phases ahead

- Phase 1: naive RTX — one BLAS over the whole scene, intersection shader runs
  the full scene_sdf raymarch (parity-trivial).
- Phase 2: validate `gpu_rtx` at 15/15 in the parity harness on RTX 2080.
- Phase 3: CSG-aware BLAS grouping for real broad-phase acceleration; benchmark
  vs the O(N) flat union.

---

## [4.30.7] — 2026-06-07

### Confirmed on hardware — GPU_IR textures, full three-path texture parity

Hardware closes the last texture check: `cpu_ir` vs `gpu_ir` on the `texture`
scene gives mean 0.000735 / max 0.001961 — at the floor, and with an even
smaller max than CPU↔GPU_GLSL because CPU_IR and GPU_IR share the exact same
IR, so the only difference is JIT-x86 vs PTX numeric rounding. The embedded
`scene_textures` global and its `ld.global.nc.u8` lowering work on the real
GPU.

Texture parity is now confirmed on hardware for all three paths:
- CPU_IR ↔ GPU_GLSL: texture 0.000747
- CPU_IR ↔ GPU_IR:   texture 0.000735

That completes the cross-path equivalence program. Every feature in the parity
library (15/15 scenes — primitives, CSG, transforms, deformations, customexpr,
all material patterns, both shading models, image textures) is equivalent
across CPU_IR, GPU_IR, and GPU_GLSL at the shared-IR floor, and textures also
travel intact to distributed workers. One F-Rep scene, three heterogeneous
executors, bit-comparable output — established end to end.

---

## [4.30.6] — 2026-06-07

### Added — texture pixels travel over the wire (Step 1g, distributed)

Completed the last texture task: a textured material now survives the scene
message a distributed worker rebuilds from.

- `serialize_scene(scene, base_dir, embed_textures)` now writes a textured
  material's pixels as `texture_rgba_b64` (base64) plus `texture_width/height`
  when `embed_textures` is set, or whenever the material has pixels but no file
  path (procedural / in-memory textures, which previously vanished on
  serialize). `deserialize_scene` decodes embedded pixels and prefers them over
  any `texture_path`, so the bytes a scene was saved with are the bytes that
  come back — on any machine, with or without the original file.
- The distributed master serializes with `embed_textures=true`, so a worker on
  another host reconstructs the exact texture from the message alone. This
  closes the previously documented distributed limitation; DISTRIBUTED.md
  updated.
- New test `DistRender.TexturePixelsSurviveSceneMessage` round-trips a
  procedural texture through the remote-node path (embed + no base_dir) and
  asserts the pixels come back byte-identical. 305 tests.

This makes textures complete end-to-end: sampled identically on all three
paths (CPU_IR / GPU_IR / GPU_GLSL) AND transported intact to distributed
workers.

### Texture status

- Full three-path texture parity at the shared-IR floor (15/15 scenes).
- GPU_IR texture lowering verified in PTX (`ld.global.nc.u8`); live hardware
  render still the one open confirmation.
- Distributed texture transport: done.

---

## [4.30.5] — 2026-06-07

### Verified — GPU_IR texture lowering to PTX is correct

Sandbox-side confirmation that the embedded texture global lowers correctly
through NVPTX (real CUDA execution still needs hardware, but the PTX is now
known-good):

- The `texture` parity scene's GPU kernel emits valid PTX (44 KB), and the
  module verifies.
- `scene_textures` lowers to a `.global .align 1 .b8 scene_textures[256]` with
  all pixel bytes embedded in the PTX — the data travels with the kernel, no
  separate upload needed.
- The sampler reads it with `ld.global.nc.u8` (read-only data cache, ideal for
  an immutable texture) at `[base]`, `[base+1]`, `[base+2]` for R/G/B, each
  scaled by `1/255` (`0f3B808081`) — byte-for-byte the same lookup as CPU_IR
  and the GLSL shader.

So all three paths share not just the IR but a verified lowering of the texture
data: CPU_IR (JIT'd x86 reading the constant array), GPU_IR (PTX `.global` +
`ld.global.nc.u8`), and GPU_GLSL (SSBO). Full three-path texture parity is
established as far as the sandbox can show; the only remaining check is a live
GPU_IR render on hardware.

### Remaining

- Live GPU_IR texture render on hardware (`--paths cpu_ir,gpu_ir --only
  texture`) — sandbox has no real CUDA.
- Distributed texture transport (Step 1g): `serialize_scene` omits pixels.

---

## [4.30.4] — 2026-06-06

### Confirmed — full three-path feature parity, textures included (15/15)

Hardware confirms the normalize-before-pow4 fix: the `texture` scene dropped
0.0198 → 0.000747, joining every other scene at the shared-IR floor. All 15
parity scenes are now equivalent CPU_IR ↔ GPU_GLSL within ~0.00075 mean |Δ| at
400×300 — the entire feature surface: primitives, every CSG op, all transforms,
both deformations (analytic dual-AD normals), customexpr, every procedural
material pattern, both shading models, and image textures (triplanar nearest).

Image textures were the last GPU-only material feature; they now sample
identically on CPU_IR (embedded module constant + IR triplanar) and therefore
on GPU_IR (shared IR). The cross-path retargeting thesis — one F-Rep scene,
three heterogeneous executors, bit-comparable output — holds across the full
feature set.

### Texture work — remaining

- GPU_IR (NVPTX) texture hardware validation: the embedded `scene_textures`
  global must lower correctly through NVPTX. Sandbox cannot test real CUDA.
  *(Sandbox PTX inspection is strongly encouraging: NVPTX retarget of the
  texture scene produces a valid module and PTX in which the texture lowers to
  `.global .align 1 .b8 scene_textures[256] = {…}` with the correct bytes, and
  the kernel addresses it cleanly via `mov.u64 %rd, scene_textures` + a
  `ld.global` load — the standard module-constant access pattern, no host
  binding needed. Real-CUDA pixel validation still pending.)*
- Distributed texture transport: `serialize_scene` still omits pixel data, so
  procedural textures don't survive the wire. Next task (Step 1g).

---

## [4.30.3] — 2026-06-06

### Fixed — texture sampler returned black where the normal faced the camera

The textured sphere rendered a black band through its centre on CPU (and only
edges showed texture), while GPU rendered it correctly. Isolated with a *solid*
texture + shading off: the centre came out (0,0,0) regardless of camera axis,
so it was the sampled albedo, not lighting or a specific triplanar plane.

Root cause: the IR triplanar weights took `abs(n)^4` of the **un-normalized**
central-difference normal. The raw differences are O(h)=O(1e-3), so `n^4`
underflowed to ~1e-12; the 1e-6 epsilon in the weight sum then dominated and
collapsed every weight toward zero — black — exactly where the geometric
normal was well defined but small in raw magnitude (the camera-facing point).
The GLSL emitter normalizes the normal before `abs()^4`; the CPU path now does
the same (normalize → abs → pow4 → normalize weights), matching the shader.

With a solid texture the surface now reads a single hue modulated only by
shading (correct), and the textured sphere shows the checker across the whole
visible surface, not just the rim.

304 tests. Hardware re-validation of `--only texture` parity pending — expect
the texture scene at the floor now (full three-path feature parity).

---

## [4.30.2] — 2026-06-06

### Texture divergence is resolution-invariant — not aliasing

Hardware confirms the `texture` scene diverges identically at 400×300, 800×600,
and 800×600 SSAA 2× (all ~0.0198 / max 0.83). A resolution-invariant
divergence rules out sampling aliasing — it is a genuine systematic difference
between the CPU and GLSL texture samplers. Sandbox checks confirm the texture
normal (central-diff vs analytic, identical to 3 decimals) and triplanar
weights are not the cause, so the difference is in the UV→texel mapping itself.

### Added — diagnostics

- **`parity_check --dump-images DIR`** writes each scene's two path frames as
  `<scene>_<path>.ppm` for offline pixel comparison (the method that resolved
  the stripes investigation). For the texture scene this will localize the
  UV/texel offset between CPU and GLSL.
- **`--dump-scene` now creates the output directory** instead of failing with
  "cannot write" when it doesn't exist.

### Known issue (carried)

- `serialize_scene` omits texture pixels — dumped texture scenes have no
  `texture_rgba`. Distributed transport fix still pending.

---

## [4.30.1] — 2026-06-06

### Investigating — CPU↔GPU_GLSL texture divergence

Hardware parity_check shows the new `texture` scene diverging (mean 0.0198,
max 0.83) while the other 14 sit at the floor — and it diverges in isolation
too (`--only texture`), so it is a real per-scene sampling difference, not the
harness-reuse artifact that affected stripes/gradient. The texture *is* being
sampled on CPU (colour varies across the surface), so the divergence is in the
exact sampling, not a fallback.

Sandbox bisection ruled out: RGBA byte vs packed-uint unpack (endianness
matches), UV plane assignment (zy/xz/xy matches), the normalize-before-pow4
(cancels in weight normalization), and that the texture normal uses the full
scene_sdf (it does, matching the shader). The remaining suspects need a manual
two-path render to localize (same method that resolved stripes): whether the
0.83 max is high-frequency checker aliasing at this resolution (8×8 checker at
scale 3 is fine relative to the 400×300 grid) or a genuine UV/coordinate
offset between the paths.

`--dump-scene` confirmed a separate real bug: `serialize_scene` omits texture
pixels (texture.json has no `texture_rgba`), so dumped texture scenes lose
their image — the distributed-transport gap, to be fixed in the same pass.

---

## [4.30.0] — 2026-06-06

### Added — image textures on the IR path (CPU_IR + GPU_IR)

Image textures were the last GPU-only material feature. The CPU JIT now
samples them, which GPU_IR inherits for free (shared IR), bringing textures to
all three paths.

- **Texture pixels embedded as a module constant.** `emit_scene_material`
  gathers every textured material's RGBA8 pixels into one flat buffer and
  emits it as a private `[N x i8]` global (`scene_textures`). Because texture
  data is fixed at codegen time (unlike per-frame params), no runtime binding
  or extra function argument is needed — the IR indexes the global directly.
  `texture_pixels()` exposes the buffer; per-entry (offset,w,h) are baked as
  IR constants. `BVH::Entry` carries the pixels/width/height.
- **Triplanar nearest sampling in IR**, matching the GLSL emitter byte-for-
  byte: central-difference scene-SDF normal (h=1e-3), per-axis weights
  `abs(n)^4` normalized, sample the zy/xz/xy planes with `fract`+clamp nearest
  lookup, blend by weight. `emit_scene_material` now takes the scene-SDF
  function so it can compute the texture normal the same way the shader does.
- **`scene_texture` added to the parity library** (15 scenes now): a red/blue
  checkerboard on a sphere, so CPU_IR↔GPU_GLSL↔GPU_IR texture equivalence is
  measured the same way as every other feature.
- **`test_codegen_texture_fallback`** updated: the CPU path no longer falls
  back to solid albedo. New `SamplesTextureColourOnCpu` renders a textured
  sphere and asserts the surface hue varies (impossible for a solid colour).

304 tests. Hardware validation pending for: CPU↔GPU_GLSL texture equivalence
via `frep_parity_check --only texture`, and that GPU_IR (NVPTX) carries the
embedded texture global correctly.

### Still to do (texture work)

- Distributed texture transport: `serialize_scene` still sends only a
  `texture_path`, so procedural textures are lost over the wire — to be fixed
  by serializing pixels into the scene message.

---

## [4.29.4] — 2026-06-06

### Confirmed — full non-textured cross-path parity (14/14 scenes)

Hardware confirms the per-scene-executor fix: every parity scene now sits at
the shared-IR floor, stripes 0.0185 → 0.000746, gradient → 0.000747. All 14
focused scenes are equivalent CPU_IR ↔ GPU_GLSL within ~0.00075 (mean |Δ|)
across the entire non-textured feature surface: primitives, all CSG ops, all
transforms, both deformations (analytic dual-AD normals), customexpr, all
procedural material patterns, and both shading models.

Four real bugs were found and fixed along the way, each surfaced because the
framework tested many focused scenes rather than one busy scene:

1. TaperY GLSL formula (wrong scale + missing Lipschitz divisor)
2. Deformation normals: central-difference on CPU vs analytic on GPU →
   rewrote CPU twist/taper gradients as analytic forward-mode AD
3. Shadow ray marched past the light on CPU (false self-shadows) — the twist
   divergence, also a genuine CPU shadow-quality fix
4. parity_check reused one GPU executor across all scenes (harness artifact)

The only remaining cross-path feature gap is image textures (currently
GPU-only), addressed next.

---

## [4.29.3] — 2026-06-06

### Fixed — parity_check reused one GPU executor across all 14 scenes

Tracked the stripes/gradient mismatch to the harness, not the render paths.
Proof chain: a sandbox `cpu_ir` render of `stripes` at 400×300 is bit-identical
to the user's manually-rendered `cpu.ppm` (mean 0, max 0); that `cpu.ppm`
matches the manually-rendered `gpu.ppm` to mean 0.0002 (bit-equivalent, stripe
bands aligned to 1 LSB) — so both paths agree when each renders the scene in a
fresh executor via `frep_multipath`. Yet `parity_check` reported 0.0185 for the
same scene at the same resolution.

The difference was that `parity_check` created **one** `GpuGlslExecutor` and
reused it across all 14 scenes; a reused executor carries Vulkan
context/buffer state between renders (the context is rebuilt on source change,
but GPU-side resource lifecycle across 14 sequential scenes is not what a
parity comparison should fold in). Now the harness recreates both executors
per scene, matching how `frep_multipath` renders a single scene — so each
comparison is independent.

Also fixed `--only SCENE` (added in 4.29.2) actually filtering to one scene.

If hardware now shows stripes/gradient at the floor, all 14 parity scenes are
confirmed equivalent CPU↔GPU_GLSL — full non-textured feature parity, and the
earlier stripes/gradient "divergence" was a harness artifact.

---

## [4.29.2] — 2026-06-06

### Investigating — parity_check vs manual render mismatch for stripes

A puzzle surfaced: manually rendering `stripes` on each path at 400×300 and
comparing the PPMs gives mean |Δ| ≈ 0.0002 (bit-equivalent — stripe bands
align to 1 LSB), yet `parity_check --width 400 --height 300` reports 0.0185 for
the same scene. The scene is identical (programmatic vs JSON round-trip render
bit-identical on CPU), the config is identical (both default CookTorrance), so
the difference is in how the GPU path is invoked in-process across 14
sequential scenes vs a single isolated render.

Added **`parity_check --only SCENE`** to render just one scene in isolation,
matching the manual single-scene render. If `--only stripes` reports ~0.0002,
the harness's full-sweep number is a sequential-GPU-state artifact (e.g. a
per-frame seed or buffer not reset between scenes), not a path divergence —
which would mean all 14 scenes are in fact at the floor.

---

## [4.29.1] — 2026-06-06

### Resolved — stripes / gradient divergence is resolution aliasing, not a bug

Direct pixel comparison of CPU vs GPU_GLSL `stripes` settled it. At the
parity harness's 200×150 the two diverge (~0.018), but rendered at 400×300 and
compared pixel-for-pixel they are **bit-equivalent**: 119998 of 120000 pixels
differ by < 0.01, two pixels by < 0.05, none above. The divergence is entirely
**boundary aliasing** — at scale 8 the stripes are only ~8 px tall at 200×150,
so a half-pixel difference in where a stripe edge lands flips a boundary
pixel's colour; at 400×300 (~16 px stripes) the edges are resolved and the
difference vanishes. The same applies to `gradient`. This is why SSAA 2× only
partially helped at 200×150 (still under-sampled) and why every smooth scene
was already at the floor.

So all 14 parity scenes are equivalent CPU↔GPU_GLSL once adequately sampled —
full non-textured feature parity. The shared-IR floor (~0.0008) holds across
the entire feature surface: primitives, CSG, transforms, both deformations
(analytic AD normals), customexpr, all material patterns, and both shading
models. The remaining gap is image textures (GPU-only), addressed next.

### Changed

- `MULTIPATH.md` notes that pattern scenes need resolution proportional to
  pattern frequency for a clean comparison (a sampling property, not a path
  difference), and records the 400×300 stripes result.

---

## [4.29.0] — 2026-06-06

### Confirmed — twist at the floor (12/14 scenes in parity)

Hardware confirms the shadow-`max_t` fix: twist 0.122 → 0.000789, joining the
~0.0008 floor. Cross-path parity now holds for every feature scene except
`stripes` and `gradient`: primitives, all CSG ops, all transforms, both
deformations (twist/taper, analytic AD normals), customexpr, checker, and both
shading models all agree CPU↔GPU_GLSL within the shared-IR floor.

### Added — `parity_check --dump-scene DIR`

Writes each parity scene as `<name>.json` so a divergence can be rendered and
inspected manually on hardware:

```
frep_parity_check --dump-scene scenes/
frep_multipath scenes/stripes.json --paths cpu_ir   --out cpu.ppm
frep_multipath scenes/stripes.json --paths gpu_glsl --out gpu.ppm
```

This is the next step for `stripes`/`gradient` (~0.018, sandbox bisection
exhausted): render both paths and see whether the difference sits in the
saturated `albedo2` bands or is uniform, which localizes it to the albedo path
vs the shading application.

---

## [4.28.2] — 2026-06-06

### Fixed — shadow ray marched past the light (the real twist cause)

The standoff alignment (v4.28.1) fixed every other scene's max |Δ| (sphere
0.051→0.005, etc.) but twist stayed at 0.122 — proving the twist cause was
elsewhere in the shadow march. Step-by-step numeric trace of the two shadow
rays revealed it: the CPU shadow ray's loop guard was `t < max_dist` (100),
while the GLSL emitter breaks at `t > max_t` where **max_t is the distance to
the light**. The CPU ray therefore marched *past the light* and could hit
geometry behind it, reporting a false shadow. A twisted bar lit from above has
exactly such occluders beyond the light along some surface rays, so twist
alone showed it.

Added a `max_t` argument to the JIT `shadow_ray` (the light distance, already
computed at both call sites) and made the loop stop there. This is also a
genuine CPU shadow-quality fix, not only an equivalence fix: spurious shadows
from geometry behind the light no longer appear. Twist should now join the
floor — full non-textured feature parity except stripes/gradient.

### Still open — stripes / gradient

Unchanged (~0.018). Sandbox bisection exhausted (formula, blend, ambient,
diffuse, gamma all proven identical; not shadows/AO/specular/aliasing). Needs
GPU-side instrumentation — likely in how a saturated, spatially-varying albedo
feeds the shaded output.

---

## [4.28.1] — 2026-06-06

### Fixed — twist shadow standoff (all three parameters together)

v4.28.0 changed only two of the three shadow-ray parameters and made twist
*worse* (0.115 → 0.121). The miss: the shadow standoff is the sum of a normal
offset at the origin **and** the ray start `t`, and they must be aligned
together. Full comparison:

| param                 | CPU (was)           | GLSL   |
|-----------------------|---------------------|--------|
| origin normal offset  | `epsilon*8` = 0.004 | 0.01   |
| ray start `t`         | `epsilon*4` = 0.002 | 0.05   |
| in-object epsilon     | `epsilon` = 0.0005  | 0.002  |

CPU's combined standoff was ~0.006; GLSL's is ~0.06 — 10× closer to the
surface on CPU, so a twisted bar (which self-shadows through its own
concavities) sampled occlusion at different depths between the paths. Aligned
all three CPU shadow parameters to the GLSL emitter. Smooth convex shapes are
insensitive to the standoff, which is why only twist diverged.

### Still open — stripes / gradient

Exhausted sandbox bisection: not shadows/AO/specular (toggles), not edge
aliasing (SSAA), and the pattern formula, albedo/albedo2 blend, ambient and
diffuse application, and the absence of gamma encoding are all proven
identical CPU↔GLSL. That `gradient` (smooth, no discretization) and `stripes`
(discrete) diverge by the *same* 0.0185 while `checker` (near-neutral
albedo2) sits at the floor suggests the difference involves how a saturated
spatially-varying albedo feeds the shaded output — needs GPU-side
instrumentation to localize.

---

## [4.27.0] — 2026-06-06

### Fixed — deformation cross-path equivalence (twist, taper)

The systematic parity harness (v4.26.0) immediately earned its keep: of 14
feature scenes, `frep_parity_check --paths cpu_ir,gpu_glsl` on hardware
flagged twist (mean 0.115), taper (0.0088), stripes and gradient as divergent
while the other ten sat at the ~0.0008 shared-IR floor. Two real bugs found,
both in the GLSL emitter / the CPU normal path:

- **TaperY formula mismatch** — the GLSL emitter used `s = y*t + 1` with no
  Lipschitz divisor, while the CPU path uses `u = clamp((y+0.5h)/h, 0, 1)`,
  `s = max(1 + u*(t-1), 1e-3)`, child scaled by `1/s`, result divided by
  `max(1, 1/s)`. Rewrote the GLSL TaperY (dual arithmetic) to match the CPU
  formula exactly. Added `d_max_s`, `d_clamp_s`, `d_one` dual helpers.
- **Deformation normals were central-difference on CPU, analytic on GPU** —
  `TwistY`/`TaperY::codegen_grad` used a finite-difference gradient (h=1e-3)
  while the GLSL emitter used analytic dual-AD, so the normals (hence shading)
  diverged. Following the project's analytic-over-numeric principle, rewrote
  both CPU gradients as **analytic forward-mode AD** matching the GLSL dual
  emitter (added `sin`, `cos`, `clamp_s`, `max_s` to `ad_ir.hpp`). This both
  closes the gap and improves CPU normal accuracy.

stripes/gradient/customexpr divergence is consistent with silhouette/edge
aliasing rather than an algorithmic difference (identical epsilon and pattern
formulas); `frep_parity_check --ssaa N` now supersamples both frames before
comparing to confirm this on hardware.

### Added

- **`frep_parity_check --ssaa N`** — supersample N× then box-downsample both
  frames before measuring, isolating edge aliasing from genuine divergence.
- **`PathParity.DeformationScenesRenderStably`** test — guards the analytic
  deformation gradient path. 304 tests.

### Changed

- **Gallery rendered at SSAA 2×** — demo images now supersample for clean
  silhouettes (no stray bright edge pixels). All eight CPU-rendered scenes
  regenerated; the two GPU-only texture scenes kept.

---

## [4.26.0] — 2026-06-06

### Added — systematic cross-path equivalence: many focused scenes

Equivalence was demonstrated from three busy showcase scenes; a per-feature
regression could hide inside a busy scene's aggregate. Added a library of
small, single-feature scenes and harnesses that walk all of them, so parity
is shown across the whole feature surface.

- **`core/exec/parity_scenes.hpp`** — 14 focused scenes (sphere, box, union,
  intersection, difference, smooth_union, twist, taper, rotate, scale,
  customexpr, checker, stripes, gradient), each one or two objects on a shared
  floor with one camera and one warm light, so a divergence localizes to the
  feature under test.
- **`tests/test_path_parity.cpp`** — renders every scene on two independent
  CPU_IR executors and asserts bit-identical frames (determinism across the
  whole surface), and checks both shading models render every scene. 3 tests.
- **`frep_parity_check`** — renders every scene on two chosen paths, prints a
  per-scene mean/max |Δ| table and aggregate, exits non-zero on divergence.
  On hardware: `--paths cpu_ir,gpu_glsl`; in a CPU-only sandbox,
  `--paths cpu_ir,cpu_ir` exercises the harness (all zeros). 303 tests.

### Noted

- Image textures remain GPU-only and are deliberately excluded from the parity
  library until they render on all three paths (next).
- **Distributed texture transport gap** documented in `DISTRIBUTED.md`:
  `serialize_scene` sends only a `texture_path`, so a remote worker loses
  procedurally-generated textures (no path) and needs the file present for
  path-based ones. To be fixed with the texture-parity work by serializing
  texture pixels into the scene message.

---

All notable changes to FRep Designer are documented in this file.

The format roughly follows [Keep a Changelog](https://keepachangelog.com/),
and the project adheres to [Semantic Versioning](https://semver.org/).

---

## [4.25.2] — 2026-06-05

### Added — gallery scenes 08–10 are now generated by `frep_gallery`

Scenes 08 (gyroid), 09 (capsule), 10 (textured) previously existed only as
committed images with no generator. Added them to `tools/gallery.cpp` so the
whole gallery is reproducible from one tool:

- **08 gyroid** — `CustomExprNode` with `abs(sin2x·cos2y + …) − 0.4`,
  intersected with a sphere. Regenerated (CPU-equivalent).
- **09 capsule** — capsule SDF as a `CustomExprNode`
  (`sqrt(x² + max(0,|y|−h)² + z²) − r`), no plugin dependency. Regenerated.
- **10 textured** — procedural wood sphere + marble cube. Texture sampling is
  GPU-only, so this scene (like 06) renders on the GPU path; on a CPU-only
  host the generator skips it and keeps the committed image.

Texture scenes (06, 10) are now explicitly GPU-only in the generator (the CPU
JIT can't sample image textures), so a CPU regeneration no longer silently
produces untextured versions. GALLERY.md updated to match. 300 tests
unchanged (gallery is a tool, not under test).

---

## [4.25.1] — 2026-06-05

### Changed — regenerated the examples gallery with the corrected renderer

The committed gallery images predated the shading and camera fixes, so they
showed the old defects — most visibly a fish-eye floor (curved perspective)
and the pre-light-colour/pre-sky shading. Regenerated scenes 01–07 with the
current renderer: straight perspective, per-light colour, NDC sky gradient,
cancelled-form specular, analytic dual-number normals, aligned raymarch.

`frep_gallery` now falls back to the CPU_IR path when no GPU is present
(instead of skipping GPU scenes). Since CPU_IR and GPU_GLSL are visually
equivalent (~0.0008), the regenerated images match what the GPU path
produces; on real hardware the GPU path is still used. (Scenes 05_gpu and
08–10 are not produced by the generator and were left for a hardware pass.)

---

## [4.25.0] — 2026-06-05

### Changed — unified CLI vocabulary across all tools (breaking)

The CLI tools predated the path / target / stage concepts and had drifted
apart. Unified them around the current vocabulary. Breaking changes (single
user, so a hard rename rather than aliases):

- **`--paths LIST`** is now the single way to select executor paths
  (`cpu_ir,gpu_glsl,gpu_ir`), replacing the older per-tool spellings:
  - `frep_bench --cpu-only` → `--paths cpu_ir` (GPU rows shown only if a
    `gpu_*` path is listed and the GPU is available).
  - `frep_dump --no-spirv` → `--paths` (path→artifact: `cpu_ir`/`gpu_ir` emit
    the `.ll` dumps, `gpu_glsl` emits `.glsl`/`.spv`).
  - `frep_dist_render --worker --path X` → `--paths X` (a worker still runs
    exactly one path; a comma-separated list is rejected with a clear error).
- **`--width N` / `--height N` / `--out FILE`** everywhere, replacing the old
  positional `[out.ppm] [width] [height]` in `frep_render` and `frep_advanced`.
- **`frep_advanced --spirv`** → `--emit spirv` (clearer: it adds a SPIR-V emit
  to the CPU render, it doesn't switch the render path).
- **`--json`** (frep_bench) kept as-is.

### Removed

- **`frep_gpu_render`** — deprecated and removed; it duplicated what
  `frep_multipath <scene> --paths gpu_glsl --merge stitch` does. Docs updated
  to point at `frep_multipath` (and `frep_dump --paths gpu_glsl` for shader
  artifacts).

300 tests, all tools build and run with the new flags.

---

## [4.24.0] — 2026-06-05

### Added — distributed master/worker + scheduler: end-to-end distributed render

The rest of Direction 2b on top of the transport layer: tile scheduling, the
master, the worker, and a CLI driver. A heterogeneous cluster now renders one
frame cooperatively over TCP.

- **`core/dist/scheduler.hpp`** — `IScheduler` with two models:
  `PullScheduler` (work-stealing: a shared atomic cursor, any worker takes the
  next tile — the distributed analogue of the local `DynamicQueue`, self-
  balancing across machines) and `PushScheduler` (round-robin pre-assignment,
  no rebalancing, for known-capability workers).
- **`core/dist/master.hpp`** — binds a listener, accepts N workers, sends the
  scene once, then serves tiles through the scheduler (one thread per worker,
  so pull gives concurrent work-stealing). Collects results into a tile vector
  for stitching; reports per-worker tile counts.
- **`core/dist/worker.hpp`** — connects, receives the scene, builds its
  configured executor (CPU_IR / GPU_GLSL / GPU_IR — compiles locally), loops
  request→render→result. A worker is a thin shell around an `IExecutor`, so a
  heterogeneous cluster (CPU worker + GPU worker) falls out naturally and the
  ~0.0008 path equivalence keeps the stitched frame seamless.
- **`frep_dist_render`** — one binary, `--master` and `--worker` roles. Master
  decomposes (grid), schedules (pull/push), stitches, SSAA-downsamples, writes
  PPM. Worker renders its pulled tiles with `--path`.

Verified on localhost: a 2-worker master run stitches **bit-identically** to a
whole-frame local render (real TCP loopback, separate processes), with the two
workers splitting tiles 6/6 under pull scheduling. 2 integration tests
(pull matches whole-frame, push covers frame). 300 total.

The PoC runs master + N worker processes on one host now; the same binaries
run across the LAN by pointing `--worker --host` at the master's address —
the multi-machine heterogeneous experiment.

---

## [4.23.0] — 2026-06-05

### Added — distributed transport layer (Direction 2b foundation)

First layer of distributed rendering: the transport abstraction and wire
protocol. In the path model the network hop is just another element of the
path — a tile crosses a transport and a RenderStage runs on the far machine —
so transport is an abstract interface with interchangeable implementations.

- **`core/dist/transport.hpp`** — `ITransport` (abstract reliable ordered
  bidirectional byte stream) + a tiny framed wire protocol
  (`[type:u32][len:u32][payload]`, little-endian). Message types: Hello,
  Scene, TileRequest, TileAssign, TileResult, NoMoreWork, Error. Plus
  (de)serialization helpers for tiles and tile results (region + RGBA floats,
  pixel count derived from dimensions).
- **`core/dist/tcp_transport.hpp`** — `TcpBinaryTransport`, the first concrete
  transport: POSIX TCP sockets, TCP_NODELAY, blocking send-all / recv-exact,
  with `TcpListener` (master accept) and `tcp_connect` (worker). No external
  dependencies.
- **`docs/DISTRIBUTED.md`** — design: transport (done), scheduler (pull
  primary + push, planned), master/worker (planned), what's sent (scene once,
  then tiles↔RGBA), and a recorded future variant: shipping the *compiled*
  artifact as its own path stage instead of the scene JSON.
- 4 tests including a real localhost TCP loopback exercising framed messages
  end to end (runs in the sandbox — loopback needs no GPU/real network). 298
  total.

Scheduler, master, and worker come next; the PoC will run master + N worker
processes on localhost first, then real multi-machine LAN runs.

---

## [4.22.0] — 2026-06-05

### Added — more post-process stages: tone-map, gamma, bilateral denoise

Filled out the post-process pipeline with the common stages, all composable
through `PostProcessPipeline` and applied to the whole stitched frame:

- **`ToneMap`** (Reinhard or ACES) — compress HDR radiance into [0,1];
  pixel-wise on RGB, exposure-scaled. Rolls off highlights instead of
  clipping them flat.
- **`GammaCorrect`** — linear→display gamma encode (default 2.2). Applied
  identically across all paths, so it doesn't perturb cross-path equivalence.
- **`BilateralDenoise`** — edge-preserving spatial+range filter over a
  (2r+1)² window. A genuine neighbour-based stage, so it must run on the
  assembled frame (per-tile would seam across a split) — the same reason SSAA
  goes after the stitch.

`frep_multipath` gains `--tonemap reinhard|aces`, `--gamma G`, `--denoise`.
The pipeline order is downsample → denoise → tone-map → gamma (resolve,
clean, compress, encode); the diff heat map skips tone-map/gamma (radiance
operators would distort a diagnostic). 6 tests added (Reinhard midpoint, ACES
clamp, gamma midtone, denoise flat-region invariance + resolution, multi-stage
order). 294 total.

---

## [4.21.0] — 2026-06-05

### Added — PostProcessStage: the first formal path stage; SSAA on the stitched frame

Post-process (so far SSAA) was inlined in the viewport. Pulled it out into a
reusable stage abstraction — the first formal stage in the path model
(Model → emit → render → **post-process** → present, docs/ARCHITECTURE_PATHS).

- **`core/postprocess/post_process.hpp`** — `Frame` (float RGBA + dims),
  abstract `PostProcessStage` (image→image, may resize), `BoxDownsampleSSAA`
  (the exact box filter the viewport used, now reusable), and
  `PostProcessPipeline` (compose stages in order; empty = identity).
- **`frep_multipath --ssaa N`** renders the whole pipeline at N× resolution,
  stitches, then box-downsamples the *assembled* frame as a post-process
  stage. Applying the downsample after the stitch — not per tile — is the
  architectural point: a CPU+GPU split supersamples correctly across the
  seam, because hi-res samples straddling the band boundary are averaged
  together. Verified: split+SSAA is bit-identical to whole-frame+SSAA (the
  seam stays invisible through the downsample).
- 5 tests (factor-1 pass-through, resolution halving, block averaging, empty
  pipeline identity, ordered application). 288 total.

This is the groundwork for the staged path model: post-process is now a
first-class, composable step that runs on a whole frame regardless of how
that frame was produced (single path, band split, or dynamic tile queue).
Future stages (denoise, tone-map, DOF) slot into the same pipeline.

---

## [4.20.1] — 2026-06-05

### Added — per-executor distribution summary in frep_multipath

Hardware dynamic-queue runs showed the work-stealing balance clearly (GPU
took 19 of 20 tiles, CPU 1) but the per-tile list is long. Added a
`distribution:` summary line per executor — tile count, total compile, total
render — so the balance and whether compile dominates are legible at a
glance. The hardware run made the key point visible: the GPU's 19 tiles cost
~380 ms total while the CPU's single tile cost ~1105 ms, almost all of it the
1.1 s JIT compile. For a single frame, an executor with an expensive compile
(CPU JIT) can be net-negative even taking one tile; the work-stealing queue
correctly starves it (1 tile), but compile still dominates wall-time until it
is amortized across frames. The split's benefit is a multi-frame /
warm-cache property — consistent with the weighted-split finding. No
behavioural change; reporting only.

---

## [4.20.0] — 2026-06-05

### Added — dynamic work-stealing tile queue (self-balancing split, no calibration)

A third frame-split strategy alongside equal and weighted bands: decompose
the frame into a grid of tiles and run a `DynamicQueue` — one worker thread
per executor, each pulling the next tile from a shared atomic cursor as it
finishes. A faster executor completes its tile sooner and grabs the next, so
it renders proportionally more tiles automatically. Advantages over weighted
bands:

- **No calibration** — the balance emerges from the pull rate; the first tile
  each worker takes warms its compile cache (cold start), every later tile
  reuses it (`compile 0.0 ms`).
- **Robust to per-region cost variation** — a tile over geometry and a tile
  over sky are scheduled independently, so the slow-then-fast row problem
  that skewed the strip calibration doesn't arise.
- **Self-tuning across frames** — no fixed split to go stale.

Use `--mode dynamic` with `--decompose grid:WxH`. Tile size is the usual
trade-off: large enough to amortize per-tile dispatch overhead, small enough
that many tiles keep all executors fed. Verified: a dynamic-queue grid render
stitches bit-identically to a whole-frame render (correctness), and the
compile cache warms after the first tile per worker. `RunMode::DynamicQueue`;
1 test added (283 total). Equal (`halves`/`bands:N`) and `weighted` strips
remain for comparison.

### Removed — stray opencl-headers .deb from the source tree

A leftover `opencl-headers_*.deb` (from the early OpenCL experiment, before
the CUDA pivot) was sitting in the project root and got packaged. Removed;
the bundled `third_party/CL/*.h` headers are what the (retained) OpenCL
context code uses.

---

## [4.19.2] — 2026-06-04

### Fixed — weighted-split calibration samples the frame centre, not the top

Hardware showed the calibration mis-estimating badly: the CPU probe on a top
strip read 4.7 ms, but the real CPU band (which dipped into the geometry)
took 102 ms — a 19:1 real ratio vs the 1.5:1 the probe guessed, so the
"balanced" split left the CPU band dominating (116 ms vs 22 ms GPU-alone).
Cause: the top strip is mostly empty sky (cheap miss rays), unrepresentative
of a band over geometry. The probe now samples a strip from the vertical
centre of the frame, where the object mass is, so the per-row cost estimate
reflects the real band cost. (Sandbox check: a CPU+CPU central-probe split
now balances to ~225/221 ms bands.)

### Note on when the weighted split wins

The three-way hardware timing also clarified the regime: for a single render
where one executor is ~19× faster (GPU vs CPU here), GPU-alone still wins —
the split's benefit is bounded by the slow path's share. The weighted split
pays off when the executors are closer in throughput, or across many frames
where compile is amortized and only the balanced render time remains. This is
the honest heterogeneous-execution picture for the paper: a CPU+GPU split
helps in proportion to how comparable the executors are, not unconditionally.

---

## [4.19.1] — 2026-06-04

### Fixed — CpuIrExecutor caches its compile (weighted calibration no longer doubles it)

The weighted split's calibration pass renders a trial strip per executor,
then renders the real band — and `CpuIrExecutor` recompiled from scratch on
every `render()` call, so the CPU compiled twice (~1.15 s each → ~2.3 s
wall-time, dominating everything). The executor now caches the JIT engine +
function pointer on the scene structure hash, so the calibration probe warms
the cache and the real band reuses it (`compile 0.0 ms`). Compile happens
once per topology; a topology change invalidates the cache. Verified the
cached path stays bit-identical (compare merge still CONSISTENT). This makes
the weighted split's wall-time reflect the actual balanced render, not a
double compile.

---

## [4.19.0] — 2026-06-04

### Added — load-balanced (weighted) frame split

The CPU/GPU frame split worked but used equal-height bands, so the slow CPU
band dominated wall-time — a 50/50 split is actually slower than GPU-alone.
`WeightedBands` sizes each band proportionally to its executor's throughput,
so all bands finish together and the split delivers a real heterogeneous
speedup.

- **`WeightedBands(weights)`** — band i gets `H · weights[i]/Σweights` rows,
  in executor order (so `OnePathPerTile` assigns band i to executor i). Bands
  tile [0,H) exactly (rounding accumulates into the boundary; degenerate
  all-zero weights fall back to equal bands).
- **`frep_multipath --decompose weighted`** calibrates automatically: it
  renders a thin trial strip on each executor and sets weight = 1/render_ms,
  then splits by those weights. `--decompose weighted:w0,w1,…` supplies
  weights directly. Verified: a 1:25 CPU:GPU weight ratio puts ~4% of a
  100-row frame on the CPU and ~96% on the GPU — the two finish together.
- 3 tests (proportional split, exact cover, degenerate fallback). 282 total.

This is the heterogeneous-execution payoff: with the paths now visually
equivalent (mean ≈ 0.0008, seam invisible on hardware), a weighted split runs
CPU and GPU concurrently on one frame with a balanced load.

---

## [4.18.6] — 2026-06-04

### Docs — record the GPU_GLSL equivalence result

The light-colour fix (4.18.5) landed: CPU_IR vs GPU_GLSL dropped to mean
|Δ| = 0.00076 with 99.9% of pixels within tolerance — the same order as the
shared-IR floor (CPU_IR vs GPU_IR = 0.00073). The independent GLSL path,
which started at mean 0.022 / max 0.69, now matches the shared-IR path after
aligning every algorithm by hand (raymarch, smooth-union, analytic normals,
specular form, sky gradient, explicit sqrt, and finally per-light colour).
`docs/MULTIPATH.md` now records the three-way comparison and the list of
alignments as the central result: a shared IR gives equivalence by
construction, while an independent retarget can be driven to the same parity
only by laboriously re-deriving each detail. No code change.

---

## [4.18.5] — 2026-06-04

### Fixed — GLSL path now applies per-light colour (the real systematic source)

The three-way comparison was decisive: CPU_IR vs GPU_IR = 0.0007 (CONSISTENT,
100% ≤tol) on the *same* scene, while GPU_IR vs GPU_GLSL = 0.0092 on the
*same GPU*. So the residual is purely a GLSL-vs-IR algorithm difference, not
hardware — exactly as Alexander argued — and 0.0092 is far too large for a
scene this simple to be rounding noise. There had to be a concrete bug.

Found it: the scene's light is warm white, `color = [1.0, 0.97, 0.9]`. The
CPU/IR path tints each light's contribution by its colour (`contrib * shadow
* intensity * light.color[ch]`). The GLSL path **ignored light colour
entirely** — the push-constant `lights[4]` only carried xyz=position,
w=intensity, with no colour — so GLSL effectively lit every scene with a pure
white `(1,1,1)` light. For this warm light that makes the GLSL green channel
~3% and blue ~10% brighter than the CPU on every lit pixel: a systematic
per-channel offset over all shaded surfaces, immune to shadow/AO/specular
toggles (it multiplies the whole light term), which is precisely the residual
the histogram and diagnostics kept pointing at.

Added `light_colors[4]` to the push constants (ShaderPush now 232 B, still
< 256), populated from `scene.lights()[i].color`, and applied it in both the
Cook-Torrance and Blinn-Phong GLSL paths (`… * lI * lcol`), matching the CPU.
Tested on hardware.

---

## [4.18.4] — 2026-06-04

### Fixed — GLSL sphere SDF + normal use explicit sqrt, not length()/normalize()

Numerical sandbox analysis (no GPU needed) settled it: with identical inputs,
the CPU and GLSL diffuse formulas are bit-identical (max |Δ| = 0), and the
aligned raymarch loops land on identical `t` for an identical SDF. So the
residual isn't in the shading math or the loop — it's in the **inputs**, and
the only thing that differs between GPU_IR and GPU_GLSL on the *same GPU* is
how each computes them. GPU_IR shares the CPU's exact `sqrt`; GPU_GLSL used
GLSL `length()` / `normalize()`, which a GPU may lower to a reduced-precision
reciprocal-sqrt (RSQ ~22-bit). That made the sphere SDF — hence every hit
point, normal, and shaded colour — differ by a small amount on every pixel:
the systematic offset the histogram localized, and exactly the kind of gap
that is GLSL-specific rather than hardware (which is why GPU_IR didn't have
it).

The sphere SDF now uses explicit `sqrt(x*x+y*y+z*z)` and the analytic normal
uses explicit `sqrt(dot(g,g))` + divide, matching the CPU operation-for-
operation instead of `length()`/`normalize()`. Tested on hardware.

(Light-direction and half-vector normalizes still use length()/normalize();
if a residual remains they're the next candidates, but the sphere SDF is the
dominant per-pixel input for this scene.)

---

## [4.18.3] — 2026-06-04

### Added — `--no-specular` diagnostic; Fresnel ruled out as the source

The Fresnel pow→manual change (4.18.2) left the numbers byte-identical, and a
quick numerical check explains why: `pow(m,5)` vs `((m²)²·m)` differ by at
most ~3e-8 over m ∈ [0,1] — far below 1/255, so it can't move a pixel. The
Fresnel form was NOT the source. (The manual expansion is kept anyway: it's
exact and makes both paths use the identical formula.)

So the residual systematic offset (mean 0.0095, 56% of pixels in (tol, 0.05])
is elsewhere in shading. Per Alexander's reasoning, it must be algorithmic,
not hardware: GPU_GLSL diverges from CPU_IR 13× more than GPU_IR does on the
same GPU, so the gap above the ~0.0007 hardware floor is a real GLSL-vs-IR
algorithm difference. To locate it empirically instead of guessing, added a
`--no-specular` diagnostic (TracerConfig.enable_specular) that drops the
specular term on both executors identically — splitting the residual into
"specular path" vs "diffuse/ambient path" so the next fix is targeted.

---

## [4.18.2] — 2026-06-04

### Fixed — GLSL Fresnel uses manual (1-vdoth)^5 expansion, not pow()

The sky + specular fixes (4.18.0/4.18.1) cut mean |Δ| from 0.022 to 0.0095,
but a residual systematic offset remained on lit surfaces (56% of pixels in
the (tol, 0.05] bucket). Key reasoning (Alexander): if the remainder were
purely CPU-vs-GPU rounding, GPU_GLSL would diverge from CPU_IR by about as
much as GPU_IR does (≈0.0007) — same GPU, same hardware rounding. It's 13×
larger, so most of it is an *algorithmic* difference in the GLSL emit, not
hardware.

Found one: the Fresnel-Schlick term. The CPU computes `(1-vdoth)^5` as a
manual chain `((1-vdoth)^2)^2 · (1-vdoth)` (exact multiplies); the GLSL path
used `pow(1-vdoth, 5.0)`, evaluated as `exp(5·log(x))`, which rounds
differently. Fresnel feeds both the specular term and the diffuse weight
`kd = (1-F)·(1-metal)`, so this offset touched every lit pixel, not just
highlights. GLSL now uses the same manual expansion. Tested on hardware.

(The triplanar-blend `pow(n, 4.0)` in textured albedo is a separate path,
only in textured scenes, left as-is for now.)

---

## [4.18.1] — 2026-06-04

### Fixed — GLSL sky gradient now uses the same input as the CPU (NDC, not normalized dir)

Hardware observation (thank you): the sky is visibly lighter on the GPU-GLSL
path than on both IR paths, while other surfaces match — the two IR paths
group together (darker sky), GLSL stands apart. Since sky is the miss case
(no shading/normals/specular), this is pure sky-colour, and the IR paths
sharing it points to an independent GLSL difference.

The sky formula is identical (`mix(horizon, top, s)`, `s = 0.5 + 0.5·y`), but
the *input* differed: the CPU feeds the **NDC** vertical coordinate `uv_y`
(range [-1,1], pre-normalization), while GLSL fed the **normalized** ray
direction's `dir.y`. normalize() shrinks the y component, so the GLSL
gradient climbed at a different rate — a constant sky-colour offset across
the whole background, a large area, which is a big part of the constant
mean |Δ| ≈ 0.022. The primary-ray sky now uses NDC `v` (matching the CPU);
reflection rays keep the direction-based form (they have no NDC, and the CPU
compare path has no reflections). Combined with the specular-form fix
(4.18.0), this targets both halves of the systematic offset — background
(sky) and lit surfaces (specular). Tested on hardware.

---

## [4.18.0] — 2026-06-04

### Fixed — GLSL specular visibility now matches the CPU's cancelled form

The divergence histogram settled the question: the residual CPU-vs-GLSL
difference on simple_spheres was NOT edge-only — 0% of pixels exceeded 0.2
(silhouettes already agree after the raymarch alignment), but ~99% of shaded
pixels sat in the small/medium buckets. A systematic per-pixel offset on
every lit surface, immune to disabling shadows/AO — i.e. in the core BRDF.

Found it: the two paths computed the Cook-Torrance specular *visibility* term
in algebraically-identical but numerically-different forms. The CPU uses the
cancelled `D·F / (4·gv·gl)` (with `gv = ndotv·(1-k)+k`, `gl = ndotl·(1-k)+k`),
where the `ndotv·ndotl` of the geometry term cancels against the BRDF
denominator. The GLSL path used the naive `D·G·F / (4·ndotv·ndotl)` with
`G = (ndotv/gv)·(ndotl/gl)` — same value in exact arithmetic, but in float it
computes `ndotv·ndotl` in both numerator and denominator, leaving a small
rounding offset on every shaded pixel. It also clamped the firefly cap at a
different point (before vs after the ×PI factor).

GLSL now uses the identical cancelled form and clamp point as the CPU. This
targets the systematic offset the histogram localized. (`_G_Smith` is no
longer called from the Cook-Torrance path; kept defined, harmless.) Tested on
hardware (GLSL can't run in the sandbox).

---

## [4.17.2] — 2026-06-04

### Added — per-pixel divergence histogram in compare

The shading-isolation diagnostics showed shadows/AO aren't the source
(toggling them changed nothing) and that mean |Δ| ≈ 0.022 is constant —
but it's unclear whether that mean comes from a few edge pixels diverging a
lot or from a systematic offset everywhere. CompareMerge now reports a
divergence histogram: the share of pixels that are within tolerance, and in
the (tol, 0.05], (0.05, 0.2], and > 0.2 buckets (on each pixel's max-channel
diff). Edge-only divergence shows as a high ≤tol share with a thin tail;
a systematic offset shows as most pixels sitting in the small bucket. The
heat map is now also computed per pixel (max channel) rather than per
channel, matching the histogram.

---

## [4.17.1] — 2026-06-04

### Added — multipath shading-isolation diagnostics

Analytic normals cut heavy_csg from 0.69 to 0.35, but simple_spheres (0.094)
and blob (0.087) didn't budge — their residual divergence is in the shading
layer (shadows / AO / lighting), not geometry or normals (mean |Δ| ≈ 0.022
has been constant throughout, suggesting a systematic shading offset). To
locate it instead of guessing, `frep_multipath` gains diagnostic flags that
toggle shading components identically on both executors:

- `--no-shadows` — disable shadow rays
- `--no-ao` — disable ambient occlusion
- `--flat` — BlinnPhong instead of Cook-Torrance

Both `CpuIrExecutor` and `GpuGlslExecutor` now take a `TracerConfig`
override so the compare stays fair (only the toggled component differs).
Running the compare with components disabled one at a time shows how much
each contributes to the divergence, pinpointing the source before a fix.

---

## [4.17.0] — 2026-06-04

### Added — analytic normals for the GLSL path via dual-number AD

The CPU path computes normals by forward-mode AD (exact); the GLSL path used
central differences (`h = 1e-3`), which diverge most at silhouettes and CSG
edges — the dominant remaining CPU-vs-GLSL difference after the raymarch and
smooth-union fixes (heavy_csg still ≈ 0.69, simple_spheres/blob ≈ 0.09).
GLSL now computes analytic normals via dual-number AD, matching the CPU.

- **`emit_node_dual`** mirrors `emit_node` in dual arithmetic: every value is
  a `Dual {float v; vec3 g;}` (value + analytic gradient). Covers sphere,
  box, plane, union/intersection/difference, smooth-union (IQ smin with the
  correct gradient), negate, translate, scale, rotate-Y, twist-Y, taper-Y.
- The shader gains a `Dual` struct + `d_*` helpers (add/sub/mul/div, sqrt,
  sin/cos, min/max, len3, dot3, box, smin) and a `scene_sdf_grad(p)` built in
  lockstep with `scene_sdf`; the nearest object owns the normal (matching the
  min() union). `scene_normal` returns `normalize(scene_sdf_grad(p).g)`.
- **Graceful fallback:** if any object lacks a dual emitter (BendXY, mesh,
  plugin, custom-expr), the whole scene reverts to central-difference
  normals, so correctness is never lost. Verified: dual scenes emit the
  analytic path and compile via glslangValidator; non-dual scenes fall back
  and still compile. 2 tests added.

Analytic normals are also faster (3 vs 6 SDF-ish evals) and are the natural
route to fuller AD later — the approach preferred over finite differences.
Tested on hardware (GLSL can't run in the sandbox).

---

## [4.16.2] — 2026-06-04

### Fixed — GLSL smooth-union missing ×0.5 (structural geometry divergence)

The raymarch alignment (4.16.1) cut the CPU-vs-GLSL divergence from 0.69 to
0.094, but a visual comparison revealed a deeper, separate problem: the
smooth-union blob *fused differently* — a distinct crease on the CPU, a
smoother merge on the GPU. That is a geometry difference, not shading.

The GLSL IQ smin emitted `mix(b,a,h) - k*h*(1-h)`, dropping the `*0.5`
factor the CPU uses (`- k*h*(1-h)*0.5`). The GPU was blending twice as hard,
rounding off the crease. Added the `*0.5` so the GLSL smooth-union matches
the CPU formula exactly. This was a real correctness bug in the independent
GLSL path, now aligned with the shared-IR ground truth.

### Note — analytic normals for GLSL still pending

The CPU path computes normals by forward-mode AD (analytic, exact); GLSL
still uses central differences. Analytic normals are preferred (faster,
exact, and the route to full AD), so porting the AD gradient to the GLSL
emitter is the next equivalence step, tracked separately.

---

## [4.16.1] — 2026-06-04

### Changed — align GLSL raymarch loop with the CPU path (GPU-GLSL equivalence, step 1)

Chasing the CPU-IR vs GPU-GLSL divergence (max |Δ| ≈ 0.69 — a bright fringe
on object silhouettes, i.e. hit/miss disagreement on edge pixels). Two
structural differences in the GLSL raymarch loop vs the CPU PHI loop were
found and fixed:

- The march parameter `t` started at **0.0** in GLSL but **0.001** on the
  CPU. The CPU's small initial offset shifts the entire march sequence; on a
  silhouette, where hit/miss hinges on a single step, that flips edge pixels.
- The `t > max_dist` bound was checked **after** the SDF eval + hit test in
  GLSL, but at the **top** of the iteration (before the SDF eval) on the CPU.
  Same one-step shift on edges.

GLSL now starts `t` at 0.001 and checks the distance bound at the top of the
loop, matching the CPU tracer exactly. This is the most likely dominant
source of the 0.69 (binary hit/miss → large max, small mean, localized to
edges — the observed profile).

### Known remaining difference (next step if needed)

The two paths still compute **normals differently**: the CPU path uses
forward-mode AD (mathematically exact), the GLSL path uses central
differences (`h = 1e-3`, an approximation). This produces gradient-shading
differences, largest at silhouettes and on CSG/deformed surfaces. If a
residual divergence remains after the raymarch alignment, this is the next
suspect — addressed by making the two normal methods agree. Tested on
hardware (GLSL can't run in the sandbox).

---

## [4.16.0] — 2026-06-04

### Added — CompilePolicy: per-parameter constant-vs-runtime placement

Compile amortization revealed the deeper question: *which* parameters should
be runtime inputs (editable without recompiling) and which should be baked
constants (specialised, optimised). That is a point on a data↔code spectrum,
and the right point is per parameter, driven by how often it changes — the
classic compiler trade-off (specialisation vs. recompilation) carried into
the geometric domain. See `docs/ARCHITECTURE_PATHS.md` for the full model.

- **`core/compiler/compile_policy.hpp`** — `CompilePolicy` decides a
  `ParamPlacement` (Constant | Runtime) per parameter. `AllConstantPolicy`
  bakes everything (one-shot final render); `ByParamClassPolicy` is runtime
  for chosen parameter *classes* (Geometry/Material/Deform/Render/Other),
  constant for the rest. `ByParamClassPolicy::interactive()` is the editing
  default (geometry + material + deform runtime; render/observer constant).
- `FRepNode::param_value` now carries the parameter's `ParamClass`; codegen's
  slot allocator consults the policy and bakes a constant (returns no slot)
  when the policy says `Constant`. `SceneCodegen::set_compile_policy` installs
  one; the CUDA viewport path uses the interactive policy.

The decision is by class today, but the interface is built to grow toward
per-parameter and later frequency/statistics-driven *promotion* without
touching call sites. Camera and render/lighting stay constant for now (camera
is already a runtime kernel arg; render settings change rarely). Material is
*classified* for runtime, but the material emitter still bakes its values
(they come from the BVH build, not the node-param path) — migrating materials
onto the runtime buffer is a separate, deliberate step. 4 policy tests added
(277 total).

---

## [4.15.1] — 2026-06-04

### Fixed — build error: IncrementalParams forward-declared in wrong namespace

v4.15.0 forward-declared the CUDA amortization member as
`std::unique_ptr<class IncrementalParams>` *inside* `namespace frep::gui`,
so it bound to a non-existent `frep::gui::IncrementalParams` instead of the
real `frep::IncrementalParams`. With CUDA enabled (and a stricter libstdc++,
e.g. GCC 16 on the build host) the incomplete type broke the Viewport
constructor/destructor. Fixed by forward-declaring `IncrementalParams` in
the `frep` namespace and referring to `frep::IncrementalParams`. Sandbox
builds were unaffected (the member is behind FREP_HAS_CUDA), which is why it
only surfaced on the CUDA host.

---

## [4.15.0] — 2026-06-04

### Added — compile amortization for the GPU-IR (CUDA) viewport path

The offscreen-GPU (CUDA) path recompiled the whole pipeline (codegen →
NVPTX → PTX → cuModuleLoadData) on *every* parameter edit, because the
cache key was the full scene hash and parameters were baked into the IR as
constants — so dragging a slider re-JIT'd the kernel each frame. It now
amortizes like the CPU path:

- The CUDA module is keyed on the scene **structure** hash (topology), not
  the full scene hash. Parameter-only edits no longer rebuild it.
- Codegen runs in **Incremental** mode (`incremental_params`), so parameters
  are read from a runtime buffer instead of baked constants. The viewport
  captures the binding table (`IncrementalParams`) at compile time.
- Each frame refreshes the params buffer from the scene's live values and
  passes it to `CudaCtx::render` — which re-uploads it to the device and
  re-launches. No codegen / NVPTX / module reload on a parameter edit.

A topology change still triggers a full recompile (correct — the kernel
structure changed). This makes parameter dragging cheap and the render-time
measurement honest (compile once per structure, then pure render), which
matters for the heterogeneous timing comparison.

### Fixed — render-mode names + status line consistent across modes

- The Render-tab Mode combo box now uses the same full-chain names as the
  View menu ("CPU · Model→IR→JIT→CPU→offscreen", etc.) — they were stale.
- The status line now names the active path in *every* mode (CPU, GPU-IR,
  realtime), not only CUDA — e.g. "… | CPU: IR→JIT" or "… | GPU: IR→CUDA".
  Offscreen-GPU reflects a CPU fallback when CUDA is unavailable.

---

## [4.14.1] — 2026-06-03

### Fixed — status line shows the GPU backend in offscreen-GPU mode

The status line always showed the CPU-style "Render … | Total …" message,
even in the GPU-IR (CUDA) mode, because it is built from the
`render_completed` signal and didn't reflect the executor. It now appends
the active GPU path ("| GPU: CUDA/IR" or "| GPU: Vulkan/GLSL") when the
viewport is in a GPU mode, so the status line matches the selected path.

---

## [4.14.0] — 2026-06-03

### Changed — GUI offscreen-GPU mode is now the GPU-IR (CUDA) path (Part 4 of 4)

The GUI's three render modes were really only two retarget paths: offscreen
CPU (CPU-IR) and *both* GPU modes ran GLSL→Vulkan — "OffscreenGPU"
(compute + readback into a QImage) and "Realtime" (compute into the
swapchain) differed only in presentation, not in the executor. So the
offscreen-GPU slot now hosts the third, genuinely distinct path — GPU-IR
via CUDA — which shares the offscreen-readback presentation model exactly:

- **OffscreenCPU** — Model→IR→JIT→CPU raymarch→offscreen→image (unchanged)
- **OffscreenGPU** — Model→IR→PTX→CUDA→GPU raymarch→offscreen→image (**new**:
  was GLSL/Vulkan offscreen)
- **Realtime** — Model→GLSL→SPIR-V→Vulkan→swapchain→screen (unchanged)

`Viewport::set_gpu_mode(on, GpuBackend)` selects the backend; OffscreenGPU
passes `Cuda`. The CUDA and Vulkan render stages both produce a hi-res RGBA8
buffer and share the same CPU box-downsample SSAA + QImage presentation — so
they differ only in the render stage, per the path model. Falls back to CPU
when CUDA is unavailable. GLSL offscreen remains available via the tools
(`frep_multipath --paths gpu_glsl`, `frep_dump`).

Render modes are renamed to spell out the full path chain (e.g.
"GPU · Model→IR→PTX→CUDA→offscreen"), making each path's emit → executor →
presentation explicit.

### Added — path/stage architecture document

`docs/ARCHITECTURE_PATHS.md` records the conceptual model now guiding the
system: a geometric model is a program in a visual language; the system is a
compiler that retargets it (incl. shaders, raymarch, accel, and
post-process) to an executor; a *path* is the whole chain
(emit→executor→postprocess→present), not just the executor; paths are built
from composable **stages** (Emit/Render/PostProcess/Present), linear today
and a graph later; and distribution may eventually need a dynamic,
non-linear scheduler. Post-process becomes its own stage (not baked into
executors) so paths stay comparable and frame-splits seamless — the
direction the next steps build toward.

---

## [4.13.0] — 2026-06-03

### Added — per-pixel CUDA kernel for the GPU-IR path (Part 2 of 4)

Part 1 ran the GPU-IR path as a single GPU thread walking the whole tile —
correct but slow (~1.9 s for 200×150 on a GTX 1050 Ti, and CPU-vs-GPU-IR
agreed to max |Δ| ≈ 0.002, i.e. only float-rounding apart, confirming the
shared-IR retarget is numerically equivalent). Part 2 makes it parallel:

- **`SceneCodegen::emit_gpu_kernel`** emits a per-pixel kernel. It reuses
  the full raymarch by renaming the per-tile renderer to `render_tile_impl`
  (internal, always-inline) and emitting a thin `render_tile` kernel that
  reads its pixel from the NVPTX thread/block id
  (`llvm.nvvm.read.ptx.sreg.*`) and calls the impl for a 1×1 tile at that
  pixel. No tracing logic is duplicated. The kernel keeps render_tile's
  name + 21-arg ABI, so CudaCtx launches it unchanged.
- **`CudaCtx`** now launches a 2D grid (16×16 thread blocks covering the
  tile) instead of one thread — each GPU thread renders one pixel.

So a W×H frame runs as W×H parallel threads. Verified in-sandbox: the
kernel PTX contains `.entry render_tile` and reads `%tid.x` (genuinely
per-pixel). Three NVPTX generation tests added (273 total). The on-GPU
speedup is measured on hardware.

This completes the heterogeneous-execution core: three real, parallel
retarget paths — CPU-IR (JIT), GPU-IR (CUDA/PTX), GPU-GLSL (Vulkan).
Part 4 (repointing the GUI's offscreen-GPU mode to GPU-IR) remains.

---

## [4.12.0] — 2026-06-03

### Changed — GPU-IR path retargeted to CUDA/NVPTX (NVIDIA-native)

Hardware testing showed NVIDIA's OpenCL driver rejects SPIR-V
(`clCreateProgramWithIL` → CL_INVALID_OPERATION): NVIDIA doesn't support
`cl_khr_il_program`. SPIR-V generation itself was correct (valid OpenCL
SPIR-V, confirmed) — NVIDIA simply can't ingest it. The native NVIDIA route
for running LLVM IR on the GPU is CUDA/PTX, so the GPU-IR executor now uses
that:

- **`NVPTXRetarget`** (`core/compiler/retarget_nvptx.hpp`) lowers the IR to
  PTX via LLVM's in-tree NVPTX backend — marking `render_tile` as a CUDA
  kernel (PTX_Kernel calling convention + the `nvvm.annotations` "kernel"
  entry) and emitting PTX text through TargetMachine. Verified in-sandbox:
  31 KB of PTX with `.entry render_tile`, target sm_50.
- **`CudaCtx`** (`core/gpu/cuda_ctx.{hpp,cpp}`) runs it through the CUDA
  Driver API: `cuModuleLoadData` JITs the PTX to SASS for the device,
  `cuLaunchKernel` runs `render_tile`. We declare the slice of the driver
  API we use, so it builds without the CUDA toolkit headers — only
  libcuda.so (shipped with the driver). Gated behind `FREP_BUILD_CUDA`.
- **`GpuIrExecutor`** now retargets to PTX and runs on CUDA, caching the
  context (PTX JIT is expensive) keyed on the IR. It compiles to a clean
  no-op (`available()` false) when CUDA isn't built, so non-NVIDIA builds
  are unaffected.

This keeps the intended structure — CPU-IR and GPU-IR share the IR source
(JIT for CPU, PTX for GPU), GPU-GLSL is independent — now realised natively
on NVIDIA. The SPIR-V retarget (v4.11.1) is retained for artifact analysis
and Intel/AMD OpenCL. Part 1 launches a single thread over the tile
(correctness first); Part 2 adds a per-pixel grid for parallelism.

---

## [4.11.1] — 2026-06-03

### Fixed — GPU-IR retarget used the in-tree LLVM SPIR-V backend

On hardware the GPU-IR path failed with "retarget produced no SPIR-V
(llvm-spirv missing?)" — the Khronos `llvm-spirv` translator isn't
installed (and is a fragile external dependency). LLVM has had a built-in
SPIR-V backend since ~18, reachable through `TargetMachine` for the
`spirv64-unknown-unknown` triple, with no external tool. `SPIRVRetarget`
now uses that backend as the primary path (falling back to the external
translator only if the backend isn't in the build), and `GpuIrExecutor`
initialises the SPIR-V target once per process. Verified in-sandbox: the
retarget now emits valid SPIR-V (magic 0x07230203) from the render_tile IR
with no external tool. This also makes `frep_dump`'s SPIR-V output and the
SPIRVRetarget tests work without `llvm-spirv` present.

The next hardware unknown is whether NVIDIA's OpenCL accepts this SPIR-V
via `clCreateProgramWithIL` — that's the remaining Part-1 validation step.

---

## [4.11.0] — 2026-06-03

### Added — GPU-IR path made executable via OpenCL (Parts 1 & 3 of 4)

Until now the GPU-IR retargeting path (codegen → LLVM IR → llvm-spirv →
SPIR-V) only *validated* its SPIR-V; it never ran on the GPU. (The GUI's
"OffscreenGPU" mode was in fact the GLSL→Vulkan path, not GPU-IR.) This
makes GPU-IR a genuinely executable path, giving the intended structure:

- **CPU-IR** and **GPU-IR** share the IR source (codegen → same
  `render_tile`), then diverge — JIT to native for the CPU, llvm-spirv →
  SPIR-V → OpenCL for the GPU. One IR, two retarget targets.
- **GPU-GLSL** is fully independent — codegen → GLSL text → SPIR-V →
  Vulkan, never touching the IR.

Three paths, three technologies, two of which share a root — exactly the
heterogeneous-retarget structure the PoC aims to demonstrate.

**`core/gpu/opencl_ctx.{hpp,cpp}`** — `OpenClCtx`, the OpenCL runtime mirror
of `VulkanCtx`: builds a program from in-memory SPIR-V via
`clCreateProgramWithIL`, runs the `render_tile` kernel (whose C signature
OpenCL accepts directly — the same one the CPU JIT uses), and reads back an
RGBA frame. `available()` reports whether an OpenCL GPU device exists.
Bundled Khronos headers (`third_party/CL`) and a soname-based loader lookup
let it build without the OpenCL dev package; gated behind `FREP_BUILD_OPENCL`.

**`GpuIrExecutor`** (`core/exec/gpu_ir_executor.hpp`) wraps the path behind
`IExecutor`, caching the OpenCL context (keyed by the emitted IR) across
tiles. `frep_multipath --paths …,gpu_ir` selects it.

This is Part 1 (runtime) + Part 3 (executor) of the 4-part plan. The kernel
currently runs as a **single work-item** over the tile (correctness first);
Part 2 switches codegen to a per-pixel kernel + NDRange launch for real GPU
parallelism. Part 4 repoints the GUI's offscreen-GPU mode to GPU-IR. Needs
a real OpenCL GPU device (validated on hardware); skipped cleanly otherwise.

---

## [4.10.1] — 2026-06-03

### Added — compare diff heat map + worst-pixel localisation

Hardware comparison of the CPU-IR and GPU-GLSL paths on a heavy CSG scene
came back DIVERGENT (max |Δ| ≈ 0.69, mean ≈ 0.022) — a small mean with a
large max, i.e. most of the frame agrees but specific pixels (likely
silhouette edges) diverge sharply. To locate it:

- `CompareMerge` now produces a per-pixel `diff_image` heat map (red ∝ the
  worst channel difference at that pixel), so where the paths disagree
  lights up, and records the `(worst_x, worst_y)` pixel of the maximum
  difference, now printed in the report.
- `frep_multipath` writes the heat map next to `--out` (e.g. `cmp.ppm` →
  `cmp.diff.ppm`) or to an explicit `--diff FILE.ppm`.

This is the diagnostic for driving the CPU/GPU paths to visual equivalence
— the prerequisite for a seamless frame split — and an artifact for the
write-up. No behaviour change to rendering.

---

## [4.10.0] — 2026-06-02

### Added — GPU tile rendering, GPU executor, and the `frep_multipath` driver

Continues Model D (2a) toward real heterogeneous execution.

**Both retargeting paths now render a sub-region.** Investigation found
neither GPU path rendered tiles (both did the whole frame); the CPU JIT
path, by contrast, was *already* tile-addressed (`render_tile` takes tile
offset + extent) but only ever covered the full frame. So:

- `RenderParams` gains an optional `region` — the tiled scheduler then runs
  only the tiles inside it, making a CPU sub-region cost proportional to
  its area (verified: a half-frame renders in ~half the time, not via
  cropping a full render). `CpuIrExecutor` uses this.
- `ShaderPush` gains `tile_x0/y0/x1/y1`; the GLSL emitter offsets the
  pixel by the tile origin (computing the ray from the full frame so the
  region matches a whole-frame render exactly) and the Vulkan dispatch is
  sized to the tile. Whole-frame is the default (tile_x1/y1 == 0), so
  existing callers are unaffected. One change covers GLSL and IR→SPIR-V
  paths since they share the push constants + shader entry.

**`GpuGlslExecutor`** wraps the GLSL path behind `IExecutor` with tile
support; the Vulkan context is built once and cached across tiles. It is
available only with a real Vulkan device and is skipped cleanly otherwise.

**`frep_multipath`** drives the framework from the CLI: pick paths,
decompose/dispatch/merge strategies, and run mode; it writes the merged
image (PPM) and a report, and exits non-zero when a `compare` run finds
divergence (scriptable equivalence check). Documented in docs/MULTIPATH.md.

Tests now cover GPU-executor availability consistency and exact
region-vs-full-frame agreement (269 total). GPU-path concurrent execution
and CPU-vs-GPU divergence measurement are validated on real hardware.

---

## [4.9.0] — 2026-06-02

### Added — Model D: multi-path execution framework (`core/exec/`)

The foundation for heterogeneous concurrent execution — using all the
available compute hardware for one render regardless of its architecture
or control language. Rather than hardcode "CPU + GPU frame split", the
framework factors a run into three orthogonal strategy axes, each an
abstract base class with virtual methods, so built-in behaviours are
named subclasses and users can add their own — and the three axes compose
(N decompose × M dispatch × K merge) instead of needing a concrete class
per combination:

- **decompose** (`IDecomposeStrategy`) — split the output into rectangular
  `Tile`s: `WholeFrame`, `HorizontalBands(n)`, `GridDecompose(w,h)`.
- **dispatch** (`IDispatchStrategy`) — assign tiles to executors:
  `OnePathPerTile` (round-robin, for splitting/distribution),
  `AllPathsPerTile` (for cross-path compare).
- **merge** (`IMergeStrategy`) — recombine results: `StitchMerge` (one
  image; last writer wins), `CompareMerge` (per-channel max/mean diff +
  consistency verdict).

`MultiPathExecutor` runs decompose → dispatch → execute → merge, with
jobs run `Concurrent` (one std::async per job) or `Serial`. An `IExecutor`
abstracts a retargeting path; `CpuIrExecutor` (codegen → IR → JIT →
tiled scheduler) is the first, always-available implementation. GPU
executors (GLSL and IR→SPIR-V paths) need a real Vulkan device and follow.

This expresses every model as a config: whole-frame + all-paths + compare
= cross-path equivalence check; two bands + one-path-each + stitch =
CPU/GPU frame split; small grid + remote paths + stitch = distributed
render (future). Crucially, **compare is the diagnostic that makes stitch
sound** — visual equivalence across paths is a prerequisite for a seamless
split, so the same framework that splits also measures the divergence that
would break it.

Nine tests cover the strategies, concurrent-vs-serial equivalence, the
exact agreement of two same-path runs (the equivalence floor), seamless
half-frame stitching, and skipping of unavailable paths. GPU-path
executors and a CLI driver come next.

---

## [4.8.2] — 2026-06-02

### Fixed — `frep_dump` failed to compile on LLVM 22

The native-code-size path used the LLVM 21-and-earlier signatures of
`Module::setTargetTriple`, `TargetRegistry::lookupTarget`, and
`Target::createTargetMachine`, all of which LLVM 22 changed to take a
`Triple` object rather than a triple string (PR #130940). On the LLVM 22
build this surfaced as `no viable conversion from 'std::string' to
'Triple'`. Routed `setTargetTriple` through the existing
`llvm_compat::set_target_triple` shim and guarded the `lookupTarget` /
`createTargetMachine` calls on `LLVM_VERSION_MAJOR >= 22`, so the tool
builds on both the project's LLVM 22 target and the LLVM 20 fallback.
`addPassesToEmitFile` + `CodeGenFileType::ObjectFile` are unchanged across
these versions and needed no guard.

---

## [4.8.1] — 2026-06-02

### Added — `frep_dump` analysis: code sizes, parallelism, GPU breakdown, render

Extended the dump tool with the remaining analysis dimensions:

- **Native machine-code size** (`native_text_bytes`) — the optimised
  module is emitted to an in-memory object file and its `.text` measured.
  This is the real CPU executable footprint, the analogue of SPIR-V size,
  enabling a direct CPU-vs-GPU code-size comparison for the same scene
  (e.g. ~119 KB `.text` vs ~184 KB SPIR-V for the CSG benchmark).
- **Parallelism scaling** (`thread_scaling`) — render time at 1, 2, 4, …
  up to hardware threads, showing how the tiled CPU renderer scales across
  cores (groundwork for the heterogeneous-execution work).
- **GPU init breakdown** — when a real Vulkan device is present
  (`ran_on_device`), records the per-phase init timing (device / shader /
  pipeline-compile / buffers / misc) and device-memory estimate, matching
  the benchmark's breakdown. Skipped gracefully without a device.
- **Rendered frame** — `<base>.render.ppm` of the final image as a visual
  reference alongside the code artifacts.

docs/BENCHMARKS.md §6 updated.

---

## [4.8.0] — 2026-06-02

### Added — `frep_dump`: artifact & analysis dump for the retargeting pipeline

A CLI tool that emits the intermediate artifacts of both retargeting
paths for a scene, for system study and write-ups, without instrumenting
the renderer. `frep_dump <scene.json> [basename] [options]` writes:

- `<base>.pre.ll` / `.post.ll` — LLVM IR before and after the O3 pipeline,
  so the optimisability of the generated code is directly inspectable. The
  stats include pre/post instruction counts and their ratio: inlined
  scenes expand to ~170% (min() chains unroll + vectorise), guarded scenes
  stay near 100% (per-object functions kept separate) — a concrete measure
  of how the two SDF-emission strategies differ under optimisation.
- `<base>.glsl` / `.spv` — generated GLSL compute shader and its SPIR-V
  (via `glslangValidator`; skipped gracefully if absent).
- `<base>.scene.json` — normalised scene dump.
- `<base>.stats.json` — structured metrics: CPU codegen/optimise/JIT/render
  times, IR sizes pre/post-opt, GPU GLSL/SPIR-V sizes and times, peak/delta
  RSS, hardware thread count, and the explicitly-chosen SDF mode with its
  reason (so test results aren't misread).

Options: `--width/--height`, `--no-guards` (force Inlined), `--no-spirv`,
`--no-render`. Side-by-side and pixel-diff analysis is left to external
tools operating on the dumps. Documented in docs/BENCHMARKS.md §6.

This is the first piece of the planned analysis tooling; it also lays the
measurement groundwork for the upcoming concurrent/heterogeneous execution
experiments (timing and result-comparison across paths).

---

## [4.7.7] — 2026-06-02

### Added — Benchmark example scenes (`examples/benchmark_*.json`)

Eight ready-made scenes for manual performance testing and for exercising
the adaptive spatial-guard heuristic, so testers don't have to hand-build
complex scenes. Generated through the SceneGraph API (schema-valid by
construction) by a new `frep_gen_benchmark_scenes` tool, and verified to
load and render end-to-end through the IncrementalCompiler.

Simple family (cheap ~2-node objects, stay Inlined): spheres_27,
boxes_64, spheres_125. Heavy family (expensive ~4–6-node objects, switch
to Guarded once calibrated): twist_27/64/125 (twist + smooth-union),
csg_48 (box-minus-sphere difference), and mixed_48 (half cheap/half heavy,
exercises the average-cost path). Each scene's object count and average
node count are chosen so it lands deterministically in its intended
heuristic branch — confirmed: simple → Inlined, heavy/csg/mixed → Guarded.

Documented in `examples/README.md` with regeneration instructions.

---

## [4.7.6] — 2026-06-02

### Changed — Adaptive spatial guards on by default, with a GUI toggle

Now that the calibration is validated on hardware (threshold 3 on the
test machine — bare spheres stay inlined, box-and-up get guarded),
`IncrementalCompiler` enables spatial guards by default. The choice stays
fully automatic and per-recompile: the calibration is conservative (its
threshold is always above a bare primitive), so simple scenes are never
guarded and the inlined path remains their behaviour — guarding only kicks
in for the CSG/deform-heavy scenes it measurably helps (3–6.5×).

A **Render → Adaptive spatial guards** menu toggle (checked by default)
lets the user turn it off. It's wired through `IViewport::set_spatial_
guards_enabled` to the offscreen-CPU compiler (other backends no-op it —
the GPU path doesn't use the JIT scene_sdf).

To keep trivial scenes free of any cost, `choose_sdf_mode` pre-screens
cheaply (object count + max node count) and skips the one-time calibration
entirely unless the scene could actually benefit (≥8 objects and at least
one object with ≥3 nodes). So the default project and simple edits never
trigger a measurement; it happens lazily only when a genuinely heavy scene
is first compiled.

---

## [4.7.5] — 2026-06-02

### Fixed — Calibration was fooled by measurement noise

The first hardware calibration returned a threshold of 2 — which would
guard bare translated spheres (node count 2), exactly the case the
hardware data showed guarding *hurts*. The probe was too light (24
objects, 96×96): render times were low enough that a few-percent "win"
was just jitter, and the bare-sphere level could trip the ×0.95 test.

Reworked for robustness: larger probe (32 objects, 128×128) so the signal
clears noise; a stricter win margin (guarded must be ≥15% faster, not 5%);
and the threshold is now required to be strictly above the bare-sphere
node count, so jitter on the simple-primitive case can never enable
guarding for simple scenes. The sweep still exits at the first winning
complexity, so it never renders the expensive high-complexity probes. In
the sandbox this yields a stable threshold of 3 (box and up guarded,
bare spheres not); on real hardware it should land similarly and well
above 2. `CalibrateProducesValidResult` now asserts the ≥3 property.

---

## [4.7.4] — 2026-06-02

### Added — Runtime calibration + adaptive selection of spatial guards

The hardware data showed the guard's payoff hinges on per-object SDF cost
(0.9× for bare spheres, but 3–6.5× for twisted-box/smooth-union objects),
and the crossover is CPU-dependent. Rather than hardcode a threshold, the
crossover is now *measured on the host* and the guard turned on adaptively.

`core/accel/guard_calibration.{hpp,cpp}` renders a few tiny inlined-vs-
guarded scenes (24 objects, 96×96) at increasing per-object complexity and
records the lowest node count where guarding wins. The result is cached to
`$XDG_CACHE_HOME`/`~/.cache/frep_guard_calibration.txt`, keyed by CPU
model; later runs load it instantly, and a different CPU (moved machine)
triggers automatic recalibration. Measurement is fast (~200 ms) — suitable
for first-run, with explicit recalibration on demand.

`node_count()` (core/frep/node.hpp) is the cheap per-object cost proxy.
`should_guard()` enables the guarded path only when a scene has enough
objects (≥8) AND their average node count meets the calibrated threshold.
`IncrementalCompiler` consumes this: `set_spatial_guards_enabled(true)`
makes it fetch/cache the calibration and pick Guarded vs Inlined per
recompile (`last_sdf_mode()` reports the choice). Off by default — the
inlined baseline stays the safe default; guarding is opt-in for the
heavy-object niche it helps.

The benchmark gained `--calibrate` / `--recalibrate` to inspect or refresh
the cached threshold. Seven tests cover node_count, the should_guard
logic, the cache round-trip, CPU-mismatch invalidation, and that
calibrate() yields a valid result.

### Fixed

- `save_calibration` silently failed when the cache directory (e.g. a
  fresh `~/.cache`) didn't exist — `std::ofstream` doesn't create parent
  directories. It now creates them first, so the calibration actually
  persists and the second run loads from cache instead of re-measuring.

---

## [4.7.3] — 2026-06-01

### Changed — Spatial guards measured on hardware: a niche, not a default

Ran the guarded SDF on real hardware (x86 NUC), which corrected the
sandbox picture. Guarding **bare spheres does not help** there (0.7× /
1.0× / 0.9× at 11 / 101 / 1001 objects): the inline `min()` chain
vectorises and is branchless, while the guard adds a data-dependent branch
and a `sqrt` per object that, for a near-free sphere SDF, costs more than
it saves. The sandbox's 1.7–1.9× was an artefact of software rendering
(no SIMD/branch-prediction).

The guard's payoff depends entirely on per-object SDF cost. The benchmark
gained `--heavy` (with `--func-split`): objects become twisted-box +
smooth-union-sphere blends — expensive to evaluate. There the result
inverts to **2.2× at just 9 objects**, where bare spheres gave 0.7×.

Conclusion: build-time spatial guards hurt simple-primitive scenes and
help CSG/deform-heavy ones, so they belong behind an adaptive heuristic
(by average per-object cost), never unconditionally. `SceneSdfMode::
Guarded` stays available for that niche; the default is unchanged
(inlined). For simple primitives at scale the vectorised inline `min()` is
hard to beat on the CPU, and the GPU remains the answer for large counts.
Documented in docs/PERFORMANCE.md §6.

---

## [4.7.2] — 2026-06-01

### Added — Build-time spatial guards in the JIT (BVH integration, approach 1)

First integration of the spatial acceleration into the actual renderer.
`emit_scene_sdf_guarded` materialises the flat AABB prune directly in the
generated code: each object becomes a non-inlined function gated by an
inline AABB-distance test against the running best distance, so an object
is evaluated only when it could be nearer than the closest surface found
so far. No runtime hierarchy or stack, so it works on the JIT path now and
ports to GLSL later. Exposed via `emit_render_tile(scene, SceneSdfMode)`
(Inlined / Split / Guarded) and the benchmark's `--func-split` mode, which
now compares all three.

Measured on the JIT path (full render, spread spheres, AD-gradient still
inlined so it's a lower bound): 1.7× at 100 objects, 1.9× at 300, growing
with object count. Verified identical to the inlined min() by
`JitCodegen.GuardedSceneSdfMatchesInline`.

Still experimental (behind the benchmark flag, not the default): making it
default needs the AD-gradient SDF guarded too and a heuristic for when to
enable it (it costs slightly below ~10 objects). Documented in
docs/PERFORMANCE.md §6.

---

## [4.7.1] — 2026-06-01

### Added — SDF invariant contract + enforcement tests

Documented the distance-field contract every `FRepNode` guarantees —
positive-outside oriented distance, approximately 1-Lipschitz — directly
on the class, with the rationale that sphere-tracing, `min()`/`max()`
composition, and BVH pruning all silently misbehave if it's violated. The
agreed policy is that externally-sourced geometry (CustomExpr, Plugin,
MeshSDF) must be normalised to this convention at import time rather than
guarded for in the render/BVH hot paths.

`tests/test_sdf_invariant.cpp` enforces it on every built-in: 1-Lipschitz
over thousands of random point pairs, the positive-outside/negative-inside
sign convention, and the box's exact Euclidean corner distance (the case
the recently-fixed Chebyshev metric got wrong). Nine tests guard against
future metric regressions in any primitive, operation, transform, or
deformation.

---

## [4.7.0] — 2026-06-01

### Added — Scene bounding-volume hierarchy (`core/accel/bvh.hpp`)

A host-side BVH over scene objects that accelerates the nearest-distance
query underlying SDF sphere-tracing. Sphere-tracing needs the true
nearest distance, so objects can't be culled the way ray-triangle tracing
does; instead the BVH uses the fact that the distance from the query point
to an object's AABB is a *lower bound* on the distance to its surface, and
prunes any subtree whose box is farther than the best distance found so
far (visiting the nearer child first). The result is identical to a
brute-force `min()` over all objects — verified by 8 tests across
primitive types, spread and overlapping layouts, unbounded planes, and
CSG units — but scales roughly logarithmically: a prototype measured ~80×
over brute force at 1000 spread objects.

Pruning is correctly disabled for interior (negative-distance) queries,
where an AABB distance (always ≥ 0) is no longer a valid bound. The
flattened node layout is designed to be uploadable to a GPU storage
buffer for a future shader-side traversal.

This is the acceleration structure itself; wiring it into the JIT and GPU
render paths is the next step (see docs/PERFORMANCE.md §6).

### Fixed — `BoxNode` distance was the Chebyshev, not Euclidean, metric

`BoxNode`'s CPU eval and JIT codegen returned `max(dx,dy,dz)` (the L∞
distance), which is correct inside and on-axis but under-estimates the
distance to a far corner. That under-estimate made sphere-tracing take
unnecessarily small steps near boxes and disagreed with the GPU path
(whose GLSL already used the correct form). Fixed to the true Euclidean
box SDF `length(max(d,0)) + min(max(d),0)` in eval and codegen, matching
GLSL; CPU and GPU box rendering now agree, and the metric satisfies the
lower-bound invariant the BVH relies on.

---

## [4.6.6] — 2026-06-01

### Changed — Performance investigation concluded

The function-split diagnostic ran on hardware and closed the
investigation into the CPU JIT's super-linear cost at scale. At 1000
objects the split compiled no faster than the inlined path (23.6 s vs
23.2 s — within noise) and rendered 3× slower, so one giant function is
**not** the cause. Combined with the earlier opt-level sweep (O1≈O3),
this rules out every codegen-structure fix: the cost is the irreducible
lowering of ~10 000+ instructions to machine code, however partitioned.

`docs/PERFORMANCE.md` §4 now documents the conclusion and the resulting
guidance: the CPU JIT is for interactive editing of small/medium scenes
(≤~100 objects, sub-second compile); large scenes (hundreds–thousands of
objects) should use **GPU mode**, which the same hardware run shows
handles mixed-1000 in ~0.9 s compile + ~6 s render with ~1 MB device
memory. The unified render-mode selector (4.3.0) makes the switch
one click.

### Fixed

- The `--func-split` header printed garbage width×height (missing printf
  arguments); now correctly shows the render resolution.

---

## [4.6.5] — 2026-06-01

### Added — Function-split JIT diagnostic (`frep_bench --func-split`)

The opt-sweep ruled out the optimisation level as the cause of the
super-linear CPU JIT cost (O1 vs O3 differ by <3% at 1000 objects), which
points at the structure: one enormous inlined function. To test that
directly, `SceneCodegen` gained `emit_scene_sdf_split()` — it emits each
object as its own non-inlined `obj_sdf_N` function and folds them with
`min()` in `scene_sdf`, instead of inlining the whole tree — and
`emit_render_tile(scene, split_objects=true)` wires it into the march
loop. The benchmark's `--func-split` mode compiles the full render_tile
both ways at growing object counts and compares JIT-compile and render
time.

This is a diagnostic, not a shipped change: the default path is unchanged
(inlined). Early build-environment numbers suggest the split is *slower*
to compile and render at small/medium sizes; the open question is whether
it crosses over and wins at the ~1000-object scale where the inlined path
explodes, which needs real-hardware measurement. `JitCodegen.SplitScene
SdfMatchesInline` verifies the split produces an identical distance field.

---

## [4.6.4] — 2026-06-01

### Added — CPU JIT optimisation-level sweep (`frep_bench --opt-sweep`)

The init-breakdown run cleared the GPU of any compile blow-up (driver
pipeline compile stayed ≤16 ms; the earlier multi-minute init was a
one-off cold shader-cache artefact). That leaves the **CPU JIT** as the
sole bottleneck at scale — LLVM's O3 pipeline lowering one giant unrolled
function (~23 s at 1000 objects, ~2.4 min for mixed-1000; codegen stays
trivial).

`JitEngine` now exposes `set_opt_level()` (default O3, unchanged), and the
benchmark gained `--opt-sweep`: for growing object counts it compiles and
renders at O0/O1/O2/O3 and tabulates JIT-compile vs per-frame-render time
for each, so the compile/render trade-off — and any crossover where a
lower level wins on total time — is measurable on real hardware. Works
with `--smoke` for a fast check.

`JitCodegen.OptLevelsAgree` verifies every level produces identical SDF
results (only compile time/codegen quality changes, never the math).

---

## [4.6.3] — 2026-06-01

### Added — GPU init phase breakdown (diagnostics)

Hardware scaling runs showed GPU `init` time exploding on large scenes
(≈21 s at 1000 objects, ~23 min for the mixed-1000 scene), while the
computed device memory stayed ~1 MB — pointing at the driver compiling a
huge unrolled shader rather than any allocation. To confirm this,
`GpuRenderStats` and the benchmark now split init into its phases:
device setup, shader-module ingest, **pipeline creation (the driver's
SPIR-V→native compile)**, buffer upload, and misc. `frep_bench --scaling`
prints a per-axis "GPU init breakdown" sub-table so the pipeline-compile
cost is isolated directly.

`docs/PERFORMANCE.md` §5 now records the hardware findings: at ~1000
objects the wall is **compilation** (CPU LLVM O3 JIT, and the GPU driver
pipeline compile), not per-frame render (GPU render beats CPU 6–24× and
both stay tractable). This revises the earlier "BVH is highest-impact"
note — the higher-leverage change is to iterate objects in a loop over a
per-object parameter buffer instead of unrolling N objects into one giant
function, making shader/IR size independent of N.

---

## [4.6.2] — 2026-06-01

### Added — Memory measurement + compile/execute split in the benchmark

The scalability suite (`frep_bench --scaling`) now reports memory and a
finer timing breakdown, so the curves show not just how fast but how
*large* a scene each back-end can hold:

- **Memory.** CPU rows report ΔRSS (working-set growth for the scene)
  and process peak RSS; GPU rows report host-side ΔRSS plus a computed
  device-memory (VRAM) footprint for the output image and any
  voxel/texture buffers (device allocations don't show in RSS). This
  bounds the practical scene-size ceiling per back-end. Measured CPU
  memory grows with object count — ~70 MB JIT floor, ~140 MB by 500
  objects — because the JIT module itself grows, not the framebuffer.
- **Compile vs execute.** CPU timing is split into `codegen` (build LLVM
  IR) and `JIT` (lower to native) — together the one-time compile — vs
  per-frame `render`. GPU timing keeps its emit / compile / init / render
  split. The split separates the once-per-edit compile (incremental in
  the live app) from the per-frame render cost.

Documented in `docs/PERFORMANCE.md` §5.

---

## [4.6.1] — 2026-05-31

### Added — Scalability benchmark suite (`frep_bench --scaling`)

The benchmark tool gained a standalone scalability sweep that charts
render time as scene complexity grows along four independent axes, each
driven by a single size parameter so any N is reachable:

- **Object count** — N spheres on a grid
- **Node depth** — one primitive in N nested transforms
- **CSG depth** — N chained boolean ops
- **Mixed primitives** — N objects cycling every primitive/deform

New CLI options: `--scaling` (full sweep, CPU + offscreen GPU when
hardware Vulkan is present), `--cpu-only` (skip GPU rows), `--smoke`
(tiny sizes, ~1-minute sanity check that every generator builds and
renders), and `--help`. The suite runs standalone on real hardware and
reports CPU JIT and GPU timings side by side per scene/size.

Measured curves and their interpretation are documented in
`docs/PERFORMANCE.md` §5. Headline finding: object count scales
**linearly** because `scene_sdf` is a flat O(N) `min()` over all objects
with no spatial acceleration — a bounding-volume hierarchy is the single
highest-impact future optimisation for large scenes (the per-node AABB
infrastructure already exists).

---

## [4.6.0] — 2026-05-31

### Added — Adaptive raymarch step (performance)

Profiling showed the conservative raymarch step (`safety_factor = 0.85`,
added in 4.2.5 for CSG correctness) was the dominant render cost — ~21%
over a full step — because it slows *every* ray, while the step-cap
raise to 192 (4.2.11) cost only ~7%. But the reduced step is only needed
where the distance field isn't a true Euclidean SDF: CSG, plugin, mesh,
custom-expression nodes, or more than one object.

Both renderers now detect this per-compile and march at the **full step
(1.0) for single-primitive scenes** — a true SDF where a full step never
oversteps — falling back to 0.85 otherwise. The grazing-ray rescue still
backstops both, so there are no new artefacts. Single-primitive scenes
(common while modelling one shape) render ~20% faster; CSG scenes are
unchanged.

A new `node_requires_safety_step()` helper walks the tree for
distance-non-preserving node kinds; `GlslEmitter.AdaptiveSafetyFactor`
guards the behaviour.

### Added — Performance & memory analysis (docs/PERFORMANCE.md)

Documents where render time goes (safety_factor vs max_steps vs
shadows), the adaptive-step win, the memory profile (≈74 MB JIT floor +
linear per-pixel growth, no leaks), the SSAA scratch-buffer hotspot, and
remaining opportunities (shadow-march cost, RGBA8 SSAA scratch, adaptive
epsilon).

---

## [4.5.1] — 2026-05-31

### Fixed — Swapped red/blue channels in real-time GPU screenshots

The true-GPU screenshot added in 4.5.0 saved with blue and red channels
swapped (images looked BGR). `QVulkanWindow::grab()` returns the bytes
in the swapchain's memory order — our preferred format is
`VK_FORMAT_B8G8R8A8_UNORM` (BGRA) — but tags the QImage as RGBA, so the
two channels came out transposed. capture_image() now checks the live
`colorFormat()` and applies `rgbSwapped()` when the swapchain is a
B-first (BGRA) format, leaving R-first formats untouched.

### Note — GPU MeshSDF & CustomExpr already complete

Planned as remaining work, but an audit confirmed both are already
implemented and tested: MeshSDFNode emits a trilinearly-sampled voxel
grid in a Vulkan storage buffer, and CustomExprNode emits GLSL via its
`emit_glsl`/`emit_glsl_ast` path (exercised by GpuMeshSdf.* and
MixedNodes.*/ExprBackendsTest.* tests). No changes were needed.

---

## [4.5.0] — 2026-05-31

### Added — Centralised camera-control config + mouse-sensitivity setting (Stage 6)

The orbit-camera tuning constants — mouse sensitivity, zoom step, min/max
orbit distance, pitch clamp, default distance — were duplicated as magic
numbers in both the offscreen and real-time viewports. They now live in a
single shared `CameraControlConfig` struct (`gui/camera_control_config.hpp`),
so the two paths behave identically and there's one place to retune the
feel. **Mouse sensitivity** is surfaced as a Render-tab spin box (the one
constant that genuinely differs between a mouse and a trackpad); the rest
keep sensible defaults. The setting is applied live and preserved across
render-mode switches.

### Changed — True GPU framebuffer grab for real-time screenshots (Stage 6)

Exporting an image from the real-time viewport previously used
`QScreen::grabWindow`, a screen-coordinate capture that could pick up
overlapping windows or compositor colour adjustments. It now uses
`QVulkanWindow::grab()`, which blits the actual rendered swapchain image
into host memory (an internal `vkCmdCopyImage` + readback) — a faithful
GPU-side capture. Export is unified behind a single `IViewport::capture_image()`
so all three backends (offscreen CPU, offscreen GPU, real-time) export
through the same path.

---

## [4.4.0] — 2026-05-31

### Added — Rotation & uniform-scale gizmo (Stage 4)

The numeric transform gizmo, previously translation-only, now also
covers **Y-axis rotation** (shown in degrees, stored in radians) and
**uniform scale**, as two new fields in the inspector's Properties
panel. Single-selection only, like translation.

Internally the three transforms are stored as nested wrapper nodes in a
canonical **T·R·S order** — `Translate(RotateY(Scale(geometry)))` —
regardless of the order the user edits them. Each setter locates its own
node type in the chain and updates it in place, inserts it at the right
depth if absent, or unwraps it when the value returns to identity (scale
1.0 / rotation 0). Repeated edits never nest duplicate wrappers, and the
existing translation gizmo continues to sit outermost. Both renderers
already emit `Scale` and `RotateY`, so rotation/scale work on CPU JIT,
offscreen GPU, and real-time alike, and round-trip through scene
save/load automatically.

New undo commands `SetRotationCommand` and `SetScaleCommand` make each
edit individually undoable.

### Added — Coalesced multi-object material undo (Stage 5)

Editing a material while several objects are selected broadcasts the
change to all of them. Previously a single Ctrl+Z reverted only the
primary object; the rest stayed changed. The editor now captures a
per-object baseline for the whole selection and commits one
`SetMaterialsCommand` covering every object that actually changed, so a
single undo reverts the entire multi-object edit (and redo re-applies
it). Objects whose material was unaffected are skipped, via a new
`Material::operator==`.

### Note — Stages 2 & 3 already present

Audit during this work confirmed GPU procedural patterns
(checker/stripes/gradient/noise + triplanar texture) and real-time
object picking (via the CPU `ScenePicker`, independent of the render
backend) were already implemented and tested in earlier work; no
changes were needed.

---

## [4.3.0] — 2026-05-31

### Added — Unified, runtime-switchable render-mode selector (Stage 1)

The three rendering backends — **Offscreen CPU (JIT)**, **Offscreen GPU
(Vulkan)**, and **Real-time (Vulkan window)** — are now chosen from a
single selector available both as a "Mode:" combobox at the top of the
Render tab and as a "Render mode" submenu under the Render menu. The two
stay in sync.

Previously mode selection was fragmented: real-time was a startup-only
`--realtime` CLI flag, while offscreen CPU↔GPU was a separate "Use GPU"
menu checkbox. Now all three are one runtime choice. The `--realtime`
flag is kept but only sets the *initial* mode.

**Recreate-on-switch lifecycle.** Switching modes tears the previous
backend down completely — the viewport widget is removed and deleted,
which (via Qt parent/child ownership and RAII) stops its timers
(offscreen `render_timer_`, real-time `sample_timer_`), ends the
real-time `requestUpdate` repaint loop, and releases its Vulkan context
(`unique_ptr<VulkanCtx>`). Only then is the newly-selected backend
built. This directly addresses the concern that an inactive backend
could keep doing work under the surface: after a switch, nothing of the
old mode remains running. The cost is a brief flicker as the widget is
recreated.

Render config and SSAA settings are preserved across switches and
re-applied to the new backend.

When hardware Vulkan is unavailable (e.g. software-only rendering), the
Real-time entry is shown but disabled in both the combobox and the
menu, and any programmatic request to enter it falls back to Offscreen
CPU with a status-bar note.

A headless smoke hook (`FREP_SMOKE_CYCLE_MODES=1`) cycles all three
modes after startup and exits, exercising the teardown/rebuild path.

### Removed

- The standalone "Use GPU (Vulkan compute)" menu toggle — folded into
  the unified render-mode selector.

---

## [4.2.14] — 2026-05-31

### Fixed — Real-time SSAA did nothing at odd factors (3×)

On the real-time path, SSAA 2× cleaned up silhouette aliasing but 3×
appeared to do nothing — the artifact pixels came back. Root cause:
the real-time downsample is a single bilinear `vkCmdBlitImage`, and a
bilinear tap only box-filters correctly at *even* supersample factors.
At an even factor the destination pixel centre lands between source
texels and averages a clean 2×2; at an odd factor (3×) it lands on the
centre texel of the 3×3 block, the neighbour weights vanish, and the
blit point-samples that one texel — no averaging at all.

The real-time SSAA control now offers **1× / 2× / 4×** (even factors,
which the bilinear blit handles correctly) instead of 1×/2×/3×; odd
values passed in are snapped up to the next even factor. The offscreen
GPU and CPU paths use a true box downsample and are unaffected — they
can take any factor — but the menu is unified to the same options for
consistency.

(A future option would be a dedicated box-filter resolve pass for the
real-time path so it too could take arbitrary factors; deferred as it
needs a second compute pass and on-hardware validation.)

---

## [4.2.13] — 2026-05-31

### Fixed — SSAA had no effect on the offscreen GPU path

The offscreen GPU renderer (`Viewport::do_render_gpu`, used when
Compile is set to GPU but not real-time) ignored the SSAA setting
entirely: it always rendered at widget resolution, so aliasing and the
single silhouette pixels that SSAA cleans up on the CPU path survived
on GPU. (The CPU offscreen path and the real-time path both already
honoured SSAA — only this third path missed it.)

The offscreen GPU path now renders at `ssaa ×` the target resolution
and **box-downsamples** the supersampled buffer (averaging each ss×ss
block) into the final image. A true box filter is actually better than
the real-time path's bilinear blit at 3× (which only mixes a 2×2
corner of the 9 samples). SSAA is still skipped while dragging the
camera, where the path already drops to half resolution for
responsiveness.

---

## [4.2.12] — 2026-05-31

### Fixed — Render tools wrote a file named "--help"

The CLI render tools (`frep_render`, `frep_advanced`) took their output
path from `argv[1]` without checking for flags, so invoking them with
`--help` wrote a PPM image literally named `--help` into the working
directory — which then got swept into the project archive. Both tools
now recognise `--help` / `-h`, print a usage summary, and exit without
writing anything. The stray file was removed and `.gitignore` updated
to catch the known offenders (they lack a `.ppm` extension, so the
existing `*.ppm` rule missed them).

---

## [4.2.11] — 2026-05-31

### Changed — Raised raymarch step cap to 192 (silhouette convergence)

A handful of single bright pixels survived the grazing-ray rescue,
concentrated on *interior* silhouettes — where one object's edge
overlaps a second object behind it (e.g. a sphere peeking out from
behind a rounded cube). There the grazing ray threads the narrow gap
between the two surfaces and can still exhaust the step cap before
converging.

Rather than widen the rescue threshold further (which risks visibly
fattening object outlines), the raymarch step cap was raised from 128
to 192. This lets the grazing rays actually converge to `epsilon`
instead of relying on the rescue to paper over an exhausted march,
clearing nearly all of the remaining specks at the source. The cost is
paid almost entirely by grazing rays — interior rays still converge in
a handful of steps — so measured render time barely moves (~195 →
~200 ms on the CPU reference scene). The grazing rescue (80·epsilon)
stays as the backstop for any ray that still runs out.

### Note

Single pixels may still appear at the exact crossing point of two
overlapping CSG silhouettes at very low resolution; SSAA 2×/3× removes
those, as they are now genuine sub-pixel sampling cases rather than
march misclassifications.

---

## [4.2.10] — 2026-05-31

### Changed — Widened grazing-ray rescue threshold

The 4.2.9 grazing-ray rescue removed almost all of the silhouette
fringe, but a few isolated bright pixels remained at points where the
CSG surface gradient is steeper and the grazing ray stops slightly
farther out (between 40·epsilon and the surface). The rescue threshold
was widened from 40·epsilon to 80·epsilon on both paths, which catches
those remaining pixels. Still small in world units (~0.04 at the
default epsilon), so genuine background pixels — which stop far from
any surface — are unaffected; object-pick tests confirm silhouette
boundaries didn't measurably grow.

---

## [4.2.9] — 2026-05-31

### Fixed — Bright sky-coloured fringe along object silhouettes

Reported as a thin light contour tracing object outlines, visible
specifically where the background behind the silhouette is dark (a
dark object behind, or a deeply-shadowed surface) and NOT removed by
SSAA — ruling out plain edge aliasing. Diagnosed as a raymarch
classification problem: along a silhouette the ray skims the thin
valley just outside the surface, taking ever-smaller steps, and can
exhaust `max_steps` while still a hair above `epsilon`. The loop then
reported a *miss*, so the pixel was painted with the (bright) sky
colour — which only stands out against a dark background, exactly as
observed.

Both renderers now perform a **grazing-ray rescue**: they track the
last sampled distance and, if the march stopped close to the surface
(within ~40·epsilon) and inside the max distance, reclassify the pixel
as a hit and shade the surface instead of returning sky. This removes
the silhouette fringe while leaving genuine sky/background pixels
untouched (those stop far from any surface). CPU JIT and GPU paths use
the identical threshold and stay in agreement.

Unlike the earlier silhouette investigation (4.2.8), this one is a
true classification bug, not aliasing — which is why SSAA didn't help
and why the fix is in the tracer rather than the sampler.

---

## [4.2.8] — 2026-05-31

### Changed — More robust Cook-Torrance specular (CPU)

The CPU specular term computed `D · G / (4·ndotv·ndotl)` with the
geometry term `G` and the BRDF denominator as independent factors.
Since `G = G_v·G_l` already carries `ndotv` and `ndotl` in its
numerators, those factors algebraically cancel against the
denominator, leaving the identical-but-stable `D / (4·gv·gl)`. The
naive form could spike at grazing angles (ndotv → 0) before the
firefly clamp caught it; the cancelled form is inherently bounded.
This matches standard height-correlated visibility formulations and
keeps the firefly clamp purely as a NaN/overflow backstop. No visible
change to normal shading; removes a class of potential grazing-angle
specular spikes.

### Investigated — Thin bright rim between objects and their contact shadows

Reported as a faint light line tracing object silhouettes (visible in
the GUI with SSAA off). Reproduced and analysed via offscreen renders:
the bright pixels are the **floor itself**, showing through a 1–2px
band between an object's silhouette and where its contact shadow
begins. With an angled key light there is a genuinely lit sliver of
floor there, so this is **silhouette aliasing of a real thin feature**,
not a shading bug. Confirmed by doubling the render resolution: the
affected pixel count drops ~35× (75 ppm → 2 ppm), i.e. far faster than
linearly — the hallmark of an aliasing edge that vanishes in the
limit.

The existing **SSAA** control (Render tab → SSAA 2× / 3×) is the
correct remedy and removes it. No shading change was made, since
tuning the shadow bias to hide the sliver risks introducing
shadow-acne or peter-panning elsewhere — a worse trade for a
sub-pixel cosmetic edge that SSAA already addresses.

---

## [4.2.7] — 2026-05-31

### Fixed — White fringe along object silhouettes (specular fireflies)

A bright white outline (a dashed shimmer under temporal accumulation)
traced object silhouettes. The cause was the Cook-Torrance specular
microfacet denominator `1 / (4·ndotv·ndotl)`: at grazing silhouette
angles `ndotv → 0`, so even a tiny `D·G·F` was divided by a near-zero
value and exploded into a huge specular value that the final
`clamp(color, 0, 1)` turned into a white fringe.

Both renderers now clamp the specular term to a sane maximum (GPU:
`min(specular, 8.0)`; CPU: the equivalent `8·PI` cap before the PI
compensation factor). This removes the silhouette fringe — and its
accumulation-time flicker — without visibly dimming legitimate
highlights. The two paths stay in pixel-level agreement.

---

## [4.2.6] — 2026-05-31

### Fixed — Radial banding along soft-shadow penumbrae (GPU)

The area-light sampler placed its N samples on fixed stratified rings
(`fi = (s + 0.5) / N`) and only rotated the spiral per frame. With a
low sample count the samples therefore always landed at the same radii,
so the penumbra showed radial banding / jagged dark edges that temporal
accumulation — which only varied the angle — never fully smoothed,
even after the image had otherwise converged.

The radial position is now jittered per pixel and per frame via a
second decorrelated hash (`fi = (s + rjit) / N`). Combined with the
existing angular rotation and temporal accumulation, the samples now
cover the light disk far more evenly, and the penumbra converges to a
clean, smooth gradient with no banding. The CPU path (which renders a
single clean frame with no temporal pass) keeps its optimal static
stratification and is unaffected.

---

## [4.2.5] — 2026-05-31

### Fixed — Dark speckle along CSG silhouettes (GPU)

The GPU raymarch loops advanced by the full reported SDF distance
(`t += d`), while the CPU JIT path scaled each step by a safety
factor (0.85). For CSG scenes (unions / intersections of primitives)
the reported distance is not a true Euclidean distance near the seams,
so full-distance stepping overshot thin features and grazing
silhouette rays — producing dark speckle along object edges and the
occasional wrong pixel, especially visible where objects meet the
ground or each other.

All four GPU march loops (primary, reflection, and both shadow
variants) now scale the step by `safety_factor`, matching the CPU JIT
path. This removes the edge speckle and brings the two renderers back
into pixel-level agreement on CSG scenes. Cost is a few extra march
iterations per ray; negligible at these scene complexities.

---

## [4.2.4] — 2026-05-31

### Changed — Denoising now freezes when converged

v4.2.3 made temporal accumulation visibly work, but it held the blend
factor at `1/target` forever, so once converged each new noisy frame
still contributed `1/target` of fresh noise — the image never fully
settled and a faint shimmer remained.

Accumulation now genuinely converges and then **freezes**: the blend
factor follows `1/(n+1)` for the first `accum_frames` frames (a true
running mean), then drops to 0. With a zero blend the shader keeps the
already-converged pixels (`mix(prev, new, 0) == prev`), so the image
holds perfectly steady until the camera or scene changes, at which
point the average restarts from frame 0.

The render loop stays continuous (it does not suspend on convergence)
so the camera-orbit handlers — which rely on the running loop to pick
up pose changes — keep working. The frozen frames are cheap (no blend
math, just a passthrough store).

Net effect: with low shadow samples (e.g. 4) and a wide light radius,
holding the camera still now shows the contact-shadow noise resolve to
a clean, stable penumbra over `accum_frames` frames, then stop moving
entirely.

---

## [4.2.3] — 2026-05-30

### Fixed — Temporal denoising had no visible effect

v4.2.2 added the accumulation blend, but the soft-shadow sampler
seeded its jitter purely from the world-space hit position
(`p.xy + p.z`). That seed is identical every frame for a static
camera, so each accumulated frame rendered the *same* jitter pattern
and averaging them converged to the single noisy image — no visible
smoothing.

The shader now also folds a per-frame `frame_seed` (pushed from the
renderer's accumulation counter) into the jitter seed, so each
accumulated frame samples a different set of points across the area
light. Averaging them now genuinely converges the penumbra from
dithered to smooth while the camera is still, and resets cleanly on
camera/scene changes. The offscreen path leaves `frame_seed` at 0
(single deterministic frame, unchanged).

### Note on visibility

The denoising effect is most visible with a **low** shadow-sample
count and a **large** light radius — that combination is noisy enough
per frame that the frame-to-frame convergence is obvious. At high
sample counts (e.g. 16) each single frame is already fairly smooth,
so accumulation refines it only subtly.

---

## [4.2.2] — 2026-05-30

### Added — Real-time temporal denoising (accumulation)

The "Denoise (accum frames)" control is now functional on the real-
time path. When the camera and scene are static, successive frames are
blended into a running average held in the storage image, converging
the noisy area-light soft shadows (and any other stochastic noise) to
a smooth result over the configured number of frames.

How it works:
- A per-frame blend factor `accum_blend = 1/(n+1)` is pushed to the
  compute shader, which does `out = mix(previous, new, accum_blend)`
  reading the previous frame back from the storage image. Each
  invocation touches only its own texel, so the in-place read-modify-
  write is race-free — no second image needed.
- The storage-image barriers were reworked to *preserve* contents
  between frames while accumulating (transitioning from GENERAL rather
  than UNDEFINED), and to return to GENERAL after the blit so the next
  frame's `imageLoad` sees a valid previous frame.
- Accumulation resets automatically when the view changes. The
  renderer hashes the camera pose, projection, light parameters, and a
  new monotonic `SceneGraph::revision()` counter; any change restarts
  the average from frame 0. A storage-image resize / SSAA change also
  resets it (the image then holds no valid history).
- The output storage image is now declared read-write (was
  `writeonly`) so the shader can read the previous frame. The offscreen
  path is unaffected: it leaves `accum_blend` at its default 1.0, which
  skips the blend entirely and renders a single clean frame as before.

Set "Denoise (accum frames)" to e.g. 16–32 and hold the camera still:
the soft shadows will visibly converge from dithered to smooth over
that many frames, then hold steady until you move.

### Added

- `SceneGraph::revision()` — a monotonic counter bumped by every
  mutation routed through the scene's setter methods. Used by the
  accumulation reset logic; also handy for future change-detection.

---

## [4.2.1] — 2026-05-30

Bug-fix release addressing three issues found in v4.2.0 hardware
testing.

### Fixed — Material editing deselected objects mid-edit

After changing any Material field, the inspector rebuilt its object
list (via `refresh()`), which cleared the list-widget selection. That
deselected the objects being edited, disabled the Material panel
fields, and forced the user to re-click the objects to continue. The
inspector now preserves (and restores) the selection across a
refresh, and a material edit no longer triggers an inspector refresh
at all (appearance-only changes don't alter the object list).

### Fixed — Soft shadows showed no change with sample count

When `shadow_samples > 1`, each shadow ray was still using the IQ
single-ray penumbra approximation *and* being averaged across area-
light samples. The two softening mechanisms cancelled out visually,
so changing "Shadow samples" / "Light radius" appeared to do nothing.
Now, multi-sample mode casts HARD binary shadow rays and derives all
softness from averaging the jittered area-light samples — so the
controls have a visible effect. Single-sample mode keeps the cheap
IQ penumbra (controlled by the "Soft shadows" softness slider).
Applied to both the GPU (GLSL) and CPU (LLVM IR) paths.

### Note

Area-light soft shadows without temporal accumulation are somewhat
noisy (dithered penumbra), since each pixel jitters its sample
pattern independently. Enabling denoising / accumulation (a planned
follow-on) will converge them to smooth gradients.

---

## [4.2.0] — 2026-05-30

A large feature release adding mirror reflections, soft shadows, a
material system upgrade, and object-editing tools (duplicate, copy/
paste, a numeric transform gizmo with grid snapping). The headline
shader features have full CPU JIT ↔ GPU parity.

### Added — Mirror reflections

Objects can now reflect their surroundings. A new per-material
`reflectivity` field (0 = matte, 1 = perfect mirror) drives secondary
reflection rays, and `TracerConfig.max_bounces` (0–4, exposed as
"Reflections (bounces)" in the Render tab) caps the recursion depth.
A Schlick-Fresnel term makes grazing angles reflect more, and a
throughput accumulator keeps reflections-of-reflections physically
proportionate.

Implemented on **both** rendering paths:
- **GPU (GLSL emitter):** a `scene_reflectivity()` lookup, a reusable
  `sky_color()` helper, and a bounce loop in `main()`.
- **CPU (LLVM IR):** the shading logic was factored into a reusable
  `shade_hit` helper; `emit_scene_reflectivity` provides a BVH-walk
  reflectivity lookup; and an unrolled bounce-block chain in
  `emit_tracer` re-marches and re-shades each reflection ray, sampling
  the sky on a miss. `BVH::Entry` gained a `reflectivity` field.

### Added — Soft shadows

`TracerConfig.shadow_samples` (1–16, "Shadow samples" in the Render
tab) plus `shadow_light_radius` cast multiple jittered shadow rays
toward a virtual spherical area light and average them, producing
soft penumbrae. A golden-angle sample spiral keeps the distribution
even; `shadow_samples == 1` collapses to the previous single-ray
hard shadow with zero overhead. Both CPU and GPU paths honour it.

### Added — Material editor upgrades

- **Reflectivity slider** in the Material tab.
- **Multi-edit:** editing pattern, scale, roughness, metallic, or
  reflectivity now broadcasts to *every* selected object (albedo
  multi-edit remains via the Inspector's colour button). The primary
  selection populates the widgets.
- **Presets:** a one-click preset dropdown — Metal, Brushed metal,
  Plastic, Matte, Glass, Mirror, Emissive — fills in the scalar PBR
  fields while leaving each object's albedo intact.
- `reflectivity` is saved/loaded in the `.frep` scene format.

### Added — Object editing tools

- **Duplicate (Ctrl+D)** clones the selected objects in place, selects
  the clones, and pushes one undo entry each.
- **Copy / Paste (Ctrl+C / Ctrl+V)** via an internal object clipboard.
- A new `io::clone_node` deep-copies any geometry subtree (including
  plugin nodes) by round-tripping through the scene JSON machinery —
  no per-node-type `clone()` needed. Descendant ids are rewritten so
  clones never collide with their originals.

### Added — Transform gizmo + snapping

- A numeric **position (X/Y/Z) gizmo** in the Properties panel moves
  the selected object in world space. It's represented as an implicit
  `TranslateNode` wrapping the geometry root, so every path (CPU JIT,
  GPU, BVH, picking, meshing) handles it automatically. Repeated edits
  update in place rather than nesting; a zero offset unwraps cleanly.
  Each commit is an undoable `SetTransformCommand`.
- **Grid snapping:** an optional toggle + step size rounds committed
  translations to the nearest grid multiple.

### Added — Denoising scaffolding (real-time)

`TracerConfig.accum_frames` and a Render-tab "Denoise (accum frames)"
control are in place for temporal accumulation on the real-time path.
The accumulation buffer itself is a follow-on (it needs a second
persistent GPU image + camera-reset logic); the field is wired
through the scene hash so it's ready for that work.

### Notes

- The transform gizmo is **numeric** (spinboxes), not a draggable 3D
  axis-handle overlay. It provides the same translation capability,
  undoable and consistent across both viewports.
- New unit tests: 6 reflection/soft-shadow IR-validity tests, 4
  clone-node deep-copy tests, 7 transform-gizmo tests (including a
  check that translation actually moves the surface in SDF eval).

---

## [4.1.0] — 2026-05-26

### Added — Multi-selection across viewport, inspector, and node graph

Objects can now be selected as a set rather than one at a time.

- **Inspector list** uses extended-selection mode: plain click
  selects one, Ctrl-click toggles individual items in/out of the
  set, Shift-click range-selects.
- **Viewport** (both offscreen and real-time): Ctrl + left-click
  adds the clicked object to the current selection; a plain click
  replaces it. A click that lands on empty background clears the
  selection (unless Ctrl is held).
- **Properties panel** shows the primary (most-recently-clicked)
  object's values. When the selection spans objects with differing
  values, the field shows an em-dash "—" (visibility checkbox goes
  to its partially-checked tri-state). This mirrors the mixed-value
  convention in DCC tools like Houdini and Blender.
- **Editing applies to the whole set.** Changing the albedo colour,
  toggling visibility, or removing applies to every selected object.
  Each object keeps its own non-edited material fields (roughness,
  metallic) — only the edited channel is broadcast.
- **Node graph** follows the primary selection (the graph editor is
  inherently single-tree). The other selected objects remain
  selected in the inspector and still receive multi-edits.

The selection signal changed from `object_selected(QString)` to
`selection_changed(QStringList)` where `ids[0]` is the primary.
Internal consumers (node graph, material editor) adapt by taking
the primary; multi-edit consumers iterate the full list.

### Added — GPU TracerConfig parity

The GLSL emitter now bakes `max_steps`, `max_dist`, `epsilon`, and
`shadow_steps` from `TracerConfig` into the emitted shader, instead
of the previous hard-coded `128` / `50.0` / `0.001` / `32`. Editing
these in the Render tab now affects all three rendering paths
identically (the CPU JIT path already honoured them). The four
fields are mixed into both GPU scene-hash recipes so a change
forces a re-emit + pipeline rebuild.

### Added — Customisable sky gradient

`TracerConfig` gains `sky_top` and `sky_horizon` RGB triples. Two
colour-picker swatches in the Render tab let the user set the
zenith and horizon colours of the background gradient. Both CPU and
GPU paths bake the colours in (hashed for rebuild). Defaults match
the previous fixed gradient (`(0.4,0.5,0.7)` → `(0.6,0.7,0.85)`),
so existing scenes look unchanged until edited.

### Added — GPU object picking on the real-time path

Clicking (without dragging) on the real-time viewport now selects
the object under the cursor in the inspector. Implementation re-uses
the CPU `ScenePicker` (the same JIT-compiled picker the offscreen
path uses) run on the click coordinates — no separate GPU ID pass
needed. Ctrl-click adds to the selection. A drag of more than 4
pixels is treated as a camera orbit, not a click, so picking never
fights with navigation.

### Notes — screenshot on the real-time path

`File → Save Image…` continues to use the Qt-side
`QScreen::grabWindow` capture introduced in v4.0.5. A true GPU-side
`vkCmdCopyImage` readback (capturing pre-compositor output) remains
a follow-on; the Qt path is adequate for the common case and has
zero per-frame cost.

---

## [4.0.6] — 2026-05-26

### Fixed — CPU and GPU paths now produce identical colours

v4.0.5 dropped the CPU's `sqrt()` gamma encoding as a cosmetic
fix, but the two paths still diverged because the shading model
itself had silently grown apart over time:

1. **CPU CookTorrance dropped Lambert `/π`**, matching the GPU
   emitter which bakes the π factor into the per-light intensity
   convention (Frostbite "Moving Frostbite to PBR" approach).

2. **CPU CookTorrance specular now multiplied by π**, mirroring
   the GPU emitter's `specular * PI` term. Without this the CPU
   highlights were noticeably dimmer.

3. **CPU Blinn-Phong dropped its ambient = 0.15 term** — ambient
   is now added once per pixel by the caller, multiplied by AO,
   exactly as the GPU does.

4. **Per-pixel ambient is now `albedo * 0.08 * ao_v`** on both
   paths. Previously CPU added `albedo * 0.03` (or `0.15` for
   Blinn-Phong) per-light inside the shader function.

5. **AO no longer multiplies the entire accumulated lighting** on
   the CPU path. It now only attenuates the ambient term, matching
   the GPU. Previously the CPU darkened the entire image when AO
   was on (default); now AO has the subtle, plausibly-soft effect
   visible on the GPU.

6. **Sky gradient uses GPU's constants** —
   `mix(vec3(0.6, 0.7, 0.85), vec3(0.4, 0.5, 0.7), s)` —
   instead of the older CPU `lerp((0.5, 0.7, 1.0), (0.1, 0.2, 0.5))`.
   Sky on both paths is now a medium-blue gradient.

After these changes, CPU JIT and offscreen GPU produce visually
identical output on the same scene — verified on NVIDIA GTX
1050 Ti by side-by-side comparison of the sphere/box/capsule/
blobs test scene.

---

## [4.0.5] — 2026-05-25

### Added — SSAA on the real-time path

Super-sampling anti-aliasing now works in `--realtime` mode, the
same way it works on the offscreen path. The combobox in the
Render tab is no longer disabled when real-time is active. Off,
2x2 (4 rays/pixel), and 3x3 (9 rays/pixel) are available; quality
scales quadratically as expected.

Implementation: `ComputeBlitRenderer` resizes its storage image to
`(swapchain_w × N, swapchain_h × N)` for SSAA factor N and
dispatches the compute shader at that scaled resolution. The final
`vkCmdBlitImage` to the swapchain uses `VK_FILTER_LINEAR` when N>1
so the GPU itself does a single-pass box-filter downsample —
effectively free relative to the compute cost. Changing factor
forces a `vkDeviceWaitIdle` + storage-image rebuild so the
descriptor set re-binds to the resized view.

### Added — Render timing on the real-time path

A Vulkan timestamp query pool (6 slots = 3 frames × 2 timestamps)
records start- and end-of-frame timestamps around the dispatch and
blit. A 10 Hz `QTimer` on the `FRepVulkanWindow` polls the most
recent resolved measurement and emits `render_time_sampled(ms)`,
which `VulkanViewport` forwards into `IViewport::render_completed`.
MainWindow's status bar consumes the signal exactly the way it
does for the offscreen path, so the user sees "Render Xms" updates
at 10 Hz on both backends.

The query pool is only created when the device's queue family
reports `timestampValidBits > 0` (true on every modern desktop
GPU). On chips that lack timestamps the feature silently disables
and the status bar reverts to the static "Real-time GPU viewport
active" string.

### Added — Mid-frame screenshot on the real-time path

`File → Save Image…` (the existing menu action) now works in
`--realtime` mode too. The implementation takes the pragmatic
Qt-side route: `QScreen::grabWindow` over the Vulkan widget's
geometry. The capture matches what the user sees on screen,
including any compositor adjustments. A "true" GPU-side capture
via `vkCmdCopyImage` into a staging buffer would unlock pre-
compositor output, and remains a v4.0.6 follow-on.

### Changed — Unified colour palette across all three backends

The CPU JIT path used to apply a `sqrt(color)` gamma approximation
in the final pixel write; both GPU paths used only `clamp(color, 0,
1)`. The CPU output therefore looked perceptibly darker than the
GPU output for the same scene. v4.0.5 drops the CPU's sqrt so all
three backends produce visually identical results — the GPU-style
brighter palette (slightly over-bright highlights, saturated mid-
tones) is now the reference for the project.

One unit test (`Camera.OrthographicShowsEqualSizedSpheres`) had a
brightness threshold tuned for the gamma-encoded output; it was
relaxed from 0.05 to 0.01 so it survives the change.

### Changed — `FRepVulkanWindow` lives in its own header

To support the new `render_time_sampled` signal, `FRepVulkanWindow`
needed a `Q_OBJECT` macro and a moc-generated meta-object. Qt
6.4.2's AutoMoc only scans `.hpp` files (an old issue we already
hit on `IViewport`), so the class declaration moved out of
`vulkan_viewport.cpp` into a new `gui/frep_vulkan_window.hpp`. The
header is in the manual `qt6_wrap_cpp` list alongside the other
problematic AutoMoc victims; the renderer (`ComputeBlitRenderer`)
moved with it into the named `frep::gui` namespace (no longer
anonymous) so the new header can refer to it.

This also required pulling `<vulkan/vulkan.h>` to the very top of
`vulkan_viewport.cpp`'s include list — Qt's `<QVulkanInstance>`
defines `VK_NO_PROTOTYPES` before including `<vulkan/vulkan.h>`
itself, which suppresses the global Vulkan function prototypes
that the rest of the file uses. Putting our `vulkan.h` first lets
its include guard short-circuit Qt's later attempt, leaving the
prototypes available.

### Known limitations carried into v4.0.5

- **GPU object picking on the real-time path is not yet
  implemented.** Clicking on the live viewport doesn't select an
  object in the Inspector. A proper implementation would dispatch
  a separate "ID pass" compute pipeline that writes the closest-hit
  object index into a storage buffer, then read back at click time.
  Estimated ~200 LoC; deferred to v4.0.6.
- The Qt-side screenshot captures whatever is on screen, including
  any window compositor processing. Users who need bit-exact GPU
  output should run with `--offscreen` and use `File → Save Image…`
  there.

---

## [4.0.4] — 2026-05-25

### Added — unified Render-tab settings across all backends

The Render tab (shadow toggle, AO toggle, shadow softness, AO
strength, shading model) now controls both the offscreen rendering
path (CPU JIT + offscreen GPU) and the real-time path (`--realtime`)
identically. Previously these settings only affected the offscreen
path; the real-time path baked its shading configuration at scene
load with no way to change it interactively.

Implementation across the stack:

1. **`GlslEmitter::emit()` accepts `const TracerConfig&`**. The
   emitter branches on `cfg.shading_model` (CookTorrance microfacet
   BRDF vs. Blinn-Phong with roughness-derived shininess), on
   `cfg.enable_shadows` (IQ-style soft shadow accumulator vs. a
   constant-returning stub the optimiser folds away), and on
   `cfg.enable_ao` (per-sample weighted accumulator vs. constant-1
   stub). Both `cfg.shadow_softness` and `cfg.ao_strength` are baked
   into the emitted GLSL constants, so a slider edit triggers
   pipeline rebuild.

2. **Both GPU paths share the same hash recipe**. The
   `compute_scene_hash` in `gui/vulkan_viewport.cpp` AND the
   `scene_hash` in `gui/viewport.cpp` (offscreen GPU mode) both mix
   shading model, shadow toggle, AO toggle, softness, and strength
   into the hash alongside the structural fields. A Render-tab
   change drifts the hash → next render's compare-and-rebuild
   triggers a re-emit + SPIR-V recompile + pipeline rebuild.
   Without the hash recipe extension the offscreen GPU path silently
   ignored Render-tab edits — the user could see the old shading
   config persist when panning the camera.

3. **`IViewport::set_tracer_config()` virtual method**. Both
   backends implement it: `OffscreenViewportAdapter` forwards to the
   `IncrementalCompiler::tracer_config()` and triggers
   `force_recompile`; `VulkanViewport` stashes the config in its
   `FRepVulkanWindow::pending_cfg_` and the renderer's next frame
   picks it up via the hash check.

4. **MainWindow holds a single `TracerConfig render_config_`**.
   Render-tab widgets bind to it both ways (initialised from its
   defaults at construction, edited by callbacks). Every callback
   calls `apply_render_config()`, which fans out via `viewport_iv_`
   regardless of active backend. An initial `apply_render_config()`
   at the end of `build_ui()` pushes the defaults so a freshly-
   launched application starts in a known-synchronised state.

5. **`tracer_box` is no longer disabled on `--realtime`**. The only
   Render-tab control still grey on the real-time path is the SSAA
   combobox — real-time renders at native swapchain resolution and
   would need a separate multi-tap dispatch to support SSAA. That
   remains a v4.0.5 follow-on.

### Added — Camera tab works in GPU paths

`scene.camera().fov_deg`, `.projection`, and `.ortho_size` were
previously read only by the CPU JIT path. The offscreen GPU path
used `build_push_from_scene` with a hard-coded `fov_radians = 1.2`
default and ignored projection mode entirely, and the real-time
path inherited the same blind spot. Editing the Camera tab silently
did nothing on either GPU path.

Fixed across three files:

- **`core/gpu/vulkan_ctx.hpp`**: `ShaderPush` extended with
  `projection_mode` (1.0 = orthographic, 0.0 = perspective) and
  `ortho_size` (half-height in world units when orthographic). New
  struct size is 168 bytes, still well under the 256-byte typical
  push-constant limit.

- **`core/gpu/shader_push_builder.hpp`**: `build_push_from_scene`
  now reads `cam.fov_deg`, `cam.projection`, and `cam.ortho_size`
  from the scene. The `fov_radians` parameter is now unused —
  retained as `/*fov_radians_unused*/` for ABI continuity.

- **`core/gpu/glsl_emitter.cpp`**: The emitted Push uniform layout
  matches the new struct; `main()` branches on
  `pc.projection_mode > 0.5` to choose ray construction. Perspective
  keeps the existing `origin = cam_pos, dir = normalize(fwd + u·fov·right + v·fov·up)`
  formula; orthographic uses `origin = cam_pos + u·ortho·right + v·ortho·up, dir = normalize(fwd)`.
  Ray-march loop and downstream shade() are unchanged.

Camera-tab edits propagate through push constants only — no
pipeline rebuild needed, so the change appears on the next render
with no recompile pause. Both GPU paths handle perspective ↔ ortho
switches and FoV slider drags interactively.

### Fixed — Qt 6.4.2 AutoMoc emitted empty moc files for IViewport hierarchy

A latent CMake / Qt interaction caused AutoMoc to produce 0-byte
`moc_*.cpp` files for any Q_OBJECT class inheriting from a custom
QObject-derived interface rather than from `QWidget`. Symptom:
linker errors about missing vtables and `staticMetaObject` symbols
for `IViewport`, `OffscreenViewportAdapter`, `VulkanViewport`, and
(after introducing the new `TracerConfig` member) `MainWindow`.

Workaround: invoke `moc` manually for those headers via
`qt6_wrap_cpp` in `CMakeLists.txt`, alongside the still-running
AutoMoc for everything else. The AutoMoc-emitted empty placeholders
are now harmlessly compiled as no-ops; the manual-moc outputs
provide the real meta-object code. Documented in `CMakeLists.txt`
comments for any future contributor wondering why those four
headers are special.

### Changed — `FRepVulkanWindow` exposed as forward declaration

`gui/vulkan_viewport.hpp` now forward-declares `FRepVulkanWindow` so
`VulkanViewport` can hold a weak pointer to it for config
forwarding. The class itself stays defined in
`vulkan_viewport.cpp` (in `frep::gui::` rather than the previous
anonymous namespace) — public consumers of the header still don't
pull in Vulkan headers.

---

## [4.0.3] — 2026-05-24

### Fixed — texture-pattern materials crashed CPU JIT codegen

`SceneCodegen::emit_scene_material` had a non-exhaustive switch over
`Material::Pattern` — the Texture case was missing. When the scene
graph contained a textured material AND the offscreen path was
active (always true unless `--realtime` is passed), the switch fell
through leaving an LLVM Value* uninitialised, and the subsequent
`IRBuilder::CreateFSubFMF` segfaulted inside LLVM.

This had been latent since textures were added because the prior
texture-path resolution code in `scene_io` silently failed to load
the pixels for `examples/06_textured_objects.json` (the relative
paths were resolved against the .frep file's directory rather than
the working directory, producing nonexistent paths). With
`texture_rgba` empty, the renderers' own emit-time
"empty-texture-fallback" paths kicked in everywhere and the codegen
switch never actually saw `Pattern::Texture` in practice.

The v4.0.2 three-tier loader fix made textures load correctly,
which immediately exposed the codegen crash. The fix here is the
matching Texture case in the codegen switch — it falls back to
solid `albedo` on the CPU JIT path, matching what the GLSL emitter
already does when `texture_rgba` is empty.

Behavioural contract:
- CPU JIT (default offscreen path) renders Texture-pattern
  materials as if they were Solid, using `albedo` directly.
- GPU compute (offscreen `Render → Use GPU` or `--realtime`)
  renders the actual triplanar texture via the `texture_pixels`
  SSBO at descriptor binding 2.

Three regression tests in `test_codegen_texture_fallback.cpp`
guard this: a single-textured-sphere compile (the minimal repro), a
multi-pattern mix (the actual `06_textured_objects.json` shape),
and a positive check that `compile_if_changed` returns a valid
function pointer rather than an error.

---

## [4.0.2] — 2026-05-24

### Added — MeshSDF / texture maps on real-time path

The `ComputeBlitRenderer` (real-time `--realtime` viewport) now
allocates and binds storage buffers for mesh voxel grids (binding 1)
and texture pixel arrays (binding 2), matching the binding layout
the offscreen `VulkanCtx` uses and the GLSL emitter expects. Scenes
that use `MeshSDF` nodes, image-based texture materials, or both no
longer fall through to the dark-teal clear fallback — they render
identically to the offscreen path.

Implementation: `create_storage_buffer()` helper packages the
familiar Vulkan dance (VkBuffer + VkDeviceMemory + memcpy upload +
descriptor write) into a single 4-arg call, reused for both
bindings. Buffers use `HOST_VISIBLE | HOST_COHERENT` memory so
upload is a single map+memcpy+unmap with no staging buffer or queue
submission — for typical sizes (~256 KB voxel grids, ~1 MB textures)
the bandwidth difference vs. `DEVICE_LOCAL` is negligible and the
simplicity is worth far more than a few percent shader-side fetch
speed. Buffers are owned at pipeline lifetime: a pipeline rebuild
(triggered by the scene-hash check) frees and re-uploads them as a
single atomic operation.

The `mesh_voxels_cache_` and `tex_pixels_cache_` members hold owned
copies of the emitted payloads across the rebuild window, so a
camera-only refresh doesn't have to re-emit GLSL just to recover the
mesh data.

### Added — IViewport abstraction

`MainWindow` no longer needs to special-case which rendering backend
is active. A minimal abstract interface `gui/iviewport.hpp` exposes
just what the window needs to wire signals (`render_completed`,
`object_picked`), trigger refreshes (`invalidate`), and obtain the
QWidget for layout placement. Both backends implement it:

- `OffscreenViewportAdapter` (`gui/offscreen_viewport_adapter.hpp`)
  wraps the existing offscreen `Viewport` QWidget. The 4-arg
  `Viewport::render_completed` signal is adapted to the 1-arg
  `IViewport::render_completed` via a connect-time lambda.
- `VulkanViewport` (`gui/vulkan_viewport.hpp/cpp`) now itself
  derives from `IViewport`, with a `create_iv()` factory that
  returns the interface pointer. The legacy `create()` returning
  `QWidget*` is kept for backward compatibility.

`MainWindow` now holds:
- `viewport_iv_` (always non-null) — the universal handle, used for
  all signal connections, status, and layout.
- `viewport_` (Viewport*) — non-null only when the offscreen backend
  is active, used for CPU-specific operations (compiler config,
  SSAA, screenshot readback). On the real-time path it is nullptr,
  and the CPU-specific UI controls (Render → Use GPU toggle, Ray
  tracer settings group, SSAA combobox, Compile mode dropdown) are
  disabled at construction with a tooltip explaining why.

The previous workaround — constructing both viewports and `hide()`-
ing the unused one — is gone. The status bar no longer shows stale
"Render NNNms" messages from a background offscreen render when the
real-time path is active.

Two follow-on items remain:

- `VulkanViewport` doesn't yet emit `render_completed` or
  `object_picked` signals. Render time would require a small
  per-frame `vkCmdWriteTimestamp` + query-pool setup; object
  picking would need a separate compute pass that writes the
  closest hit's object ID into a single-uint storage buffer for
  CPU readback. Documented in `iviewport.hpp`.
- `MainWindow::on_export_image()` declines on the real-time path —
  the rendered frames live entirely on the GPU and are immediately
  consumed by the presentation engine. A vkCmdCopyImage into a
  staging buffer mid-frame would unlock this; deferred to v4.0.3.

---

## [4.0.1] — 2026-05-24

### Added — polish sprint (Doxygen, perf gate, texture UX, real-time viewport)

Four follow-on items that finish the loose threads in the v4.0 series.

**Doxygen API reference** — `cmake --build build --target docs` now
produces a clean HTML reference under `docs/api/html/`. Two obsolete
tags (`CLASS_DIAGRAMS`, `DOT_TRANSPARENT`) were removed from
`Doxyfile`; `MARKDOWN_ID_STYLE = GITHUB` was added so the existing
`[Section](#section-id)` links inside `USER_GUIDE.md` resolve
correctly; the three formerly-missing markdown docs (`TUTORIAL.md`,
`PLUGIN_AUTHORING.md`, `PERFORMANCE_TUNING.md`) were added to
`INPUT`. End state: 119 documented classes, 1773 HTML files, **0
warnings** during the doxygen run.

**Perf regression as a hard CI gate** — `tools/perf_check.py` grew
`--cpu-threshold` and `--gpu-threshold` per-kind overrides so the
GPU rows (noisier on Mesa llvmpipe in GitHub-hosted runners) can use
a more lenient ratio without loosening the CPU ratio. The legacy
`--threshold` still works as a default-for-both. The CI workflow
removed `continue-on-error: true` from the perf step and uses
2.0× / 3.0× respectively, so a real CPU-side regression now blocks
PRs. Two convenience CMake targets — `perf-check` and
`perf-update-baseline` — wire the bench + checker together for local
use; `perf-update-baseline` rewrites the committed baseline after an
intentional perf change.

**Image texture maps GUI workflow** — `MaterialEditor` now shows a
64×64 thumbnail of the loaded texture next to a dimensions + memory
size label, both refreshed on selection, browse, and clear. More
substantially, `scene_io` learned about *relative* texture paths:
`serialize_scene` accepts an optional `base_dir`; when given, any
absolute `texture_path` that points inside `base_dir` is rewritten
relative to it. The file wrappers (`save_scene` / `load_scene`)
derive `base_dir` automatically from the scene file's path, so the
common case of "scene + textures live in the same directory" just
works — copy the directory to another machine and the references
still resolve. Paths that escape `base_dir` (e.g. external asset
libraries) are deliberately left absolute. Five regression tests in
`test_scene_io_relative_textures.cpp` cover rewrite, resolve,
portable-across-move, out-of-tree, and the empty-`base_dir`
backward-compat path. Total: **218 tests pass** (213 baseline +
5 new).

**Real-time Vulkan viewport** — the QVulkanWindow path now runs the
F-Rep compute shader for real. `ComputeBlitRenderer` replaces the
old clear-color stub: at `initResources` it emits GLSL from the
current scene, compiles to SPV via the existing
`compile_glsl_to_spv_managed`, and builds a compute pipeline using
the QVulkanWindow's device handles. At `initSwapChainResources` it
allocates a `STORAGE | TRANSFER_SRC` image sized to the swapchain.
At `startNextFrame` it dispatches `⌈w/8⌉ × ⌈h/8⌉` workgroups into
the storage image, then `vkCmdBlitImage`s the result into the
current swapchain image with the canonical layout transition
chain (`UNDEFINED → TRANSFER_DST → PRESENT_SRC_KHR`). Scene
structural changes are detected via the same hash recipe as the
offscreen path and trigger a pipeline rebuild on the next frame.

Three real-hardware bugs (NVIDIA GTX 1050 Ti / Linux) were found
and fixed during validation:

1. **Frozen after first frame.** The compute-only renderer never
   begins a graphics render pass, which left QVulkanWindow's
   internal state machine convinced the application was still
   drawing — so it stopped scheduling further frames. Setting
   `QVulkanWindow::PersistentResources` flag in the window
   constructor fixes this, exactly as documented for compute-based
   ray tracers in the Qt forum. Without this flag, `startNextFrame`
   was being called exactly once before the event loop stalled.

2. **Material / light changes didn't update the view.** The
   pipeline-rebuild decision was driven by
   `FRepNode::structural_hash`, which deliberately captures only
   geometric topology — but `GlslEmitter` bakes albedo, pattern,
   roughness, metallic, and PointLight parameters directly into the
   emitted source. Without those in the hash, a Material tab edit
   silently failed to rebuild. The hash recipe inside the
   `ComputeBlitRenderer` (a private detail, not the project-wide
   `structural_hash`) was widened to include those parameters.
   Cost: ~10-50ms pipeline rebuild per material edit on the hot path,
   which the user perceives as a brief pause.

3. **Mouse drag and wheel did nothing in the viewport.** The
   QVulkanWindow widget naturally doesn't inherit mouse handlers
   from the offscreen `Viewport` class. The fix subclasses
   QVulkanWindow with `mousePressEvent` / `mouseMoveEvent` /
   `mouseReleaseEvent` / `wheelEvent` overrides that drive an
   orbital camera state local to the realtime window. The state is
   lazy-initialised from `scene_->camera()` on the first
   interaction, so a freshly-loaded scene's camera position is
   preserved until the user actually drags.

Layout change: `--realtime` now replaces the offscreen viewport
entirely rather than rendering side-by-side; the offscreen
`Viewport` object stays alive but `hide()`-n because some
`MainWindow` signal sinks (object-picked, render-completed,
screenshot save) still terminate on it. A proper `IViewport`
abstraction is a follow-on item.

One limitation is documented in code: scenes that require storage
buffers for `MeshSDF` or image textures fall through to a
clear-color fallback for now — the offscreen path supports them but
wiring buffer descriptors into the swapchain renderer is a follow-on
item. The offscreen path remains the recommended one for those
scenes.

---

## [Previously Unreleased]

### Fixed — GPU shadow ray and silhouette artefacts

User-reported regression in the GPU compute path: scenes with a
ground plane and objects exhibited dark blob-shaped "holes" on the
floor near the horizon, plus scattered white speckle pixels along
object silhouettes.

Two distinct bugs were responsible:

1. **Shadow over-marching**: `_shadow_ray` used `t += max(d, 0.01)`
   which forced a minimum step of 0.01 units. When the SDF returned
   tiny positive values (ray skimming parallel to a surface),
   forcing the larger step jumped past the actual safe distance and
   read SDF samples deep inside or past nearby geometry — producing
   spurious shadow hits. Fix: pure sphere tracing, `t += d` only.
   Termination eps relaxed from 0.001 → 0.002 to avoid the ray
   self-intersecting its starting surface.

2. **NaN normals at silhouettes**: `scene_normal` did
   `normalize(vec3(dx, dy, dz))` without guarding against the case
   where all three central differences collapse to near-zero — which
   happens at object silhouette edges. The resulting NaN propagated
   through the Cook-Torrance BRDF (`D_GGX`, `F_Schlick`) producing
   infinite specular highlights — the white pixels. Fix: check
   `length(g) < 1e-6` and fall back to an up-vector; plus explicit
   `isnan` check on the final color before `imageStore`.

The performance baseline (`tools/perf_baseline.json`) was updated
to reflect the new (correct) shadow rendering cost.

New regression test: `tests/test_gpu_shadow_regression.cpp` — two
GoogleTest cases that render a known scene and statistically assess
floor-pixel darkness near horizon plus isolated near-white pixel
count, both of which would have caught the original bugs.

### Added — Real-time viewport scaffolding (F4 partial)

- `gui/vulkan_viewport.{hpp,cpp}` — QVulkanWindow-based real-time
  viewport with a runtime device-type probe that distinguishes
  hardware GPUs from software Vulkan (Mesa llvmpipe, swiftshader).
- `--realtime` CLI flag on `frep_designer` opts into the path when
  hardware is detected; on software-only systems it prints a clear
  diagnostic and falls back to the existing offscreen `Viewport`.
- `FREP_REALTIME_VIEWPORT={0,force}` env var for debugging the
  capability probe.
- `MainWindow` constructor accepts an extra `realtime_viewport` bool;
  when true and hardware Vulkan is available, builds a QSplitter
  showing the real-time widget alongside the offscreen viewport
  (which remains the source of the actual rendered scene until the
  compute integration lands).

**Note**: this delivers the architecture and the working fallback,
not the full compute integration. The `ClearRenderer` currently
presents a debug-color clear pass at native refresh — proving the
window + swapchain + present chain compiles and works end-to-end.
The actual F-Rep scene compute dispatch inside `startNextFrame` is
left as documented stubs (initResources/initSwapChainResources)
because completing it requires shared `VkDevice` between
QVulkanWindow and `VulkanCtx`, plus testing on real GPU hardware
that the development sandbox doesn't provide.

### Added — Cross-platform CI

- `.github/workflows/ci.yml` now has Windows (windows-2022) and
  macOS (macos-14, Apple silicon) jobs alongside the existing Linux
  job. Both are marked `continue-on-error: true` as best-effort
  portability indicators — the project is developed on Linux and
  Windows/macOS green is not a release gate.
- Windows job uses chocolatey for LLVM 20 + Vulkan SDK, jurplel
  install-qt-action for Qt 6.6, and vcpkg for GoogleTest + libpng.
- macOS job uses Homebrew for llvm@20, qt@6, vulkan-headers,
  vulkan-loader, and molten-vk (Vulkan→Metal translation).
- Both jobs require the CPU smoke test to pass; the GPU smoke is
  best-effort on macOS via MoltenVK (which doesn't expose all
  features needed by `frep_gpu_render` in CI).

---

## [4.0.0] — 2026-05-23

The first stable release. FRep Designer 4.0 is a complete F-Rep
geometric modeling system built on C++26, LLVM 20, and Vulkan 1.3.

### Added — Core engine

- **CSG primitives**: Sphere, Box, Plane, Torus, RoundedBox.
- **CSG operations**: Union, Difference, Intersection, plus smooth
  variants (SmoothUnion, SmoothDifference, SmoothIntersection) with
  blending parameter `k`.
- **Affine transforms**: Translate, Scale, RotateY, with Jacobian
  propagation so transformed SDFs remain correctly Lipschitz-bounded.
- **Deformations**: TwistY, Bend, Taper, all with Lipschitz
  correction via `/sqrt(1+(k·r)²)`-style scaling so ray marching
  remains conservative.
- **Procedural patterns**: Solid, Checker, Stripes, GradientY, Noise,
  Texture. Triplanar projection for textures, sharpened with
  pow(weight, 4).
- **CustomExpr nodes**: define an SDF as a runtime text expression
  in `x, y, z` with arithmetic, transcendentals, and named constants
  `pi`/`e`. Parsed once into a shared AST consumed by three back-
  ends (LLVM IR, CPU eval interpreter, GLSL emitter).
- **Mesh import**: load `.obj` / `.stl`, build a triangle BVH for
  fast point queries, then voxelize to a 3D SDF grid (`MeshSDFNode`).
- **Sparse octree compression** for voxel SDFs — ~30× smaller for
  surface-sparse meshes at 128³.

### Added — Execution back-ends

- **CPU JIT pipeline**: scene → LLVM IR → x86-64 native via Orc JIT,
  with Cook-Torrance PBR shading (full GGX/Smith/Schlick + shadow
  rays).
- **GPU compute pipeline**: scene → GLSL → SPIR-V via glslangValidator
  → Vulkan compute shader. Multi-light push constants (up to 4
  lights), procedural materials in shader, image textures via storage
  buffers with triplanar projection.
- **AST evaluator**: direct interpretation of CustomExpr nodes for
  use in the picker, marching cubes, and BVH voxelizer — no JIT
  required.
- **AD codegen**: forward-mode automatic differentiation via dual
  numbers, used by central-difference normal computation.
- **Incremental recompilation**: hash-based cache keyed by scene
  structure; material and light edits flow through push constants
  without shader recompile.

### Added — Plugin system

- **Dynamic plugin loading**: `.so`/`.dll` files dropped into one of
  four discovery directories are loaded automatically at GUI start.
  C-ABI entry point (`frep_register_plugin`); plugin sees host's
  `frep_core` symbols via `--whole-archive` + `ENABLE_EXPORTS`.
- **Reference plugin**: `capsule_plugin` ships as a working example,
  with CPU eval, LLVM codegen, GLSL emit, and structural hash.
- **GPU support for plugins** via the `NodeKind::Plugin` fallback in
  the GLSL emitter — plugins emit their own GLSL source, no host
  recompile needed.

### Added — Graphical user interface (Qt 6.4)

- **Viewport** with live GPU compute rendering, async image readback,
  ray-cast object picking.
- **Scene Inspector** tab with object list, per-object parameters,
  visibility toggles, color picker, undo-aware add/remove.
- **Expression editor** tab — multi-line text editor with 6 pre-canned
  samples, live syntax validation (debounced 250 ms), parse errors
  with column positions.
- **Material editor** tab — pattern dropdown, dual color pickers,
  scale/roughness/metallic spin boxes, texture file browser. All
  edits coalesce into atomic undo entries.
- **Lights** tab — add/remove point lights, position spin boxes,
  intensity slider, color picker.
- **Node Graph** tab — visual editing of the FRepNode tree for the
  currently active object. Two-way sync with the scene: edits in
  inspector/toolbar/expression editor propagate into the graph;
  graph edits update only the active object (other scene objects
  preserved). Houdini-style "Editing:" dropdown switches active
  object; "Follow" toggle locks the graph view when desired;
  "Fit" button frames all nodes in view.
- **Undo/redo** through entire system, including node graph edits
  (via new `SetGeometryCommand`).
- **Plugin auto-discovery** searches `./plugins/`, `./build/plugins/`,
  `<exe_dir>/plugins/`, `<exe_dir>/../lib/frep/plugins/`.
- **Mesh export menu** — OBJ/STL output via marching cubes, async
  with progress dialog showing live elapsed time, user-selectable
  resolution (32–256).
- **Mesh import menu** — STL/OBJ → BVH → voxelized SDF, async with
  progress dialog.

### Added — Tooling

- **`frep_bench`** — performance benchmark harness covering 5 scenes
  (Simple/Moderate/Complex/Heavy/CustomExpr) on both CPU and GPU at
  three resolutions. `--json` mode for machine-readable output.
- **`perf_check.py`** — regression detection script comparing
  current run against `tools/perf_baseline.json`; fails CI on >2×
  slowdowns.
- **`frep_gallery`** — renders 7 showcase scenes to PNG.
- **`frep_render` / `frep_gpu_render`** — headless CPU and GPU
  rendering tools with `--scene` and `--out` flags.
- **`frep_designer`** — full Qt GUI with `--scene`, `--empty`, and
  `--plugin-dir` flags.

### Added — Tests and CI

- **211 GoogleTest tests** covering primitives, codegen, transforms,
  deformations, mesh import, sparse SDF, undo, incremental recompile,
  SPIR-V, GPU rendering, patterns, textures, CustomExpr backends,
  scene I/O, mixed-node integration, plugin runtime, scene/graph
  sync.
- **`test_node_graph`** headless logic test (no Qt event loop) for
  graph topology operations.
- **GitHub Actions CI** — Ubuntu 24, full build, all tests, every
  CLI tool's smoke test, every example scene rendered, perf
  regression check.

### Added — Documentation

- **Tutorial** — step-by-step "build your first scene" walkthrough.
- **User guide** — comprehensive reference.
- **Architecture overview** — component map, data flow, three-
  backend AST design.
- **Plugin authoring guide** — complete code example with build
  instructions.
- **Performance tuning guide** — CPU vs GPU decision matrix,
  MeshSDF/marching-cubes resolution guidance.
- **Doxygen API reference** — generated via `cmake --build build
  --target docs`, 453 HTML pages.
- **Examples gallery** — 11 showcase renders documenting features.
- **Benchmark report** — auto-generated performance numbers.

### Added — Example scenes and assets

- 6 hand-editable JSON example scenes covering CSG, deformations,
  smooth union, patterns, carving, textures.
- 2 bundled BMP textures (procedurally generated wood + marble,
  256×256 each) so the textured examples render out of the box.

---

## Key design decisions

### Shared AST for CustomExpr (refactor during 4.0 development)

The runtime-text expression system originally had two independent
parsers — one in `CustomExprCompiler` for LLVM IR, and a separate
recursive-descent interpreter inside `CustomExprNode::eval` for CPU
use. Adding a new function meant editing two lexers, two parsers,
two arity tables, and hoping they stayed in sync. The GLSL backend
was even worse — it just emitted the source text verbatim, leaving
the GPU shader at the mercy of GLSL's own parser.

The 4.0 refactor introduces a single AST (`core/frep/expr_ast.{hpp,cpp}`)
shared by all three back-ends. The grammar lives in exactly one
place. Adding a new function is now: one line in `builtin_arity`
plus three small additions in `custom_expr.cpp` (one per backend).
The AST also gains a constant-folding pass that pre-evaluates pure-
constant subtrees, simplifying downstream code generation.

This is the standard compiler architecture pattern (frontend / IR /
backends) applied at small scale. See ARCHITECTURE.md for the diagram.

### Two-way scene ↔ node graph sync

Earlier iterations had a destructive node graph editor: every edit
rebuilt the scene by `remove_object()`-ing everything and then
`add_object()`-ing a single new tree. This wiped objects the user
had added via the toolbar, inspector, or expression editor.

4.0 introduces a Houdini-style model: the graph edits a single
**active object**. Selection in the inspector switches the active
object; edits in the graph replace only that object's geometry
(via the new `SceneGraph::set_geometry()` API), with all other
scene objects preserved. A `syncing_` flag prevents infinite loops
between the two directions. A "Follow" checkbox lets users lock the
graph view when they want to explore the scene without losing their
editing context.

### Plugin GPU support

Plugins originally only worked on CPU — the GPU shader had a fixed
switch over `NodeKind` and didn't know about plugin nodes. 4.0 adds
the `NodeKind::Plugin` fallback in the GLSL emitter that calls into
the node's `emit_glsl()` virtual; plugins now render on GPU exactly
like built-in nodes.

### Performance regression infrastructure

Without committed baseline numbers, performance regressions creep in
unnoticed. 4.0 ships `tools/perf_baseline.json` (15 CPU + 10 GPU
measurements) and a Python script that compares any new run against
it, exiting non-zero on >2× slowdowns. The CI workflow runs this on
every PR.

---

## Statistics

- **96 source files** (`.hpp` / `.cpp` / `.comp`)
- **22 test files** with **211 GoogleTest tests** + 1 node-graph
  logic test (all passing)
- **6 markdown documentation files** + Doxygen API reference
- **6 example JSON scenes** + 2 bundled textures
- **11 gallery PNGs**
- **10 build artifacts**: `frep_designer`, `frep_render`,
  `frep_advanced`, `frep_gpu_render`, `frep_gallery`, `frep_bench`,
  `frep_tests`, `test_node_graph`, `capsule_plugin.so`,
  `sphere_trace.spv`
- **~872 KB** source tarball, **~12 MB** built workspace

---

## Versioning

This is a single-release project. The 4.x sequence represents the
fourth distinct iteration of the F-Rep designer prototype:

- **1.x** (legacy) — pure CPU, no JIT, single-threaded
- **2.x** (legacy) — added LLVM JIT pipeline
- **3.x** (legacy) — added Qt GUI, mesh import
- **4.0** (this release) — full system as documented above

There is no planned 4.x patch series. Future work (animation
timeline, real-time swapchain viewport, additional primitives) would
roll into a hypothetical 5.0.

---

[4.0.0]: https://github.com/anthropic-quickstarts/frep-designer/releases/tag/v4.0.0
