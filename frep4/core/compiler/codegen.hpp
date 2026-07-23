#pragma once
// core/compiler/codegen.hpp

#include "core/frep/node.hpp"
#include "core/frep/scene.hpp"
#include "core/compiler/compile_policy.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace frep {

// Sphere tracing parameters
struct TracerConfig {
    int   max_steps     = 512;     // raymarch iteration cap. Higher lets
                                   // grazing silhouette rays converge to
                                   // epsilon instead of exhausting steps
                                   // (which the grazing-rescue then has to
                                   // paper over). Measured on the horizon.json
                                   // scene (camera grazing the floor): 192 left
                                   // ~540 hole pixels along the horizon line and
                                   // object silhouettes; 384 cleared them all.
                                   // 512 keeps a margin for steeper grazing.
                                   // The cost is paid only by grazing rays —
                                   // interior rays still converge in a handful
                                   // of steps.
    float max_dist      = 100.0f;
    float epsilon       = 0.0005f;
    float fd_eps        = 0.001f;   // finite-diff epsilon for the normal
    float safety_factor = 0.85f;    // reduced step for non-true SDF (CSG)

    // ── GPU workgroup tile cull ─────────────────────────────────────────────
    // Before marching, each 8x8 workgroup bounds its own frustum slab stack and
    // discards the depth slabs that provably hold no surface, using the SDF's
    // 1-Lipschitz invariant: over a box of circumradius r around c, f lies in
    // [f(c) - L*r, f(c) + L*r]. The march then runs only over the surviving
    // depth span (or the pixel takes the miss path outright).
    //
    // Costs `cull_slabs` scalar SDF evaluations per workgroup, amortised over
    // its 64 pixels. 0 disables the pass. `cull_lipschitz` must upper-bound
    // |grad f|: 1.0 is guaranteed by the node contract, but a CustomExpr that
    // is a raw implicit rather than a distance field needs its own value.
    int   cull_slabs     = 0;
    float cull_lipschitz = 1.0f;

    // Which box bound the cull uses. Both live in the codebase; this only
    // selects between them (soundness is each method's own responsibility):
    //   Lipschitz — f(box) in [f(c)-L*r, f(c)+L*r]; cheap, needs L >= |grad f|.
    //               L=1 is exact for a true SDF (metric primitives + affine/CSG);
    //               a raw implicit needs its real L or it culls real surface.
    //   Interval  — interval arithmetic on the field; sound by construction with
    //               no L, and tighter on periodic/implicit fields (gyroid, gears),
    //               but currently only available for a single-CustomExpr scene.
    //   Auto      — pick per scene: a true-SDF node tree -> Lipschitz(L=1);
    //               otherwise, if an interval path exists and prunes more on a
    //               coarse probe, Interval; else Lipschitz with an estimated L.
    //   Off       — no cull (identical to cull_slabs = 0).
    enum class CullMethod { Auto, Lipschitz, Interval, Off };
    CullMethod cull_method = CullMethod::Auto;

    // Debug visualisation (task 3). Off = normal shaded render. StepHeatmap
    // colours each pixel by how many march iterations it took (blue = few,
    // red = many) so expensive regions are visible. CullSpan colours by the
    // fraction of the ray the tile cull kept (green = most culled, red = little),
    // showing where the cull is effective. Both replace the shaded output.
    enum class DebugView { Off, StepHeatmap, CullSpan };
    DebugView debug_view = DebugView::Off;

    // Level 2 instancing: emit each instanced geometry as a shared GLSL function
    // called by the original and its instances, instead of inlining a copy at
    // each use. Shrinks the emitted code (and its recompile cost) for models with
    // many repeated shapes. On by default; can be turned off to force inlining
    // (Level 1), e.g. to trade code size for avoiding the function-call overhead
    // or the sub-pixel silhouette FP differences inlining vs a call can produce.
    bool instance_shared_subprograms = true;

    // Ray-box near/far clip for the IR raymarch paths (CpuIr/GpuIr). When on and
    // the scene is bounded, camera rays are clipped to the scene AABB before
    // marching, skipping empty space before/after the scene. The GLSL path has
    // its own richer tile cull; this is the IR-path equivalent for the common
    // "small object in a large view" case. Off falls back to marching from the
    // near plane to max_dist.
    bool bbox_clip = true;

    // Future work: when true and cull_method is Auto, the GPU executor selects
    // between Lipschitz and Interval by *timing* a few frames of each rather than
    // by topology (see the probe note in compile_sdf.hpp). Off by default — the
    // timed probe costs real frames and only pays off on scenes whose wall-clock
    // spread between methods is large. Left as a wired opt-in for when scene data
    // justifies enabling it; today Auto is topology-based and this is ignored.
    bool cull_auto_timed_probe = false;
    // Over-relaxation factor for enhanced sphere tracing. DISABLED (1.0 =
    // classic sphere tracing): empirically, omega>1 over-steps the floor plane
    // seen at grazing angle (the horizon), so the overshoot guard backtracks and
    // the wasted iterations make grazing horizon rays exhaust max_steps *more*,
    // widening the horizon "holes" at object boundaries — worse than classic on
    // floor-heavy scenes. Kept as a knob (the march still consults it) but left
    // at 1.0; the horizon is addressed by max_steps + the grazing-rescue instead.
    float over_relax    = 1.0f;

    // Shading model:
    //   BlinnPhong — fast, plausible specular with metallic tinting.
    //   CookTorrance — physically-based GGX microfacet BRDF.
    // Both honour the per-material roughness and metallic fields.
    enum class ShadingModel { BlinnPhong, CookTorrance };
    ShadingModel shading_model = ShadingModel::CookTorrance;

    // Specular term toggle. Primarily a diagnostic to isolate whether a
    // CPU-vs-GLSL divergence lives in the specular or the diffuse path;
    // when false, only diffuse + ambient are shaded.
    bool  enable_specular = true;

    // Shadow rays
    bool  enable_shadows  = true;
    int   shadow_steps    = 64;
    float shadow_softness = 16.0f;  // higher = sharper shadows

    // Ambient occlusion
    bool  enable_ao    = true;
    int   ao_samples   = 5;
    float ao_step      = 0.15f;
    float ao_strength  = 0.6f;

    // Sky / background gradient. Two colours are mixed by the ray's
    // vertical component (s = 0.5 + 0.5 * dir.y). `sky_top` is what
    // a ray pointing straight up sees; `sky_horizon` is what a
    // horizontal ray sees. The CPU JIT path and both GPU paths read
    // these from cfg and bake them into the emitted code, so a
    // Render-tab edit to either colour forces a rebuild.
    float sky_top[3]     = {0.4f, 0.5f, 0.7f};
    float sky_horizon[3] = {0.6f, 0.7f, 0.85f};

    // ── Reflections ────────────────────────────────────────────────────────
    // Max recursion depth for mirror reflection rays. 0 disables
    // reflections entirely (the per-material `reflectivity` field is
    // then ignored and no secondary rays are cast). 1 = single bounce
    // (object reflects the environment + other objects but not
    // reflections-of-reflections). 2+ allows mirror-in-mirror. Each
    // bounce roughly doubles trace cost, so we cap the UI at 4.
    int   max_bounces = 0;

    // ── Soft-shadow quality ──────────────────────────────────────────────────
    // Number of jittered shadow samples per light. 1 = the classic
    // single-ray hard/penumbra shadow (cheapest). Higher values cast
    // multiple rays toward slightly offset points on a virtual area
    // light of half-extent `shadow_light_radius`, averaging the
    // results — producing physically-softer penumbrae at the cost of
    // N shadow traces per light. UI-capped at 16.
    int   shadow_samples      = 1;
    float shadow_light_radius = 0.3f;

    // ── Denoising / temporal accumulation ────────────────────────────────────
    // Real-time only. When > 1, consecutive frames whose camera and
    // scene haven't moved are blended into an accumulation buffer,
    // averaging out the noise from jittered soft shadows / AO. The
    // value is the max number of frames to accumulate before the
    // estimate is considered converged. 1 disables accumulation
    // (every frame stands alone). Camera movement resets the
    // accumulator. Has no effect on the offscreen paths, which
    // already render a single clean frame.
    int   accum_frames = 1;

    // Compilation mode for node parameters.
    //   Constant: parameters are baked as IR constants (full O3 folding).
    //             Best per-render performance, but every parameter change
    //             requires a full recompile (~75 ms).
    //   Incremental: parameters are loaded from a runtime float buffer.
    //                Slider edits update the buffer (no recompile, ~0 ms),
    //                but the render itself is 1.5-2x slower because O3
    //                cannot constant-fold parameter values.
    //   Auto: a runtime heuristic switches between the two — see
    //         IncrementalCompilerAuto for the policy. Plain TracerConfig
    //         users that pick Auto are treated as Constant for the
    //         purposes of this single emission; the policy lives in the
    //         higher-level compiler wrapper.
    enum class CompileMode { Constant, Incremental, Auto };
    CompileMode compile_mode = CompileMode::Constant;

    // Incremental parameter mode. When true, the emitted render_tile takes
    // an extra `float* params_buffer` argument; node parameters (Sphere::r,
    // Box dims, Plane normal/d, Translate offsets, Scale s, RotateY a) are
    // loaded from the buffer at runtime instead of baked as IR constants.
    // The host-side update path can change values in the buffer and re-run
    // render_tile without recompiling.
    //
    // Trade-off: O3 loses constant propagation on those parameters, so a
    // single render is 1.5–2× slower. Interactive editing wins big.
    //
    // This field is the low-level toggle wired into emit_*. Typical users
    // set `compile_mode` and let SceneCodegen translate that into the
    // boolean (Constant=false, Incremental=true; Auto handled externally).
    bool  incremental_params = false;
};

// SceneCodegen — generates a single LLVM module containing:
//   scene_sdf(x,y,z)         -> float
//   scene_normal(x,y,z,*nx,*ny,*nz)  -> void
//   shade_pixel(*nx,ny,nz, lx,ly,lz, albR,albG,albB, *r,*g,*b) -> void
//   render_tile(out, tx,ty,tw,th,iw,ih, camOx,Oy,Oz, Dx,Dy,Dz, Rx,Ry,Rz, Ux,Uy,Uz, fovScale) -> void
//
// All internal functions are AlwaysInline → O3 inlines them into render_tile.
class SceneCodegen {
public:
    explicit SceneCodegen(llvm::LLVMContext& ctx,
                          TracerConfig       cfg = {},
                          std::string        name = "frep");

    // Main method: generates render_tile from a scene.
    // How the march-loop scene_sdf is built. Inlined is the default
    // (whole object tree in one function). Split and Guarded are the
    // scalability experiments: Split = N non-inlined per-object functions
    // folded with min(); Guarded = those functions gated by an AABB
    // distance test (the build-time spatial prune, BVH approach 1).
    enum class SceneSdfMode { Inlined, Split, Guarded };

    llvm::Function* emit_render_tile(const SceneGraph& scene,
                                     SceneSdfMode mode = SceneSdfMode::Inlined);

    // Approach B — CPU vector (W-wide) render_tile emitted in IR. Same signature
    // as emit_render_tile; selected behind FREP4_VEC_RENDER. Stage 1: masked
    // march + central-difference normals + single-light Lambert.
    llvm::Function* emit_render_tile_vec(const SceneGraph& scene,
                                         SceneSdfMode mode = SceneSdfMode::Inlined,
                                         unsigned W = 8);

    // Emit a per-pixel GPU compute kernel ("render_tile") for the CUDA/NVPTX
    // path. It reads the pixel from the thread/block id (NVPTX sreg
    // intrinsics), maps thread → one pixel, and reuses the full raymarch by
    // calling the existing per-tile renderer with a 1×1 tile at that pixel.
    // This replaces the single-thread tile loop with a W×H grid for real GPU
    // parallelism. The emitted kernel keeps the same name + ABI as the CPU
    // render_tile, so CudaCtx launches it unchanged (just with a 2D grid).
    llvm::Function* emit_gpu_kernel(const SceneGraph& scene,
                                    SceneSdfMode mode = SceneSdfMode::Inlined);

    // Access to individual steps (for tests).
    llvm::Function* emit_scene_sdf      (const FRepNode& root);

    // Diagnostic / scalability variant of emit_scene_sdf. Instead of
    // inlining the whole object tree into one function, it emits each
    // top-level object as its own *non*-inlined function and makes
    // scene_sdf call them and fold with min(). This trades per-object
    // inlining for many small functions, which LLVM compiles in roughly
    // linear total time rather than the super-linear cost of optimising
    // one giant function. `objects` are the per-object geometry roots (as
    // emit_render_tile collects them before union_all). Used by the
    // benchmark's --func-split diagnostic to measure the compile-time
    // difference; may become the default path for large scenes.
    llvm::Function* emit_scene_sdf_split(const std::vector<FRepNode::Ptr>& objects);

    // Build-time spatial guard variant (BVH approach-1 prototype). Emits
    // per-object functions gated by an inline AABB-distance test against
    // the running best distance — the flat prune materialised in code,
    // valid on JIT and GLSL paths. See the definition for the invariant.
    llvm::Function* emit_scene_sdf_guarded(const std::vector<FRepNode::Ptr>& objects);

    // Forward-mode AD version of scene_sdf.
    // float scene_sdf_grad(float x_val, float x_dot,
    //                      float y_val, float y_dot,
    //                      float z_val, float z_dot,
    //                      float* out_dot)  -> returns val, writes dot to out_dot
    //
    // Called 3 times by scene_normal with tangents (1,0,0), (0,1,0), (0,0,1).
    llvm::Function* emit_scene_sdf_grad (const FRepNode& root);

    llvm::Function* emit_scene_normal   (llvm::Function* sdf_grad_fn);
    llvm::Function* emit_shader         ();

    // Returns the albedo of the nearest object at point (x,y,z) — for multi-material.
    // Calls the sdf of each object and picks the min.
    llvm::Function* emit_scene_material (const SceneGraph& scene,
                                         llvm::Function* sdf_fn = nullptr);

    // Returns the mirror reflectivity [0,1] of the nearest object at
    // (x,y,z). Compact analogue of emit_scene_material that tracks only
    // the reflectivity scalar. Emitted only when cfg_.max_bounces > 0;
    // used by emit_tracer's reflection bounce loop (CPU parity with the
    // GLSL emitter's scene_reflectivity). signature:
    //   (x, y, z, params) -> float
    // NOTE: implementation in progress — see CPU reflections work.
    llvm::Function* emit_scene_reflectivity(const SceneGraph& scene);

    // Shades a single hit point and writes the resulting RGB into the
    // out pointers. Factors the per-hit shading (normal fetch, material
    // fetch, multi-light loop with shadows, AO, ambient) out of
    // emit_tracer so it can be reused by the reflection bounce loop.
    // signature:
    //   shade_hit(hx,hy,hz, vx,vy,vz, out_r,out_g,out_b, params) -> void
    // where (vx,vy,vz) is the view direction (toward the camera/eye).
    llvm::Function* emit_shade_hit(const SceneGraph& scene,
                                   llvm::Function* normal_fn,
                                   llvm::Function* shader_fn,
                                   llvm::Function* mat_fn,
                                   llvm::Function* shadow_fn,
                                   llvm::Function* ao_fn);

    llvm::Function* emit_tracer         (const SceneGraph& scene,
                                         llvm::Function* sdf_fn,
                                         llvm::Function* normal_fn,
                                         llvm::Function* shader_fn,
                                         llvm::Function* mat_fn);

    // Emits shadow_factor(ox,oy,oz, dx,dy,dz) -> float in [0,1].
    // Soft shadow via the Inigo Quilez "k-soft" formula in sphere tracing.
    llvm::Function* emit_shadow         (llvm::Function* sdf_fn);

    // Emits ao(px,py,pz, nx,ny,nz) -> float in [0,1] (1 = no occlusion).
    // Samples the SDF at N points along the normal -> how "dense" the surroundings are.
    llvm::Function* emit_ao             (llvm::Function* sdf_fn);

    // Emits scene_pick(ox,oy,oz, dx,dy,dz) -> int32 object index.
    // Casts a ray, sphere-traces, and returns the index of the hit object
    // (in scene traversal order). Returns -1 if the ray hits nothing.
    // Used for interactive selection in the GUI viewport.
    llvm::Function* emit_scene_pick     (const SceneGraph& scene);

    std::unique_ptr<llvm::Module> take_module() { return std::move(mod_); }
    llvm::Module* module() { return mod_.get(); }

    // ── Incremental parameter binding ───────────────────────────────────────
    // Populated during codegen when cfg_.incremental_params is true. Each
    // entry maps (node_id, param_name) → buffer index, recording the
    // default value so the host can seed the runtime buffer.
    struct ParamBinding {
        std::string node_id;
        std::string param_name;
        int         slot;
        float       default_value;
    };
    const std::vector<ParamBinding>& param_bindings() const { return param_bindings_; }
    // Convenience: number of slots = max(slot+1) in param_bindings_, or 0.
    int param_slot_count() const {
        int n = 0;
        for (const auto& b : param_bindings_) n = std::max(n, b.slot + 1);
        return n;
    }

    // Flat RGBA8 buffer of all textured materials' pixels, built during
    // emit_scene_material and embedded as a module constant. Exposed so an
    // executor can confirm/inspect; the JIT'd code reads its own copy.
    const std::vector<std::uint8_t>& texture_pixels() const { return texture_pixels_; }
    bool has_textures() const { return !texture_pixels_.empty(); }

private:
    llvm::LLVMContext&            ctx_;
    TracerConfig                  cfg_;
    std::unique_ptr<llvm::Module> mod_;

    // Slot table for incremental params. The (node_id, param_name) → slot
    // assignment is built lazily during codegen; the map keeps the
    // assignment stable across multiple lookups of the same key.
    std::vector<ParamBinding>                          param_bindings_;
    std::unordered_map<std::string, int>               param_slot_by_key_;
    std::vector<std::uint8_t>                          texture_pixels_;
    // Instancing Level 2: shared LLVM function per instance target, memoised by
    // target node pointer so N instances of one shape emit one function called N
    // times instead of N inline copies. Reset per module emission.
    std::unordered_map<const FRepNode*, llvm::Function*> instance_fn_by_target_;
    std::unordered_map<const FRepNode*, llvm::Function*> instance_grad_fn_by_target_;
    // Compile policy: decides per-parameter constant vs runtime placement.
    // When null, every parameter that reaches acquire_param_slot gets a
    // runtime slot (the previous all-incremental behaviour). Set via
    // set_compile_policy() before emitting.
    const CompilePolicy* policy_ = nullptr;

public:
    // Install a compile policy (not owned; must outlive codegen). Without
    // one, incremental mode makes every supported parameter runtime.
    void set_compile_policy(const CompilePolicy* p) { policy_ = p; }

private:
    // Acquires (or returns existing) slot for a given (node_id, param) pair.
    // Returns -1 when the compile policy places this parameter as a constant.
    int acquire_param_slot(const std::string& node_id,
                           const std::string& param_name,
                           float default_value,
                           int param_class = 0);

    // Instancing Level 2: return a CreateCall to the target's shared SDF
    // function, creating that function on first use (memoised by target
    // pointer). `caller` is the builder emitting the call site; `params` is the
    // params buffer threaded through so the shared body can read runtime params.
    llvm::Value* acquire_instance_fn(const FRepNode* target,
                                     llvm::IRBuilder<>& caller,
                                     llvm::Value* x, llvm::Value* y, llvm::Value* z,
                                     llvm::Value* params);

    // Level 2 for the AD path: shared gradient function. Fills out_val/out_dot
    // via a call to the target's shared inst_grad function (created on first
    // use). Returns true on success.
    bool acquire_instance_grad_fn(const FRepNode* target,
                                  llvm::IRBuilder<>& caller,
                                  llvm::Value* xv, llvm::Value* xd,
                                  llvm::Value* yv, llvm::Value* yd,
                                  llvm::Value* zv, llvm::Value* zd,
                                  llvm::Value* params,
                                  llvm::Value*& out_val, llvm::Value*& out_dot);

    // Types
    llvm::Type*  f32()  { return llvm::Type::getFloatTy(ctx_); }
    llvm::Type*  i32()  { return llvm::Type::getInt32Ty(ctx_); }
    llvm::Type*  b_i8t(){ return llvm::Type::getInt8Ty(ctx_); }
    llvm::Type*  vd()   { return llvm::Type::getVoidTy(ctx_); }
    // float* — opaque pointer in LLVM 17+
    llvm::Type*  fptr() { return llvm::PointerType::getUnqual(ctx_); }

    llvm::Value* fc(llvm::IRBuilder<>& b, float v) {
        return llvm::ConstantFP::get(f32(), v);
    }
    llvm::Value* ic(llvm::IRBuilder<>& b, int v) {
        return llvm::ConstantInt::get(i32(), v, true);
    }

    // Builds a CgCtx pre-configured with the incremental-params machinery.
    // Pass a non-null params_buffer for the SDF / SDF-grad emission inside
    // render_tile; pass nullptr for stand-alone emit_scene_sdf calls (which
    // means parameters fall back to constants even in Incremental mode).
    CgCtx make_cgctx(llvm::IRBuilder<>& b, llvm::Value* params_buffer);

    // Alloca in the entry block (invariant: must be there for mem2reg).
    llvm::AllocaInst* entry_alloca(llvm::Function* fn, const std::string& name);
};

} // namespace frep
