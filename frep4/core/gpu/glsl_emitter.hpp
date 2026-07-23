#pragma once
// core/gpu/glsl_emitter.hpp
//
// GlslEmitter — recursively walks an FRepNode tree and emits a GLSL
// compute shader that evaluates the same SDF. The emitted shader
// follows the same calling convention as gpu/sphere_trace.comp:
//   - 8×8 workgroup
//   - rgba8 storage image at binding 0
//   - camera + scene params in a push-constant block
//   - one invocation per pixel
//
// Output format: a single string containing a complete compute shader.
// The caller writes it to disk and passes it to glslangValidator (or
// links it through libshaderc later); the resulting SPIR-V plugs into
// the existing VulkanCtx with no further changes.
//
// Supported node kinds:
//   Primitives:  Sphere, Box, Plane
//   CSG:         Union, Intersection, Difference, SmoothUnion, Negate
//   Transforms:  Translate, Scale, RotateY
//   Deforms:     TwistY, BendXY, TaperY
//
// Unsupported (return false from try_emit):
//   MeshSDFNode  — needs voxel grid in a storage buffer; future work
//   CustomExprNode — needs expression-to-GLSL converter; future work
//
// Per-object material is emitted as `select_closer(...)` chains so each
// object can have its own solid albedo. Procedural patterns from the
// CPU side are not emitted yet (they'd compose with the existing
// scene-material function — small future iteration).

#include "core/frep/scene.hpp"
#include "core/frep/node.hpp"
#include "core/compiler/codegen.hpp"   // TracerConfig
#include "core/compiler/param_binding_table.hpp"

#include <array>
#include <expected>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace frep::gpu {

// One scene-translation result.
struct GlslEmitResult {
    std::string source;           // complete .comp shader text
    int         object_count = 0; // how many scene objects went in
    int         expr_lines   = 0; // for diagnostics — lines of emitted SDF

    // ── MeshSDF resources ───────────────────────────────────────────────────
    // If the scene contains MeshSDFNode children, GlslEmitter collects
    // every mesh grid into one big float array which the host needs to
    // upload into a single storage buffer (binding = 1). `mesh_offsets`
    // gives the start index (in floats) of each mesh's grid within the
    // concatenated array; the emitter inserts a constant array of
    // these offsets directly into the GLSL source, so the host only has
    // to upload `mesh_voxels`. If the scene has no MeshSDF nodes, both
    // vectors are empty and the storage buffer is unused.
    std::vector<float> mesh_voxels;
    int                mesh_count = 0;

    // ── Texture resources ───────────────────────────────────────────────────
    // All material textures concatenated into one uint8 array (RGBA8,
    // top-down). The host uploads it to a storage buffer at binding = 2,
    // and each per-object texture sample emits an indexed read into it
    // with constant offset / width / height baked into the shader.
    std::vector<std::uint8_t> texture_pixels;
    int                       texture_count = 0;

    // ── Runtime parameter buffer ────────────────────────────────────────────
    // Non-empty when emit() was given a ParamBindingTable. Each entry is a
    // model parameter the policy placed Runtime; the shader reads it from the
    // std430 `Params` buffer at binding = 3 (P.v[slot]) instead of baking a
    // literal. The host uploads `bindings.size()` floats (seeded with each
    // default_value) and overwrites individual slots on a parameter edit, with
    // NO shader re-emit/recompile. Identical layout to the CPU/GPU-IR buffer.
    std::vector<ParamSlot> param_bindings;
    std::size_t            placement_hash = 0;  // 0 when no runtime params
};

class GlslEmitter {
public:
    // Translate the visible objects of `scene` into a GLSL compute shader.
    // Returns the shader text on success, or a string describing the
    // first node type that has no GLSL emitter yet.
    //
    // `cfg` controls the shading model, shadow / AO toggles, and their
    // respective tuning floats — mirroring the CPU JIT path (codegen.cpp)
    // so both backends honour Render-tab changes identically. The
    // emitter bakes the selected configuration into the shader source,
    // so any change to `cfg` requires a re-emit + re-compile of the
    // SPIR-V (which the higher-level viewport handles via its scene-
    // hash detection — see `IncrementalCompiler::structural_hash` and
    // the corresponding GPU rebuild gate in vulkan_viewport.cpp).
    // `bindings`, when non-null, places some model parameters in a runtime
    // buffer (P.v[slot]) instead of baking literals, so editing those values
    // needs only a buffer re-upload, not a re-emit. When null (default) every
    // parameter is baked — bit-identical to the previous behaviour.
    static std::expected<GlslEmitResult, std::string>
    emit(const SceneGraph& scene, const TracerConfig& cfg = {},
         const ParamBindingTable* bindings = nullptr);

private:
    // Aggregate state shared across all per-object emit() calls. Holds
    // the running list of MeshSDF voxel grids and metadata that the
    // outer emit() pass turns into a single storage buffer at the end.
    struct MeshMeta {
        int        offset;
        int        res;
        int        gpu_index;   // 0, 1, 2... — used in sample_mesh<i> name
        float      bmin[3];
        float      bmax[3];
        float      cell[3];
    };
    struct MeshAccum {
        std::vector<MeshMeta>  meshes;
        std::vector<float>     mesh_voxels;
        // Dedup: the SDF body and the albedo body each emit every node once, so
        // a MeshSDFNode is visited twice. Map the node pointer to its already-
        // allocated gpu_index so the second visit reuses the slot + voxels
        // instead of appending a duplicate copy (which doubled mesh_voxels and
        // the sample-function count).
        std::unordered_map<const void*, int> node_to_index;
    };

    // Level 2 instancing: shared subprograms. A geometry subtree referenced by
    // one or more InstanceNodes (and by its original object) is emitted as a GLSL
    // function ONCE and called, instead of inlined at each use — so the emitted
    // code, and the cost of recompiling it, grow with the number of *distinct*
    // shapes, not the total instance count (the point of instancing for large
    // repetitive models like a detector geometry). Dedup is by the shared node
    // pointer: the InstanceNode's children[0] is the very same pointer as the
    // target object's geometry root, so both map to the same function.
    struct InstanceFuncs {
        std::ostringstream defs;                       // emitted SDF function definitions
        std::unordered_map<const void*, std::string> ptr_to_fn;  // target root -> SDF fn name
        std::ostringstream grad_defs;                  // dual-number AD function defs
        std::unordered_map<const void*, std::string> ptr_to_grad_fn;  // target -> grad fn name
        int next_fn = 0;
        int next_grad_fn = 0;
        // Which target roots are worth functionalising: a subtree only becomes a
        // function if at least one InstanceNode references it (a plain object with
        // no instances stays inlined, avoiding a needless call). Filled by a
        // pre-pass over the scene before emit.
        std::unordered_set<const void*> shared_targets;
    };

    // Per-call state held on the stack during emit_node().
    struct Ctx {
        std::ostringstream sdf_body;
        std::ostringstream albedo_body;
        std::ostringstream grad_body;   // parallel dual-number AD body
        int                next_var = 0;
        int                next_dvar = 0;
        MeshAccum*         mesh_accum = nullptr;  // borrowed
        InstanceFuncs*     inst_funcs = nullptr;  // borrowed (Level 2 instancing)
        // Shared slot authority (borrowed). When set, a parameter the policy
        // placed Runtime is read from P.v[slot]; otherwise it is baked.
        const ParamBindingTable* bindings = nullptr;
        std::string fresh(const char* prefix = "v") {
            return std::string(prefix) + std::to_string(next_var++);
        }
        std::string fresh_d(const char* prefix = "d") {
            return std::string(prefix) + std::to_string(next_dvar++);
        }
    };

    // The parameter choke point shared by the SDF and dual-AD paths: returns a
    // GLSL rvalue for a node parameter — a baked literal when it is Constant
    // (or there is no binding table), or `P.v[slot]` when it is Runtime. All
    // node emitters route parameter access through this so the same placement
    // governs both backends and the runtime-buffer layout matches the CPU path.
    static std::string pval(Ctx& c, const FRepNode& n, const char* name);

    // Level 2 instancing: emit the given target subtree as a GLSL function into
    // c.inst_funcs->defs (once, deduplicated by node pointer) and return its
    // name, e.g. "_inst_fn_0". Called by the Instance dispatch and by the object
    // loop when an object is itself an instance target, so both use one body.
    static std::string emit_instance_fn(Ctx& c, const FRepNode& target);

    // Dual-number AD twin of emit_instance_fn: emit the target as
    // `Dual _inst_grad_fn_N(vec3 p)` once and return its name. Lets the gradient
    // (normal) body share one body per shape instead of inlining the dual code
    // at every instance — the gradient body is the largest per-object emission,
    // so this is where most of the instancing memory saving comes from.
    static std::string emit_instance_grad_fn(Ctx& c, const FRepNode& target);

    // Recursive emitter for the geometry of a single object. Returns the
    // variable name holding the SDF value at (x, y, z), or an error.
    static std::expected<std::string, std::string>
    emit_node(Ctx& c, const FRepNode& node,
              const std::string& x, const std::string& y, const std::string& z);

    // Parallel dual-number AD emitter: returns the name of a `Dual` holding
    // both the SDF value and its analytic gradient at the dual coords
    // (x, y, z). Mirrors emit_node's structure but in dual arithmetic, so
    // scene_normal can use an exact analytic normal (normalize(grad)) instead
    // of finite differences. The inputs are Dual variable names.
    static std::expected<std::string, std::string>
    emit_node_dual(Ctx& c, const FRepNode& node,
                   const std::string& x, const std::string& y,
                   const std::string& z);

    // Helpers for specific node kinds.
    static std::string emit_sphere   (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_box      (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_plane    (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);

    static std::string emit_translate(Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_scale    (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_rotate_y (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_rotate_x (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_rotate_z (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);

    static std::string emit_twist_y  (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_bend_xy  (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
    static std::string emit_taper_y  (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);

    // MeshSDFNode → call a per-mesh sample function that does trilinear
    // interpolation against the global voxel storage buffer.
    static std::string emit_mesh_sdf (Ctx& c, const FRepNode& n,
        const std::string& x, const std::string& y, const std::string& z);
};

} // namespace frep::gpu
