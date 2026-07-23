#pragma once
// core/compiler/picker.hpp
//
// ScenePicker — ray-cast object selection.
//
// Compiles the scene_pick function (JIT) and provides a convenient interface:
// give me a pixel (px,py) + camera → returns the id of the object under it.
//
// Used by the GUI viewport: on a mouse click we generate a camera ray for
// the corresponding pixel, call pick(), and highlight the result in the
// scene inspector.

#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/frep/scene.hpp"

#include <cmath>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace frep {

class ScenePicker {
public:
    // Compiles scene_pick for a given scene. Must be called again when the
    // geometry changes (new/removed objects).
    std::expected<void, std::string>
    rebuild(const SceneGraph& scene) {
        // Keep the order of the visible objects — scene_pick returns an index
        // in exactly this order.
        index_to_id_.clear();
        for (const auto& [id, obj] : scene.objects())
            if (obj.visible)
                index_to_id_.push_back(id);

        if (index_to_id_.empty()) {
            pick_fn_ = nullptr;
            return {};   // empty scene — pick will always return nullopt
        }

        auto ctx = std::make_unique<llvm::LLVMContext>();
        SceneCodegen cg(*ctx);
        cg.emit_scene_pick(scene);
        auto mod = cg.take_module();

        jit_ = std::make_unique<JitEngine>();
        auto fn_or = jit_->load_as<ScenePickFn>(
            std::move(mod), std::move(ctx), "scene_pick");
        if (!fn_or)
            return std::unexpected(fn_or.error());

        pick_fn_   = *fn_or;
        scene_hash_ = scene.scene_hash();
        return {};
    }

    // Casts a ray from the camera through pixel (px,py) in an iw x ih image.
    // Returns the id of the hit object, or nullopt on a miss / empty scene.
    std::optional<std::string>
    pick_pixel(const Camera& cam, int px, int py, int iw, int ih) const {
        if (!pick_fn_) return std::nullopt;

        // Generate the camera ray — the same math as in emit_tracer.
        // Forward / right / up basis.
        float fx = cam.target[0] - cam.position[0];
        float fy = cam.target[1] - cam.position[1];
        float fz = cam.target[2] - cam.position[2];
        normalize(fx, fy, fz);

        // right = forward × up
        float rx, ry, rz;
        cross(fx, fy, fz, cam.up[0], cam.up[1], cam.up[2], rx, ry, rz);
        normalize(rx, ry, rz);

        // up = right × forward
        float ux, uy, uz;
        cross(rx, ry, rz, fx, fy, fz, ux, uy, uz);

        float aspect    = static_cast<float>(iw) / static_cast<float>(ih);
        float fov_scale = std::tan(cam.fov_deg * 0.5f * 3.14159265f / 180.0f);

        // Normalized pixel coordinates in [-1, 1]
        float ndc_x = (2.0f * (static_cast<float>(px) + 0.5f) / iw - 1.0f)
                      * aspect * fov_scale;
        float ndc_y = (1.0f - 2.0f * (static_cast<float>(py) + 0.5f) / ih)
                      * fov_scale;

        // Ray direction
        float dx = fx + ndc_x * rx + ndc_y * ux;
        float dy = fy + ndc_x * ry + ndc_y * uy;
        float dz = fz + ndc_x * rz + ndc_y * uz;
        normalize(dx, dy, dz);

        int idx = pick_fn_(cam.position[0], cam.position[1], cam.position[2],
                           dx, dy, dz);
        if (idx < 0 || idx >= static_cast<int>(index_to_id_.size()))
            return std::nullopt;
        return index_to_id_[static_cast<std::size_t>(idx)];
    }

    // Checks whether the picker is valid for a given scene (by hash).
    bool valid_for(const SceneGraph& scene) const {
        return pick_fn_ != nullptr && scene_hash_ == scene.scene_hash();
    }

    std::size_t object_count() const { return index_to_id_.size(); }

private:
    std::unique_ptr<JitEngine> jit_;
    ScenePickFn                pick_fn_   = nullptr;
    std::size_t                scene_hash_ = 0;
    std::vector<std::string>   index_to_id_;

    static void normalize(float& x, float& y, float& z) {
        float len = std::sqrt(x * x + y * y + z * z);
        if (len > 1e-8f) { x /= len; y /= len; z /= len; }
    }
    static void cross(float ax, float ay, float az,
                      float bx, float by, float bz,
                      float& cx, float& cy, float& cz) {
        cx = ay * bz - az * by;
        cy = az * bx - ax * bz;
        cz = ax * by - ay * bx;
    }
};

} // namespace frep
