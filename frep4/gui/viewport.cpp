// gui/viewport.cpp

#include "viewport.hpp"

#include "core/compiler/incremental.hpp"
#include "core/compiler/picker.hpp"
#include "core/frep/scene.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#ifdef FREP_HAS_CUDA
#include "core/gpu/cuda_ctx.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/retarget_nvptx.hpp"
#include "core/compiler/incremental_params.hpp"
#include "core/compiler/compile_policy.hpp"
#include <llvm/Support/TargetSelect.h>
#include <mutex>
#endif

#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWheelEvent>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <functional>
#include <cstdint>

namespace frep::gui {

Viewport::Viewport(SceneGraph* scene, QWidget* parent)
    : QWidget(parent)
    , scene_(scene)
    , compiler_(std::make_unique<IncrementalCompiler>())
    , picker_(std::make_unique<ScenePicker>())
{
    setMinimumSize(640, 480);
    setMouseTracking(true);
    recompute_camera();

    render_timer_ = new QTimer(this);
    connect(render_timer_, &QTimer::timeout, this, &Viewport::on_render_tick);
    render_timer_->start(33); // ~30 FPS poll
}

Viewport::~Viewport() = default;

IncrementalCompiler& Viewport::compiler() { return *compiler_; }

void Viewport::set_ssaa(int n) {
    n = std::max(1, std::min(4, n));
    if (n == ssaa_) return;
    ssaa_ = n;
    invalidate();
}

void Viewport::invalidate() { dirty_ = true; }

void Viewport::paintEvent(QPaintEvent*) {
    QPainter p(this);
    if (image_.isNull()) {
        p.fillRect(rect(), Qt::darkGray);
        p.setPen(Qt::white);
        p.drawText(rect(), Qt::AlignCenter, "Waiting for render...");
        return;
    }
    p.drawImage(rect(), image_);
}

void Viewport::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        dragging_ = true;
        moved_since_press_ = false;
        last_pos_  = e->pos();
        press_pos_ = e->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void Viewport::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        bool was_dragging = dragging_;
        dragging_ = false;
        setCursor(Qt::ArrowCursor);
        // If the mouse barely moved between press and release →
        // it is a click, not a drag → attempt a pick.
        if (!moved_since_press_)
            try_pick(e->pos());
        // Force a fresh render at full resolution now that the user is
        // no longer dragging — the preview rendering only kicks in
        // while dragging_ is true.
        if (was_dragging) dirty_ = true;
    }
}

void Viewport::mouseMoveEvent(QMouseEvent* e) {
    if (!dragging_) return;
    auto delta = e->pos() - last_pos_;
    last_pos_ = e->pos();
    // 3px threshold — below it we still count as a click (avoids jitter on click).
    if ((e->pos() - press_pos_).manhattanLength() > 3)
        moved_since_press_ = true;
    cam_yaw_   -= delta.x() * cam_cfg_.mouse_sensitivity;
    cam_pitch_ -= delta.y() * cam_cfg_.mouse_sensitivity;
    cam_pitch_ = std::clamp(cam_pitch_, -cam_cfg_.max_pitch, cam_cfg_.max_pitch);
    recompute_camera();
    dirty_ = true;
}

void Viewport::wheelEvent(QWheelEvent* e) {
    cam_dist_ *= (e->angleDelta().y() > 0)
        ? (1.0f / cam_cfg_.zoom_step) : cam_cfg_.zoom_step;
    cam_dist_  = std::clamp(cam_dist_, cam_cfg_.min_distance, cam_cfg_.max_distance);
    recompute_camera();
    dirty_ = true;
}

void Viewport::resizeEvent(QResizeEvent* e) {
    QWidget::resizeEvent(e);
    dirty_ = true;
}

void Viewport::on_render_tick() {
    if (!dirty_ || rendering_) return;
    rendering_ = true;
    do_render();
    rendering_ = false;
}

void Viewport::try_pick(const QPoint& pos) {
    int w = width();
    int h = height();
    if (w < 8 || h < 8) return;

    // The picker must be up to date with the current scene geometry.
    // ScenePicker::rebuild is a JIT compile — we do it only if the scene has
    // changed (valid_for checks the scene_hash).
    if (!picker_->valid_for(*scene_)) {
        auto rb = picker_->rebuild(*scene_);
        if (!rb) {
            qWarning("Picker rebuild error: %s", rb.error().c_str());
            return;
        }
    }

    // The image is drawn scaled to rect(); but pixels_ are rendered at
    // widget size directly, so click pos == pixel coord 1:1.
    auto hit = picker_->pick_pixel(scene_->camera(), pos.x(), pos.y(), w, h);
    if (hit)
        Q_EMIT object_picked(QString::fromStdString(*hit));
    else
        Q_EMIT object_picked(QString());   // click on empty space → deselect
}

void Viewport::recompute_camera() {
    float r = cam_dist_;
    float sx = r * std::cos(cam_pitch_) * std::sin(cam_yaw_);
    float sy = r * std::sin(cam_pitch_);
    float sz = r * std::cos(cam_pitch_) * std::cos(cam_yaw_);
    scene_->camera().position = {sx, sy, sz};
    scene_->camera().target   = {0.0f, 0.0f, 0.0f};
    scene_->camera().up       = {0.0f, 1.0f, 0.0f};
}

void Viewport::do_render() {
    using clk = std::chrono::high_resolution_clock;
    auto t_total = clk::now();

    // GPU path: attempt the compute pipeline. If it fails for any reason
    // we fall through to the CPU path so the viewport never goes blank.
    if (gpu_mode_) {
        if (do_render_gpu()) {
            auto t_end = clk::now();
            double total_ms = std::chrono::duration<double, std::milli>(
                t_end - t_total).count();
            // Report a "GPU" render — was_cached / structure_unchanged
            // values are placeholders here.
            Q_EMIT render_completed(total_ms, total_ms, true, true);
            dirty_ = false;
            update();
            return;
        }
        // fall through to CPU on failure
    }

    auto fn_or = compiler_->compile_if_changed(*scene_);
    if (!fn_or) {
        qWarning("Compile error: %s", fn_or.error().c_str());
        dirty_ = false;
        return;
    }
    auto fn = *fn_or;

    int w = width();
    int h = height();
    if (w < 8 || h < 8) return;

    pixels_.assign(static_cast<std::size_t>(w) * h * 4, 0.0f);
    RenderParams rp;
    rp.width = w; rp.height = h; rp.tile_size = 64;
    rp.ssaa  = ssaa_;

    auto t_render = clk::now();
    TileScheduler::render(fn, pixels_.data(), scene_->camera(), rp,
                          compiler_->params_buffer());
    auto t_end = clk::now();

    image_ = QImage(w, h, QImage::Format_RGBA8888);
    auto* dst = image_.bits();
    for (int i = 0; i < w * h; ++i) {
        auto clamp = [](float v) -> std::uint8_t {
            v = std::max(0.0f, std::min(1.0f, v));
            return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        };
        dst[i*4 + 0] = clamp(pixels_[i*4 + 0]);
        dst[i*4 + 1] = clamp(pixels_[i*4 + 1]);
        dst[i*4 + 2] = clamp(pixels_[i*4 + 2]);
        dst[i*4 + 3] = 255;
    }
    dirty_ = false;
    update();

    double render_ms = std::chrono::duration<double, std::milli>(t_end - t_render).count();
    double total_ms  = std::chrono::duration<double, std::milli>(t_end - t_total).count();
    const auto& stats = compiler_->last_stats();
    Q_EMIT render_completed(render_ms, total_ms,
                            stats.was_cached, stats.structure_unchanged);
}

// ─── GPU path ───────────────────────────────────────────────────────────────

// CUDA helpers, isolated behind FREP_HAS_CUDA so the rest of the file stays
// free of #ifdefs. When CUDA isn't built, cuda_available() is false and the
// CUDA render path is never reached.
#ifdef FREP_HAS_CUDA
bool Viewport::cuda_available() const { return gpu::CudaCtx::available(); }
void Viewport::reset_cuda_ctx() {
    cuda_ctx_.reset();
    cuda_params_.reset();
    cuda_structure_hash_ = 0;
    cuda_incremental_ = false;
}

// Pull every bound parameter's current value from the scene into the params
// buffer. Mirrors IncrementalCompiler::refresh_params_from_scene. Unwritten
// slots keep their previous value; the structure_hash gate guarantees the
// binding table still matches the scene.
void Viewport::refresh_cuda_params() {
    if (!cuda_params_) return;
    std::function<void(const FRepNode*)> walk = [&](const FRepNode* n) {
        if (!n) return;
        for (const auto& [k, v] : n->params) {
            auto sep_id = n->id;
            if (cuda_params_->has(sep_id, k)) cuda_params_->set(sep_id, k, v);
        }
        for (const auto& c : n->children) walk(c.get());
    };
    for (const auto& [id, obj] : scene_->objects())
        walk(obj.geometry.get());
}

// Render one frame via the GPU-IR (CUDA/PTX) path into `rgba` (W*H*4 RGBA8).
// Rebuilds the CUDA module (codegen → IR → PTX → cuModuleLoadData) only when
// the scene *structure* changes. Parameter-only edits refresh a runtime
// params buffer and re-launch — no recompile. Returns false on failure
// (caller falls back to CPU). Post-process (SSAA downsample) is shared in
// do_render_gpu — this only produces the raw rendered pixels.
bool Viewport::render_gpu_cuda(int hi_w, int hi_h, std::size_t scene_hash,
                               std::vector<std::uint8_t>& rgba) {
    static std::once_flag init_flag;
    std::call_once(init_flag, [] {
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
    });

    // Amortize on the structure hash: topology changes force a recompile;
    // parameter edits do not. This mirrors IncrementalCompiler for the CPU
    // path — compile once per structure, then re-render cheaply.
    const std::size_t structure_hash = scene_->structure_hash();
    const bool structure_changed = (!cuda_ctx_ || structure_hash != cuda_structure_hash_);

    if (structure_changed) {
        auto ctx = std::make_unique<llvm::LLVMContext>();
        // Compile in Incremental mode: parameters become reads from a
        // runtime buffer instead of baked constants, so the module stays
        // valid across parameter edits.
        TracerConfig tc = compiler_->tracer_config();
        tc.incremental_params = true;
        SceneCodegen cg(*ctx, tc);
        // Interactive compile policy: geometry/material/deform parameters are
        // runtime (editable without recompiling), render/observer settings
        // stay constant. The policy is static, so a stack instance scoped to
        // this compile is fine — codegen reads it synchronously below.
        static const ByParamClassPolicy cuda_policy =
            ByParamClassPolicy::interactive();
        cg.set_compile_policy(&cuda_policy);
        try {
            cg.emit_gpu_kernel(*scene_);
        } catch (const std::exception& e) {
            gpu_status_ = QString("codegen failed: %1").arg(e.what());
            return false;
        }
        // Capture the parameter binding table before the module is consumed.
        cuda_params_ = std::make_unique<IncrementalParams>(cg);
        cuda_incremental_ = (cuda_params_->size() > 0);

        auto mod = cg.take_module();
        NVPTXRetarget retarget;
        auto ptx = retarget.retarget(*mod);
        if (!ptx) {
            gpu_status_ = QString("nvptx failed: %1")
                .arg(QString::fromStdString(retarget.last_error));
            return false;
        }
        auto ctx_or = gpu::CudaCtx::create(*ptx, "render_tile");
        if (!ctx_or) {
            gpu_status_ = QString("CUDA init: %1")
                .arg(QString::fromStdString(ctx_or.error()));
            return false;
        }
        cuda_ctx_ = std::move(*ctx_or);
        cuda_structure_hash_ = structure_hash;
        gpu_scene_hash_ = scene_hash;
    }

    // Refresh the params buffer from the scene's current values every frame.
    // When incremental codegen is active this is what makes a parameter edit
    // show up without a recompile.
    std::vector<float> params;
    if (cuda_incremental_ && cuda_params_) {
        refresh_cuda_params();
        params.assign(cuda_params_->buffer(),
                      cuda_params_->buffer() + cuda_params_->size());
    }

    // Camera basis for the kernel (mirrors GpuIrExecutor / TileScheduler).
    const Camera& cam = scene_->camera();
    auto sub = [](const auto& a, const auto& b) {
        return std::array<float,3>{a[0]-b[0], a[1]-b[1], a[2]-b[2]}; };
    auto nrm = [](std::array<float,3> v) {
        float l = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        if (l < 1e-8f) return std::array<float,3>{0,0,-1};
        return std::array<float,3>{v[0]/l, v[1]/l, v[2]/l}; };
    auto crs = [](const auto& a, const auto& b) {
        return std::array<float,3>{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                                   a[0]*b[1]-a[1]*b[0]}; };
    auto fwd = nrm(sub(cam.target, cam.position));
    auto right = nrm(crs(fwd, cam.up));
    auto up = crs(right, fwd);
    float view_scale = (cam.projection == Camera::Projection::Orthographic)
        ? -0.5f * cam.ortho_size
        :  std::tan(cam.fov_deg * 3.14159265f / 360.0f);

    gpu::CudaRenderArgs args;
    args.tx = 0; args.ty = 0; args.tw = hi_w; args.th = hi_h;
    args.iw = hi_w; args.ih = hi_h;
    for (int i = 0; i < 3; ++i) {
        args.cam_pos[i] = cam.position[i]; args.cam_fwd[i] = fwd[i];
        args.cam_right[i] = right[i]; args.cam_up[i] = up[i];
    }
    args.fov_scale = view_scale;

    auto rr = cuda_ctx_->render(args, params, rgba);
    if (!rr) {
        gpu_status_ = QString("CUDA render: %1")
            .arg(QString::fromStdString(rr.error()));
        return false;
    }
    return true;
}
#else
bool Viewport::cuda_available() const { return false; }
void Viewport::reset_cuda_ctx() {}
bool Viewport::render_gpu_cuda(int, int, std::size_t, std::vector<std::uint8_t>&) {
    gpu_status_ = "CUDA not built (FREP_BUILD_CUDA off)";
    return false;
}
#endif

bool Viewport::set_gpu_mode(bool on, GpuBackend backend) {
    if (on) {
        gpu_backend_ = backend;
        // Availability depends on the chosen backend.
        bool avail = false;
        const char* what = "";
#ifdef FREP_HAS_CUDA
        if (backend == GpuBackend::Cuda) { avail = cuda_available(); what = "CUDA"; }
#endif
        if (backend == GpuBackend::Vulkan) { avail = gpu::VulkanCtx::available(); what = "Vulkan"; }
        if (backend == GpuBackend::Cuda && !what[0]) what = "CUDA (not built)";
        if (!avail) {
            gpu_status_ = QString("%1 not available").arg(what);
            gpu_mode_ = false;
            return false;
        }
        // Switching backend invalidates any cached context.
        gpu_ctx_.reset();
        reset_cuda_ctx();
        gpu_scene_hash_ = 0;
    }
    gpu_mode_ = on;
    if (on) gpu_status_ = QString("GPU mode active (%1)")
                .arg(backend == GpuBackend::Cuda ? "CUDA/IR" : "Vulkan/GLSL");
    else    gpu_status_.clear();
    dirty_ = true;
    do_render();
    return on;
}

bool Viewport::do_render_gpu() {
    using clk = std::chrono::high_resolution_clock;
    int w = width(), h = height();
    if (w < 8 || h < 8) return false;

    // While the user is actively dragging the camera, render at half
    // resolution to keep interactivity responsive. The image is scaled
    // up by QPainter (drawImage to rect()) so the lower resolution
    // shows as soft pixels rather than a smaller picture. On drag
    // release, the next render restores full resolution.
    int render_w = w, render_h = h;
    if (dragging_) {
        render_w = std::max(64, w / 2);
        render_h = std::max(48, h / 2);
    }

    // (Re)build the GPU context if scene structure changed. We use the
    // structural hash of every visible object's geometry as the cache
    // key — parameter edits stay cached, topology changes invalidate.
    // We also mix in the TracerConfig fields that GlslEmitter bakes
    // into the source (shading model, shadow / AO toggles, softness,
    // strength). Without those, a Render-tab edit doesn't drift the
    // hash, the re-emit is skipped, and the user sees a stale image
    // with the previous shading config — visible only when the user
    // pans the camera (which re-runs the shader on the cached
    // pipeline, hence the "old shadows still there after I disabled
    // shadows" symptom reported on offscreen GPU mode).
    std::size_t scene_hash = 0;
    auto mix = [&](std::size_t v) {
        scene_hash ^= v + 0x9E3779B9 + (scene_hash << 6) + (scene_hash >> 2);
    };
    auto hf = [](float f) -> std::size_t {
        std::uint32_t u; std::memcpy(&u, &f, sizeof u);
        return static_cast<std::size_t>(u);
    };
    for (const auto& [id, obj] : scene_->objects()) {
        if (!obj.visible) continue;
        mix(obj.geometry->structural_hash());
    }
    const auto& tc = compiler_->tracer_config();
    mix(static_cast<std::size_t>(tc.shading_model));
    mix(static_cast<std::size_t>(tc.enable_shadows));
    mix(static_cast<std::size_t>(tc.enable_ao));
    mix(hf(tc.shadow_softness));
    mix(hf(tc.ao_strength));
    mix(hf(tc.ao_step));
    mix(static_cast<std::size_t>(tc.ao_samples));
    // v4.1.0: tracer iteration limits / thresholds bake into the
    // emitted GLSL too. Hash them so a Render-tab edit to max_steps
    // etc. forces a re-emit + pipeline rebuild.
    mix(static_cast<std::size_t>(tc.max_steps));
    mix(hf(tc.max_dist));
    mix(hf(tc.epsilon));
    mix(static_cast<std::size_t>(tc.shadow_steps));
    // Sky gradient constants — affect the emitted miss-branch source.
    for (float c : tc.sky_top)     mix(hf(c));
    for (float c : tc.sky_horizon) mix(hf(c));
    // Reflections change the emitted shader structure (extra functions
    // + bounce loop), so max_bounces must drift the hash.
    mix(static_cast<std::size_t>(tc.max_bounces));
    // Soft-shadow sample count + light radius are baked into the
    // shadow loop, and accumulation frames affect temporal blending.
    mix(static_cast<std::size_t>(tc.shadow_samples));
    mix(hf(tc.shadow_light_radius));

    mix(static_cast<std::size_t>(tc.shadow_samples));
    mix(hf(tc.shadow_light_radius));

    // Supersample factor. Render at ssaa_× the target resolution, then
    // box-downsample below. Skipped while dragging (already at reduced
    // resolution for responsiveness). The render stage (below) branches on
    // the chosen GPU backend; the SSAA downsample is shared post-process.
    const int ss = dragging_ ? 1 : std::max(1, ssaa_);
    const int hi_w = render_w * ss;
    const int hi_h = render_h * ss;

    std::vector<std::uint8_t> rgba;
    // Render stage: branch on the chosen executor. Both produce a
    // hi_w×hi_h RGBA8 buffer; the post-process (downsample) below is shared,
    // so the two backends differ only in the render stage — keeping them
    // comparable per the path model (docs/ARCHITECTURE_PATHS.md).
    if (gpu_backend_ == GpuBackend::Cuda) {
        if (!render_gpu_cuda(hi_w, hi_h, scene_hash, rgba))
            return false;
    } else {
        // Vulkan/GLSL path: (re)build the GLSL→SPIR-V→Vulkan context, then
        // render via push constants.
        if (!gpu_ctx_ || scene_hash != gpu_scene_hash_) {
            auto emit_or = gpu::GlslEmitter::emit(*scene_,
                                                  compiler_->tracer_config());
            if (!emit_or) {
                gpu_status_ = QString("emit failed: %1")
                    .arg(QString::fromStdString(emit_or.error()));
                return false;
            }
            auto spv_or = gpu::compile_glsl_to_spv_managed(emit_or->source);
            if (!spv_or) {
                gpu_status_ = QString("compile failed: %1")
                    .arg(QString::fromStdString(spv_or.error()).left(120));
                return false;
            }
            auto ctx_or = gpu::VulkanCtx::create(spv_or->path(),
                                                 emit_or->mesh_voxels,
                                                 emit_or->texture_pixels);
            if (!ctx_or) {
                gpu_status_ = QString("Vulkan init: %1")
                    .arg(QString::fromStdString(ctx_or.error()));
                return false;
            }
            gpu_ctx_ = std::move(*ctx_or);
            gpu_scene_hash_ = scene_hash;
        }
        gpu::ShaderPush p = gpu::build_push_from_scene(*scene_, hi_w, hi_h);
        auto rr = gpu_ctx_->render(p, rgba);
        if (!rr) {
            gpu_status_ = QString("render failed: %1")
                .arg(QString::fromStdString(rr.error()));
            return false;
        }
    }

    // Box-downsample the ss×ss supersampled buffer into the target
    // image. Averaging every ss×ss block of source texels gives a true
    // box filter — better than the real-time path's bilinear blit for
    // 3× (which only mixes a 2×2 neighbourhood of the 9 samples). For
    // ss == 1 this is a straight copy.
    image_ = QImage(render_w, render_h, QImage::Format_RGBA8888);
    if (ss == 1) {
        std::memcpy(image_.bits(), rgba.data(),
                    static_cast<std::size_t>(render_w) * render_h * 4);
    } else {
        std::uint8_t* dst = image_.bits();
        const int blk = ss * ss;
        for (int y = 0; y < render_h; ++y) {
            for (int x = 0; x < render_w; ++x) {
                int acc[4] = {0, 0, 0, 0};
                for (int sy = 0; sy < ss; ++sy) {
                    const int srow = (y * ss + sy) * hi_w;
                    for (int sx = 0; sx < ss; ++sx) {
                        const std::uint8_t* s =
                            &rgba[(static_cast<std::size_t>(srow) + x * ss + sx) * 4];
                        acc[0] += s[0]; acc[1] += s[1];
                        acc[2] += s[2]; acc[3] += s[3];
                    }
                }
                std::uint8_t* d =
                    &dst[(static_cast<std::size_t>(y) * render_w + x) * 4];
                d[0] = static_cast<std::uint8_t>(acc[0] / blk);
                d[1] = static_cast<std::uint8_t>(acc[1] / blk);
                d[2] = static_cast<std::uint8_t>(acc[2] / blk);
                d[3] = static_cast<std::uint8_t>(acc[3] / blk);
            }
        }
    }

    double gpu_render_ms = 0.0;
#ifdef FREP_HAS_CUDA
    if (gpu_backend_ == GpuBackend::Cuda && cuda_ctx_)
        gpu_render_ms = cuda_ctx_->stats().render_ms;
    else
#endif
    if (gpu_ctx_) gpu_render_ms = gpu_ctx_->stats().render_ms;

    gpu_status_ = QString("GPU render: %1 ms%2 [%3]")
        .arg(gpu_render_ms, 0, 'f', 1)
        .arg(dragging_ ? " (preview)" : "")
        .arg(gpu_backend_ == GpuBackend::Cuda ? "CUDA/IR" : "Vulkan/GLSL");
    return true;
}

} // namespace frep::gui
