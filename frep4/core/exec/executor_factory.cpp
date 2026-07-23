// core/exec/executor_factory.cpp
#include "core/exec/executor_factory.hpp"

#include "core/exec/cpu_executor.hpp"
#include "core/exec/gpu_executor.hpp"
#include "core/exec/gpu_ir_executor.hpp"
#include "core/exec/rtx_executor.hpp"

namespace frep::exec {

std::unique_ptr<IExecutor> make_executor(const std::string& path,
                                         const TracerConfig& cfg) {
    if (path == "cpu_ir")
        return std::make_unique<CpuIrExecutor>(
            SceneCodegen::SceneSdfMode::Inlined, cfg);
    if (path == "gpu_glsl")
        return std::make_unique<GpuGlslExecutor>(cfg);
    if (path == "gpu_ir")
        return std::make_unique<GpuIrExecutor>(
            SceneCodegen::SceneSdfMode::Inlined, cfg);
    if (path == "gpu_rtx")
        return std::make_unique<RtxExecutor>(cfg);
    return nullptr;
}

std::unique_ptr<IExecutor> make_executor(PathKind kind,
                                         const TracerConfig& cfg) {
    if (kind == PathKind::Remote) return nullptr;
    return make_executor(std::string(path_kind_name(kind)), cfg);
}

}  // namespace frep::exec
