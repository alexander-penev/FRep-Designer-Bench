// core/exec/executor_factory.hpp
//
// One place that maps a path name / PathKind to a concrete IExecutor. The CLI
// tools (parity_check, dist_render) and the GUI all need this mapping; keeping
// it in one spot stops them from drifting (e.g. one tool learning about a new
// path while another doesn't).

#pragma once

#include "core/exec/multipath.hpp"
#include "core/compiler/codegen.hpp"

#include <memory>
#include <string>
#include <vector>

namespace frep::exec {

// Build the executor for a path name ("cpu_ir", "gpu_glsl", "gpu_ir",
// "gpu_rtx"). Returns nullptr for an unknown name. GPU paths may still fail
// later at render time if the device is absent — that's reported per-render,
// not here, so the GUI can offer the option and degrade gracefully.
std::unique_ptr<IExecutor> make_executor(const std::string& path,
                                         const TracerConfig& cfg = {});

// Same, keyed by enum (Remote has no local executor → nullptr). Use
// path_kind_name() from multipath.hpp for the name of a kind.
std::unique_ptr<IExecutor> make_executor(PathKind kind,
                                         const TracerConfig& cfg = {});

// The locally-runnable paths, in display order (excludes Remote).
inline std::vector<PathKind> local_paths() {
    return { PathKind::CpuIr, PathKind::GpuGlsl,
             PathKind::GpuIr, PathKind::GpuRtx };
}

}  // namespace frep::exec
