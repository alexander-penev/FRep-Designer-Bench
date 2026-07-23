#pragma once
// core/gpu/glsl_compile.hpp
//
// Compiles a GLSL compute shader source string into a SPIR-V .spv file
// by spawning glslangValidator as a subprocess. The binary is written
// to a temp file path which the caller can hand to VulkanCtx::create().
//
// Why a subprocess: linking libshaderc / glslang directly is fiddly
// (the package layout varies between distros). For a PoC, the simplest
// thing that works is a subprocess call — the same approach we already
// use for the Khronos llvm-spirv translator in spirv_external.hpp.

#include <atomic>
#include <cstdio>
#include <cstring>
#include <expected>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <sys/wait.h>
#include <unistd.h>

namespace frep::gpu {

// RAII wrapper around a temporary SPIR-V file. Deletes the file on
// destruction (unless `release()` was called). Move-only; copying a
// SpvFile would attempt to delete the same file twice on its
// destructor.
class SpvFile {
public:
    SpvFile() = default;
    explicit SpvFile(std::string p) : path_(std::move(p)) {}
    ~SpvFile() { cleanup(); }

    SpvFile(const SpvFile&)            = delete;
    SpvFile& operator=(const SpvFile&) = delete;
    SpvFile(SpvFile&& o) noexcept : path_(std::move(o.path_)) { o.path_.clear(); }
    SpvFile& operator=(SpvFile&& o) noexcept {
        if (this != &o) {
            cleanup();
            path_ = std::move(o.path_);
            o.path_.clear();
        }
        return *this;
    }

    const std::string& path()  const { return path_; }
    bool               empty() const { return path_.empty(); }
    // Hand off ownership — the caller becomes responsible for deletion.
    std::string release() { auto p = std::move(path_); path_.clear(); return p; }

private:
    void cleanup() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
            path_.clear();
        }
    }
    std::string path_;
};

// Locate glslangValidator on PATH.
inline std::string find_glslang() {
    const char* path_env = std::getenv("PATH");
    if (!path_env) path_env = "/usr/bin:/usr/local/bin";
    std::string p(path_env);
    std::size_t start = 0;
    while (start < p.size()) {
        auto end = p.find(':', start);
        if (end == std::string::npos) end = p.size();
        std::string full = p.substr(start, end - start) + "/glslangValidator";
        if (access(full.c_str(), X_OK) == 0) return full;
        start = end + 1;
    }
    return {};
}

// Compile `glsl_source` (a compute shader) to a SPIR-V file at the path
// returned in `spv_path_out`. The file lives in /tmp and the caller owns
// it (or just lets process exit clean it up).
inline std::expected<std::string, std::string>
compile_glsl_to_spv(const std::string& glsl_source)
{
    auto exe = find_glslang();
    if (exe.empty())
        return std::unexpected(
            "glslangValidator not found on PATH "
            "(install with: apt install glslang-tools)");

    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path();
    // Use a monotonic counter so repeated compiles get distinct files
    // even when the same source string is reused — the previous
    // `&glsl_source` pointer trick failed in that case.
    static std::atomic<std::uint64_t> compile_seq{0};
    std::uint64_t seq = compile_seq.fetch_add(1, std::memory_order_relaxed);
    std::string base = "frep_glsl_" + std::to_string(::getpid()) + "_"
                     + std::to_string(seq);
    auto src_path = (tmp / (base + ".comp")).string();
    auto spv_path = (tmp / (base + ".spv")).string();

    {
        std::ofstream f(src_path);
        if (!f) return std::unexpected("cannot write " + src_path);
        f << glsl_source;
    }

    int err_pipe[2];
    if (pipe(err_pipe) != 0)
        return std::unexpected(std::string("pipe: ") + std::strerror(errno));

    pid_t pid = fork();
    if (pid < 0) {
        close(err_pipe[0]); close(err_pipe[1]);
        return std::unexpected(std::string("fork: ") + std::strerror(errno));
    }
    if (pid == 0) {
        // Child — invoke glslangValidator -V src.comp -o out.spv.
        close(err_pipe[0]);
        dup2(err_pipe[1], STDERR_FILENO);
        dup2(err_pipe[1], STDOUT_FILENO);  // capture stdout too
        close(err_pipe[1]);
        const char* argv[] = {
            exe.c_str(), "-V", src_path.c_str(),
            "-o", spv_path.c_str(), nullptr
        };
        execvp(argv[0], const_cast<char* const*>(argv));
        std::fprintf(stderr, "execvp: %s\n", std::strerror(errno));
        _exit(127);
    }

    close(err_pipe[1]);
    std::string log;
    char buf[4096];
    ssize_t n;
    while ((n = read(err_pipe[0], buf, sizeof(buf))) > 0)
        log.append(buf, buf + n);
    close(err_pipe[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        return std::unexpected(std::string("waitpid: ") + std::strerror(errno));
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (code != 0 || !fs::exists(spv_path)) {
        fs::remove(src_path);
        return std::unexpected(
            "glslangValidator failed (exit " + std::to_string(code) + "):\n"
            + log);
    }
    fs::remove(src_path);
    return spv_path;
}

// Convenience wrapper: returns a SpvFile that auto-deletes the .spv on
// destruction. Use this in callers that don't need to keep the file
// around beyond the VulkanCtx::create() call.
inline std::expected<SpvFile, std::string>
compile_glsl_to_spv_managed(const std::string& glsl_source)
{
    auto r = compile_glsl_to_spv(glsl_source);
    if (!r) return std::unexpected(r.error());
    return SpvFile(*r);
}

// Compile a ray-tracing stage to SPIR-V. `stage_ext` is the glslang stage
// suffix without the dot: "rgen", "rint", "rchit", "rmiss", etc. RT stages
// need Vulkan 1.2 (SPIR-V 1.4) and the right file extension so glslang infers
// the stage. Returns the .spv path (caller deletes) or an error+log.
inline std::expected<std::string, std::string>
compile_rt_stage_to_spv(const std::string& glsl_source,
                        const std::string& stage_ext)
{
    auto exe = find_glslang();
    if (exe.empty())
        return std::unexpected("glslangValidator not found on PATH "
                               "(install with: apt install glslang-tools)");

    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path();
    static std::atomic<std::uint64_t> rt_seq{0};
    std::uint64_t seq = rt_seq.fetch_add(1, std::memory_order_relaxed);
    std::string base = "frep_rt_" + std::to_string(::getpid()) + "_" +
                       std::to_string(seq);
    auto src_path = (tmp / (base + "." + stage_ext)).string();
    auto spv_path = (tmp / (base + "_" + stage_ext + ".spv")).string();

    { std::ofstream f(src_path); if (!f) return std::unexpected("cannot write " + src_path); f << glsl_source; }

    int err_pipe[2];
    if (pipe(err_pipe) != 0)
        return std::unexpected(std::string("pipe: ") + std::strerror(errno));
    pid_t pid = fork();
    if (pid < 0) { close(err_pipe[0]); close(err_pipe[1]);
        return std::unexpected(std::string("fork: ") + std::strerror(errno)); }
    if (pid == 0) {
        close(err_pipe[0]);
        dup2(err_pipe[1], STDERR_FILENO);
        dup2(err_pipe[1], STDOUT_FILENO);
        close(err_pipe[1]);
        const char* argv[] = {
            exe.c_str(), "--target-env", "vulkan1.2",
            "-V", src_path.c_str(), "-o", spv_path.c_str(), nullptr
        };
        execvp(argv[0], const_cast<char* const*>(argv));
        std::fprintf(stderr, "execvp: %s\n", std::strerror(errno));
        _exit(127);
    }
    close(err_pipe[1]);
    std::string log; char buf[4096]; ssize_t n;
    while ((n = read(err_pipe[0], buf, sizeof(buf))) > 0) log.append(buf, buf + n);
    close(err_pipe[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        return std::unexpected(std::string("waitpid: ") + std::strerror(errno));
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (code != 0 || !fs::exists(spv_path)) {
        fs::remove(src_path);
        return std::unexpected("glslangValidator (RT " + stage_ext +
                               ") failed (exit " + std::to_string(code) + "):\n" + log);
    }
    fs::remove(src_path);
    return spv_path;
}

} // namespace frep::gpu
