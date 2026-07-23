#pragma once
// core/compiler/spirv_external.hpp
//
// Helper that invokes the Khronos llvm-spirv translator as an external
// process to convert LLVM bitcode into a binary SPIR-V module.
//
// We do NOT want a hard runtime dependency on the translator — it is
// optional. The wrapper:
//   1. Searches PATH for one of (llvm-spirv-22, llvm-spirv-21, llvm-spirv-20,
//      llvm-spirv) — first hit wins.
//   2. Writes the input bitcode to a temp file.
//   3. Spawns the translator with `<input>.bc -o <output>.spv`.
//   4. Reads back the resulting binary and returns it.
//   5. Optionally invokes `spirv-val` to validate.
//
// Returns std::expected: ok = vector<char> with the .spv bytes;
// error = human-readable diagnostic string.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>
#include <sys/wait.h>

namespace frep::spirv_ext {

// Result of an invocation: the binary SPIR-V module, plus a record of which
// translator (and validator, if any) was used. Useful for diagnostics.
struct InvokeResult {
    std::vector<unsigned char> bytes;          // the .spv content
    std::string                translator_path; // e.g. "/usr/bin/llvm-spirv-20"
    std::string                validator_path;  // empty if not run
    bool                       validated = false;
    std::string                validator_message;
};

// Finds an llvm-spirv binary on PATH. Returns the absolute path of the
// first match, or empty if none.
inline std::string find_translator() {
    static const char* candidates[] = {
        "llvm-spirv-22",
        "llvm-spirv-21",
        "llvm-spirv-20",
        "llvm-spirv-19",
        "llvm-spirv-18",
        "llvm-spirv",
        nullptr,
    };
    const char* path_env = std::getenv("PATH");
    if (!path_env) path_env = "/usr/bin:/usr/local/bin";

    for (const char** name = candidates; *name; ++name) {
        std::string p(path_env);
        std::size_t start = 0;
        while (start < p.size()) {
            auto end = p.find(':', start);
            if (end == std::string::npos) end = p.size();
            std::string dir = p.substr(start, end - start);
            std::string full = dir + "/" + *name;
            if (access(full.c_str(), X_OK) == 0) return full;
            start = end + 1;
        }
    }
    return {};
}

inline std::string find_validator() {
    const char* path_env = std::getenv("PATH");
    if (!path_env) path_env = "/usr/bin:/usr/local/bin";
    std::string p(path_env);
    std::size_t start = 0;
    while (start < p.size()) {
        auto end = p.find(':', start);
        if (end == std::string::npos) end = p.size();
        std::string full = p.substr(start, end - start) + "/spirv-val";
        if (access(full.c_str(), X_OK) == 0) return full;
        start = end + 1;
    }
    return {};
}

// Reads the entire file at `path` into a vector of bytes.
inline std::expected<std::vector<unsigned char>, std::string>
read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return std::unexpected("cannot open " + path);
    auto size = f.tellg();
    f.seekg(0);
    std::vector<unsigned char> buf(static_cast<std::size_t>(size));
    f.read(reinterpret_cast<char*>(buf.data()), size);
    if (!f) return std::unexpected("cannot read " + path);
    return buf;
}

// Spawns `argv[0]` with the given arguments. Returns the exit status, plus
// the contents of its stderr (so we can surface translator diagnostics).
// argv must end with nullptr.
inline std::expected<std::pair<int, std::string>, std::string>
run_subprocess(const char* const argv[]) {
    int err_pipe[2];
    if (pipe(err_pipe) != 0)
        return std::unexpected(std::string("pipe failed: ") + std::strerror(errno));

    pid_t pid = fork();
    if (pid < 0) {
        close(err_pipe[0]); close(err_pipe[1]);
        return std::unexpected(std::string("fork failed: ") + std::strerror(errno));
    }
    if (pid == 0) {
        // Child — wire stderr to the pipe, run the binary.
        close(err_pipe[0]);
        dup2(err_pipe[1], STDERR_FILENO);
        close(err_pipe[1]);
        // /dev/null stdout so the binary's chatter doesn't pollute caller output.
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDOUT_FILENO); close(devnull); }
        execvp(argv[0], const_cast<char* const*>(argv));
        std::fprintf(stderr, "execvp failed: %s\n", std::strerror(errno));
        _exit(127);
    }

    close(err_pipe[1]);
    std::string err_text;
    char buf[4096];
    ssize_t n;
    while ((n = read(err_pipe[0], buf, sizeof(buf))) > 0)
        err_text.append(buf, buf + n);
    close(err_pipe[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        return std::unexpected(std::string("waitpid failed: ")
                               + std::strerror(errno));
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return std::pair{exit_code, err_text};
}

// Main entry. Translates the given bitcode buffer into binary SPIR-V.
// Optionally validates with spirv-val.
inline std::expected<InvokeResult, std::string>
translate_bitcode_to_spirv(const std::vector<unsigned char>& bitcode,
                           bool validate = true)
{
    auto translator = find_translator();
    if (translator.empty())
        return std::unexpected(
            "llvm-spirv translator not found on PATH "
            "(tried llvm-spirv-22..18, llvm-spirv). "
            "Install with: apt install llvm-spirv-20");

    // Write bitcode to a temp file.
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path();
    std::string base = "frep_spirv_" + std::to_string(::getpid()) + "_"
                     + std::to_string(reinterpret_cast<std::uintptr_t>(&bitcode));
    auto bc_path  = tmp / (base + ".bc");
    auto spv_path = tmp / (base + ".spv");

    {
        std::ofstream f(bc_path, std::ios::binary);
        if (!f) return std::unexpected("cannot write " + bc_path.string());
        f.write(reinterpret_cast<const char*>(bitcode.data()), bitcode.size());
        if (!f) return std::unexpected("write failed: " + bc_path.string());
    }

    // Invoke the translator.
    std::string bc_str  = bc_path.string();
    std::string spv_str = spv_path.string();
    const char* argv[] = {
        translator.c_str(), bc_str.c_str(),
        "-o", spv_str.c_str(),
        nullptr
    };
    auto run = run_subprocess(argv);
    if (!run) { fs::remove(bc_path); return std::unexpected(run.error()); }

    auto [code, err_text] = *run;
    if (code != 0 || !fs::exists(spv_path)) {
        fs::remove(bc_path);
        std::string msg = "llvm-spirv failed (exit " + std::to_string(code) + ")";
        if (!err_text.empty()) msg += ":\n" + err_text;
        return std::unexpected(msg);
    }

    auto bytes = read_file_bytes(spv_str);
    fs::remove(bc_path);
    if (!bytes) {
        fs::remove(spv_path);
        return std::unexpected(bytes.error());
    }

    InvokeResult res;
    res.bytes           = std::move(*bytes);
    res.translator_path = translator;

    // Optionally validate.
    if (validate) {
        auto validator = find_validator();
        if (!validator.empty()) {
            const char* vargv[] = {
                validator.c_str(), spv_str.c_str(), nullptr
            };
            auto vrun = run_subprocess(vargv);
            res.validator_path = validator;
            if (vrun) {
                auto [vc, vt] = *vrun;
                res.validated         = (vc == 0);
                res.validator_message = vt.empty()
                    ? (res.validated ? "OK" : "non-zero exit, no stderr")
                    : vt;
            } else {
                res.validator_message = vrun.error();
            }
        }
    }

    fs::remove(spv_path);
    return res;
}

} // namespace frep::spirv_ext
