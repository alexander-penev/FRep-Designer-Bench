#pragma once
// core/plugin/plugin_loader.hpp
//
// Dynamic loading of plugins from .so/.dll files.
//
// Plugin library contract:
//   - Exports `extern "C" frep_plugin_register_fn frep_plugin_register;`
//   - The function is called on load and registers the plugins in PluginRegistry.
//
// Example (inside the plugin):
//   extern "C" void frep_plugin_register(frep::plugin::PluginRegistry& reg) {
//       reg.register_primitive(MyPrimitivePlugin{});
//   }

#include "core/plugin/plugin_api.hpp"

#include <dlfcn.h>

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace frep::plugin {

// Type of the plugin entry point function.
using PluginRegisterFn = void(*)(PluginRegistry&);

// LoadedPlugin — a handle to a dlopen-ed shared library.
class LoadedPlugin {
public:
    LoadedPlugin(LoadedPlugin&& o) noexcept
        : handle_(std::exchange(o.handle_, nullptr))
        , path_(std::move(o.path_)) {}

    LoadedPlugin& operator=(LoadedPlugin&& o) noexcept {
        if (this != &o) {
            close();
            handle_ = std::exchange(o.handle_, nullptr);
            path_   = std::move(o.path_);
        }
        return *this;
    }

    LoadedPlugin(const LoadedPlugin&)            = delete;
    LoadedPlugin& operator=(const LoadedPlugin&) = delete;

    ~LoadedPlugin() { close(); }

    const std::string& path() const noexcept { return path_; }

    // Loads a .so file and calls the entry point against the given registry.
    static std::expected<LoadedPlugin, std::string>
    load(const std::filesystem::path& so_path, PluginRegistry& reg) {
        void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            return std::unexpected(std::string("dlopen error: ") + dlerror());
        }

        // Clear any stale dlerror state
        dlerror();
        auto* sym = dlsym(handle, "frep_plugin_register");
        if (const char* err = dlerror(); err != nullptr) {
            dlclose(handle);
            return std::unexpected(std::string("dlsym error: ") + err);
        }
        if (!sym) {
            dlclose(handle);
            return std::unexpected("The plugin does not export frep_plugin_register");
        }

        auto entry = reinterpret_cast<PluginRegisterFn>(sym);
        entry(reg);

        LoadedPlugin p;
        p.handle_ = handle;
        p.path_   = so_path.string();
        return p;
    }

    // Scans a directory for .so files and loads all of them.
    static std::vector<LoadedPlugin>
    load_directory(const std::filesystem::path& dir, PluginRegistry& reg) {
        std::vector<LoadedPlugin> loaded;
        if (!std::filesystem::is_directory(dir)) return loaded;

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            auto ext = entry.path().extension();
            if (ext != ".so" && ext != ".dylib" && ext != ".dll") continue;

            auto res = load(entry.path(), reg);
            if (res) {
                loaded.push_back(std::move(*res));
            }
            // Silent fail: invalid plugins in the directory are skipped.
        }
        return loaded;
    }

private:
    LoadedPlugin() = default;
    void close() {
        // Intentional non-close: once a plugin has registered factories
        // / RTTI / type info into the host's PluginRegistry (a Meyer
        // singleton with process lifetime), unloading the .so leaves
        // dangling references. The singleton's destructor will run
        // after this LoadedPlugin's, and try to destroy lambdas /
        // shared_ptrs whose vtables live in the now-unloaded .so —
        // segv on exit. Letting the process tear-down close the .so
        // is the standard idiom for plugin systems.
        if (handle_) handle_ = nullptr;  // ownership simply transferred to the process
    }

    void*       handle_ = nullptr;
    std::string path_;
};

} // namespace frep::plugin
