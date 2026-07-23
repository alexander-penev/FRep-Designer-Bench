// gui/node_types.cpp

#include "gui/node_types.hpp"
#include "core/plugin/plugin_api.hpp"

#include <algorithm>
#include <string>

namespace frep::gui {

NodeTypeInfo make_plugin_node_info_by_type_name(
        const plugin::PluginRegistry& reg, const QString& node_type_name)
{
    NodeTypeInfo info;
    info.type_name   = node_type_name;
    info.category    = NodeCategory::Primitive;  // all plugins so far are primitives
    info.input_count = 0;

    const auto* slot = reg.find_primitive_by_type_name(
                            node_type_name.toStdString());
    if (!slot) {
        info.display = node_type_name;  // placeholder
        return info;
    }

    info.display = QString::fromStdString(std::string(slot->info.name));

    // Map param_names + param_defaults to NodeParam with reasonable generic
    // slider ranges. Plugins that need different ranges can be revisited
    // later (a richer plugin metadata API would let them say so directly).
    auto names = slot->param_names;
    auto defs  = slot->param_defaults;
    for (std::size_t i = 0; i < names.size(); ++i) {
        float def = i < defs.size() ? defs[i] : 0.0f;
        // For positions / signed values, span +-|def|*4 so the default sits
        // in the middle; for strictly positive ones (radii, sizes) keep a
        // [0.05, def*4] range.
        float lo, hi;
        if (def < 0.0f) {
            lo = def * 4.0f; hi = -def * 4.0f;
        } else if (def > 0.0f) {
            lo = 0.05f; hi = std::max(def * 4.0f, 1.0f);
        } else {
            lo = -2.0f; hi = 2.0f;
        }
        info.params.push_back(NodeParam{
            QString::fromStdString(std::string(names[i])),
            def, lo, hi
        });
    }
    return info;
}

} // namespace frep::gui
