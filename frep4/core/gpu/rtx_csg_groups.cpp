// core/gpu/rtx_csg_groups.cpp — see rtx_csg_groups.hpp.

#include "core/gpu/rtx_csg_groups.hpp"

namespace frep::gpu {

namespace {

// Recursively collect group roots: descend through UnionNodes (separable),
// and treat any other node as a leaf group root (kept whole).
void collect(const FRepNode::Ptr& n, std::vector<FRepNode::Ptr>& out) {
    if (!n) return;
    if (n->kind == NodeKind::Union) {
        // A union of A and B → the groups of A plus the groups of B. Works for
        // n-ary unions built as a left-leaning chain too, since each UnionNode
        // is visited and its children recursed.
        for (const auto& c : n->children) collect(c, out);
    } else {
        // Anything else (primitive, smooth-union, intersection, difference,
        // transform, custom expr, mesh, ...) is one indivisible group.
        out.push_back(n);
    }
}

}  // namespace

std::vector<CsgGroup> partition_csg_groups(const FRepNode::Ptr& root) {
    std::vector<CsgGroup> groups;
    if (!root) return groups;

    std::vector<FRepNode::Ptr> roots;
    collect(root, roots);

    // Degenerate case: a tree with no top-level union collapses to one group
    // (collect pushed the root itself). That's the single-BLAS degenerate case.
    if (roots.empty()) roots.push_back(root);

    groups.reserve(roots.size());
    for (const auto& r : roots)
        groups.push_back(CsgGroup{r, r->aabb()});
    return groups;
}

}  // namespace frep::gpu
