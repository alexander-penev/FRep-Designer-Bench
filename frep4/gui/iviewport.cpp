// gui/iviewport.cpp
//
// Translation unit for IViewport. The class is pure-virtual and has
// no out-of-line functions to define, but a corresponding .cpp does
// two useful things:
//
//   1. Forces AutoMoc to actually generate a moc_iviewport.cpp with
//      content (when only a .hpp exists in the source list and
//      nothing #includes it as a primary source, Qt's CMake AutoMoc
//      sometimes emits an empty placeholder instead of running moc).
//
//   2. Anchors the vtable so the compiler's hidden-symbols policy
//      doesn't elide the metaobject when the interface ends up with
//      only one in-tree implementer.

#include "gui/iviewport.hpp"

// Intentionally empty — IViewport's destructor and signals are inline
// in the header. The translation unit exists purely to ensure moc
// runs and the vtable lands in the .so/.a output.
