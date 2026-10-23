#pragma once
// core/gpu/gpu_numa.hpp
//
// Keep the GPU's host side on the GPU's socket.
//
// On a two-socket host the card sits on one socket's PCIe root. A process the
// scheduler places on the other socket pays a QPI hop on every readback and
// on every host-side touch of the frame: measured at 512² on this machine,
// +48% on GPU_IR (4 MB float4 frame), +22% on GPU_RTX, +15% on GPU_GLSL, and
// the scheduler put it there two runs in three. Each GPU context therefore
// pins the thread that creates it — the thread that will submit and read
// back — to the CPUs of the card's NUMA node, before the driver allocates
// the host-visible memory that thread will use, so those pages land on the
// same node. The pin is intersected with the process's launch affinity, so
// an operator's numactl/taskset is narrowed, never widened; a host without
// NUMA information (one node, no sysfs entry) is left alone. The CPU render
// pool's workers reset themselves to the launch set (cpu_affinity.hpp), so
// the pin never spreads to the CPU path. FREP_GPU_NUMA=0 disables the pin.
#include <string>

namespace frep::gpu {

// NUMA node of a PCI device ("0000:07:00.0", as sysfs spells it); -1 when
// unknown or not NUMA.
int numa_node_of_pci(const std::string& bdf);

// Pin the calling thread to `node`'s CPUs (within the launch set). Returns
// whether a pin was applied.
bool bind_thread_to_node(int node);

// Both: the node bound to, or -1 when nothing was done.
int bind_thread_to_pci_device(const std::string& bdf);

// PCI address of a VkPhysicalDevice (passed as void* to keep Vulkan out of
// this header), "" when the device does not report one.
std::string vk_pci_bdf(void* vk_physical_device);

} // namespace frep::gpu
