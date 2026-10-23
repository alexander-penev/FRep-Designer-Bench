// core/gpu/gpu_numa.cpp
#include "core/gpu/gpu_numa.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
namespace frep { const cpu_set_t& launch_cpu_set(); }   // cpu_affinity.cpp
#endif

namespace frep::gpu {

int numa_node_of_pci(const std::string& bdf) {
    if (bdf.empty()) return -1;
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/numa_node");
    int n = -1;
    if (!(f >> n)) return -1;
    return n;
}

bool bind_thread_to_node(int node) {
#ifdef __linux__
    if (node < 0) return false;
    if (const char* s = std::getenv("FREP_GPU_NUMA"); s && s[0] == '0') return false;
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string list;
    if (!std::getline(f, list) || list.empty()) return false;

    // "0-13,28-41" → set
    cpu_set_t want; CPU_ZERO(&want);
    std::size_t pos = 0;
    while (pos < list.size()) {
        std::size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        std::string tok = list.substr(pos, comma - pos);
        int lo = -1, hi = -1;
        if (std::sscanf(tok.c_str(), "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) if (c >= 0) CPU_SET(c, &want);
        } else if (std::sscanf(tok.c_str(), "%d", &lo) == 1 && lo >= 0 && lo < CPU_SETSIZE) {
            CPU_SET(lo, &want);
        }
        pos = comma + 1;
    }
    cpu_set_t both;
    CPU_AND(&both, &want, &frep::launch_cpu_set());
    if (CPU_COUNT(&both) == 0) return false;   // the operator excluded this node
    return pthread_setaffinity_np(pthread_self(), sizeof(both), &both) == 0;
#else
    (void)node;
    return false;
#endif
}

int bind_thread_to_pci_device(const std::string& bdf) {
    const int node = numa_node_of_pci(bdf);
    return bind_thread_to_node(node) ? node : -1;
}

std::string vk_pci_bdf(void* vk_physical_device) {
    auto pd = static_cast<VkPhysicalDevice>(vk_physical_device);
    if (!pd) return {};
    std::uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, ext.data());
    bool has = false;
    for (const auto& e : ext)
        if (std::strcmp(e.extensionName, "VK_EXT_pci_bus_info") == 0) { has = true; break; }
    if (!has) return {};
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{};
    pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.%x",
                  pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
    return buf;
}

} // namespace frep::gpu
