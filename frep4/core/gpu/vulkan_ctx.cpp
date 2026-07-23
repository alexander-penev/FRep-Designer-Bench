// core/gpu/vulkan_ctx.cpp
//
// Minimal Vulkan compute pipeline that runs the precompiled
// sphere_trace.spv shader on the first available device.
//
// Flow (steps mirror create() and render() below):
//   create():
//     1. vkCreateInstance — optionally with the KHRONOS validation layer
//        in debug builds.
//     2. Enumerate physical devices; pick the first one that exposes
//        a compute-capable queue family.
//     3. vkCreateDevice + vkGetDeviceQueue (single compute queue).
//     4. vkCreateShaderModule from the .spv bytes.
//     5. VkDescriptorSetLayout (binding 0 = STORAGE_IMAGE, COMPUTE_BIT).
//     6. VkPipelineLayout that adds a push-constant range matching
//        `ShaderPush`.
//     7. VkComputePipeline with the shader's main entry.
//     8. VkDescriptorPool + a single descriptor set.
//     9. VkCommandPool + a primary command buffer.
//
//   render():
//     1. (Re)create the storage image at (width × height), RGBA8,
//        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT.
//     2. Allocate its memory; create the VkImageView; write the
//        descriptor.
//     3. (Re)create a host-visible staging buffer of
//        width*height*4 bytes for the readback.
//     4. Record the command buffer:
//          - transition image GENERAL
//          - bind pipeline + descriptor set + push constants
//          - vkCmdDispatch( (w+7)/8, (h+7)/8, 1 )
//          - transition image TRANSFER_SRC_OPTIMAL
//          - vkCmdCopyImageToBuffer into the staging buffer
//     5. Submit, wait on a fence, memcpy staging→caller buffer.
//
// Memory management: each Vulkan object lives behind an RAII wrapper
// to ensure correct teardown order; see the `Impl` struct.

#include "core/gpu/vulkan_ctx.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace frep::gpu {

// ─────────────────────────────────────────────────────────────────────────────
// File-scope helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = static_cast<std::size_t>(f.tellg());
    if (sz == 0) return {};
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

// Find a memory type matching the requirements; -1 if none.
int find_memory_type(VkPhysicalDevice phys, std::uint32_t type_filter,
                     VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i))
            && (mp.memoryTypes[i].propertyFlags & props) == props)
            return static_cast<int>(i);
    }
    return -1;
}

// Tiny error helper.
#define VK_TRY(expr) do {                                                 \
        VkResult _r = (expr);                                              \
        if (_r != VK_SUCCESS)                                              \
            return std::unexpected(std::string(#expr " failed: ")          \
                                 + std::to_string(_r));                    \
    } while (0)

} // anonymous

// ─────────────────────────────────────────────────────────────────────────────
// Impl — holds every Vulkan object and tears them down in reverse order.
// ─────────────────────────────────────────────────────────────────────────────
struct VulkanCtx::Impl {
    VkInstance       instance        = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice         device          = VK_NULL_HANDLE;
    VkQueue          compute_queue   = VK_NULL_HANDLE;
    std::uint32_t    compute_family  = 0;

    VkShaderModule        shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout    = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline      = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool     = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set      = VK_NULL_HANDLE;
    VkCommandPool         cmd_pool      = VK_NULL_HANDLE;
    VkCommandBuffer       cmd_buf       = VK_NULL_HANDLE;
    VkFence               fence         = VK_NULL_HANDLE;

    // Per-render objects — recreated when size changes.
    int           image_w = 0, image_h = 0;
    VkImage       image       = VK_NULL_HANDLE;
    VkDeviceMemory image_mem  = VK_NULL_HANDLE;
    VkImageView    image_view = VK_NULL_HANDLE;
    VkBuffer       staging    = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    VkDeviceSize   staging_sz  = 0;

    // Optional mesh storage buffer (binding = 1). Allocated once at
    // create() time when the caller provides mesh voxel data.
    VkBuffer       mesh_buffer    = VK_NULL_HANDLE;
    VkDeviceMemory mesh_buffer_mem = VK_NULL_HANDLE;
    VkDeviceSize   mesh_buffer_sz  = 0;
    bool           has_mesh_buffer = false;

    // Optional texture storage buffer (binding = 2). Same lifetime as
    // the mesh buffer.
    VkBuffer       tex_buffer     = VK_NULL_HANDLE;
    VkDeviceMemory tex_buffer_mem = VK_NULL_HANDLE;
    VkDeviceSize   tex_buffer_sz  = 0;
    bool           has_tex_buffer = false;

    // Optional runtime parameter buffer (binding = 3). Unlike mesh/texture
    // (static, uploaded once), this is kept persistently mapped so a parameter
    // edit is a cheap memcpy via update_params() — no rebuild, no remap.
    VkBuffer       param_buffer     = VK_NULL_HANDLE;
    VkDeviceMemory param_buffer_mem = VK_NULL_HANDLE;
    VkDeviceSize   param_buffer_sz  = 0;
    void*          param_mapped     = nullptr;
    bool           has_param_buffer = false;

    std::string device_name;

    ~Impl() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            if (tex_buffer_mem)  vkFreeMemory(device, tex_buffer_mem, nullptr);
            if (tex_buffer)      vkDestroyBuffer(device, tex_buffer, nullptr);
            if (mesh_buffer_mem) vkFreeMemory(device, mesh_buffer_mem, nullptr);
            if (param_buffer)     vkDestroyBuffer(device, param_buffer, nullptr);
            if (param_buffer_mem) vkFreeMemory(device, param_buffer_mem, nullptr);
            if (mesh_buffer)     vkDestroyBuffer(device, mesh_buffer, nullptr);
            if (staging_mem)  vkFreeMemory(device, staging_mem, nullptr);
            if (staging)      vkDestroyBuffer(device, staging, nullptr);
            if (image_view)   vkDestroyImageView(device, image_view, nullptr);
            if (image_mem)    vkFreeMemory(device, image_mem, nullptr);
            if (image)        vkDestroyImage(device, image, nullptr);
            if (fence)        vkDestroyFence(device, fence, nullptr);
            if (cmd_pool)     vkDestroyCommandPool(device, cmd_pool, nullptr);
            if (desc_pool)    vkDestroyDescriptorPool(device, desc_pool, nullptr);
            if (pipeline)     vkDestroyPipeline(device, pipeline, nullptr);
            if (pipe_layout)  vkDestroyPipelineLayout(device, pipe_layout, nullptr);
            if (set_layout)   vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            if (shader_module)vkDestroyShaderModule(device, shader_module, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE)
            vkDestroyInstance(instance, nullptr);
    }

    // Allocates/recreates the storage image at (w, h). Returns whether
    // we needed to recreate (so the caller can rewrite descriptors).
    std::expected<bool, std::string> ensure_image(int w, int h) {
        if (image != VK_NULL_HANDLE && w == image_w && h == image_h)
            return false;
        // Tear down previous.
        vkDeviceWaitIdle(device);
        if (image_view)  { vkDestroyImageView(device, image_view, nullptr); image_view = VK_NULL_HANDLE; }
        if (image_mem)   { vkFreeMemory(device, image_mem, nullptr);        image_mem  = VK_NULL_HANDLE; }
        if (image)       { vkDestroyImage(device, image, nullptr);          image      = VK_NULL_HANDLE; }
        image_w = w; image_h = h;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent        = { static_cast<std::uint32_t>(w),
                              static_cast<std::uint32_t>(h), 1 };
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_TRY(vkCreateImage(device, &ici, nullptr, &image));

        VkMemoryRequirements mr{};
        vkGetImageMemoryRequirements(device, image, &mr);
        int type = find_memory_type(physical_device, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0) return std::unexpected("no device-local memory");

        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = static_cast<std::uint32_t>(type);
        VK_TRY(vkAllocateMemory(device, &mai, nullptr, &image_mem));
        VK_TRY(vkBindImageMemory(device, image, image_mem, 0));

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        VK_TRY(vkCreateImageView(device, &vci, nullptr, &image_view));

        // Staging buffer (host visible, transfer dst).
        VkDeviceSize sz = static_cast<VkDeviceSize>(w) * h * 4;
        if (staging_mem) { vkFreeMemory(device, staging_mem, nullptr); staging_mem = VK_NULL_HANDLE; }
        if (staging)     { vkDestroyBuffer(device, staging, nullptr); staging = VK_NULL_HANDLE; }
        staging_sz = sz;

        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = sz;
        bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_TRY(vkCreateBuffer(device, &bci, nullptr, &staging));
        VkMemoryRequirements bmr{};
        vkGetBufferMemoryRequirements(device, staging, &bmr);
        int btype = find_memory_type(physical_device, bmr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (btype < 0) return std::unexpected("no host-visible memory");
        VkMemoryAllocateInfo bmai{};
        bmai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        bmai.allocationSize  = bmr.size;
        bmai.memoryTypeIndex = static_cast<std::uint32_t>(btype);
        VK_TRY(vkAllocateMemory(device, &bmai, nullptr, &staging_mem));
        VK_TRY(vkBindBufferMemory(device, staging, staging_mem, 0));

        return true;
    }

    // Update the descriptor set to point at the (possibly new) image view.
    void update_descriptor() {
        VkDescriptorImageInfo dii{};
        dii.imageView   = image_view;
        dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = desc_set;
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.descriptorCount = 1;
        w.pImageInfo      = &dii;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Static probe — returns true if Vulkan can enumerate ≥ 1 physical device.
// ─────────────────────────────────────────────────────────────────────────────
bool VulkanCtx::available() {
    VkApplicationInfo ai{};
    ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "frep4-probe";
    ai.apiVersion       = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) return false;
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    vkDestroyInstance(inst, nullptr);
    return count > 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// create()
// ─────────────────────────────────────────────────────────────────────────────
std::expected<std::unique_ptr<VulkanCtx>, std::string>
VulkanCtx::create(const std::string& spv_path,
                  const std::vector<float>& mesh_voxels,
                  const std::vector<std::uint8_t>& texture_pixels,
                  const std::vector<float>& params)
{
    auto t0 = std::chrono::steady_clock::now();

    auto ctx = std::unique_ptr<VulkanCtx>(new VulkanCtx());
    ctx->impl_ = std::make_unique<Impl>();
    Impl& I    = *ctx->impl_;

    // 1. Instance.
    {
        VkApplicationInfo ai{};
        ai.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.pApplicationName = "frep4-gpu";
        ai.apiVersion       = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{};
        ici.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;
        VK_TRY(vkCreateInstance(&ici, nullptr, &I.instance));
    }

    // 2. Physical device: first with a compute-capable queue.
    {
        std::uint32_t n = 0;
        vkEnumeratePhysicalDevices(I.instance, &n, nullptr);
        if (n == 0)
            return std::unexpected("no Vulkan-capable physical devices");
        std::vector<VkPhysicalDevice> phys(n);
        vkEnumeratePhysicalDevices(I.instance, &n, phys.data());
        for (auto pd : phys) {
            std::uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qf(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
            for (std::uint32_t i = 0; i < qn; ++i) {
                if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    I.physical_device = pd;
                    I.compute_family  = i;
                    break;
                }
            }
            if (I.physical_device) break;
        }
        if (!I.physical_device)
            return std::unexpected("no compute-capable queue family");
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(I.physical_device, &props);
        I.device_name = props.deviceName;
    }

    // 3. Logical device + compute queue.
    {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo dqi{};
        dqi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex = I.compute_family;
        dqi.queueCount       = 1;
        dqi.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{};
        dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &dqi;
        VK_TRY(vkCreateDevice(I.physical_device, &dci, nullptr, &I.device));
        vkGetDeviceQueue(I.device, I.compute_family, 0, &I.compute_queue);
    }

    auto ts_device = std::chrono::steady_clock::now();  // device setup done

    // 4. Shader module from .spv bytes.
    {
        auto bytes = read_binary_file(spv_path);
        if (bytes.empty())
            return std::unexpected("cannot read shader: " + spv_path);
        VkShaderModuleCreateInfo smi{};
        smi.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = bytes.size();
        smi.pCode    = reinterpret_cast<const std::uint32_t*>(bytes.data());
        VK_TRY(vkCreateShaderModule(I.device, &smi, nullptr, &I.shader_module));
    }

    auto ts_shader = std::chrono::steady_clock::now();   // shader module done

    // Mesh storage buffer (optional, only if caller passed voxel data).
    I.has_mesh_buffer = !mesh_voxels.empty();
    I.has_tex_buffer  = !texture_pixels.empty();
    I.has_param_buffer = !params.empty();

    // 5. Descriptor set layout: storage image @ binding 0,
    //    plus optional storage buffer @ binding 1 for mesh voxels,
    //    plus optional storage buffer @ binding 2 for texture pixels.
    {
        std::vector<VkDescriptorSetLayoutBinding> bs;
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bs.push_back(b);
        if (I.has_mesh_buffer) {
            b.binding         = 1;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bs.push_back(b);
        }
        if (I.has_tex_buffer) {
            b.binding         = 2;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bs.push_back(b);
        }
        if (I.has_param_buffer) {
            b.binding         = 3;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bs.push_back(b);
        }
        VkDescriptorSetLayoutCreateInfo dsli{};
        dsli.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsli.bindingCount = static_cast<std::uint32_t>(bs.size());
        dsli.pBindings    = bs.data();
        VK_TRY(vkCreateDescriptorSetLayout(I.device, &dsli, nullptr, &I.set_layout));
    }

    // 6. Pipeline layout (with push-constant range).
    {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(ShaderPush);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount         = 1;
        pli.pSetLayouts            = &I.set_layout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcr;
        VK_TRY(vkCreatePipelineLayout(I.device, &pli, nullptr, &I.pipe_layout));
    }

    // 7. Compute pipeline.
    {
        VkPipelineShaderStageCreateInfo ssi{};
        ssi.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssi.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        ssi.module = I.shader_module;
        ssi.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = ssi;
        cpci.layout = I.pipe_layout;
        VK_TRY(vkCreateComputePipelines(I.device, VK_NULL_HANDLE, 1, &cpci,
                                        nullptr, &I.pipeline));
    }

    auto ts_pipeline = std::chrono::steady_clock::now();  // pipeline (driver compile) done
    // 8. Descriptor pool + set.
    {
        std::vector<VkDescriptorPoolSize> ps;
        ps.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1});
        std::uint32_t sb_count = (I.has_mesh_buffer  ? 1 : 0)
                               + (I.has_tex_buffer   ? 1 : 0)
                               + (I.has_param_buffer ? 1 : 0);
        if (sb_count > 0)
            ps.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sb_count});
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets       = 1;
        dpi.poolSizeCount = static_cast<std::uint32_t>(ps.size());
        dpi.pPoolSizes    = ps.data();
        VK_TRY(vkCreateDescriptorPool(I.device, &dpi, nullptr, &I.desc_pool));
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = I.desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &I.set_layout;
        VK_TRY(vkAllocateDescriptorSets(I.device, &dsai, &I.desc_set));
    }

    // 8.5. Upload mesh voxel data to a host-visible storage buffer (one-time).
    if (I.has_mesh_buffer) {
        I.mesh_buffer_sz = mesh_voxels.size() * sizeof(float);
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = I.mesh_buffer_sz;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_TRY(vkCreateBuffer(I.device, &bci, nullptr, &I.mesh_buffer));
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(I.device, I.mesh_buffer, &mr);
        int t = find_memory_type(I.physical_device, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t < 0) return std::unexpected("no host-visible memory for mesh buffer");
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = static_cast<std::uint32_t>(t);
        VK_TRY(vkAllocateMemory(I.device, &mai, nullptr, &I.mesh_buffer_mem));
        VK_TRY(vkBindBufferMemory(I.device, I.mesh_buffer, I.mesh_buffer_mem, 0));

        void* mapped = nullptr;
        VK_TRY(vkMapMemory(I.device, I.mesh_buffer_mem, 0, I.mesh_buffer_sz,
                           0, &mapped));
        std::memcpy(mapped, mesh_voxels.data(), I.mesh_buffer_sz);
        vkUnmapMemory(I.device, I.mesh_buffer_mem);

        // Bind into descriptor set immediately — image view comes later.
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = I.mesh_buffer;
        dbi.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = I.desc_set;
        w.dstBinding      = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &dbi;
        vkUpdateDescriptorSets(I.device, 1, &w, 0, nullptr);
    }

    // 8.6. Upload texture pixel data (RGBA8 → uint32-packed).
    if (I.has_tex_buffer) {
        I.tex_buffer_sz = texture_pixels.size();  // already bytes
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = I.tex_buffer_sz;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_TRY(vkCreateBuffer(I.device, &bci, nullptr, &I.tex_buffer));
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(I.device, I.tex_buffer, &mr);
        int t = find_memory_type(I.physical_device, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t < 0) return std::unexpected("no host-visible memory for tex buffer");
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = static_cast<std::uint32_t>(t);
        VK_TRY(vkAllocateMemory(I.device, &mai, nullptr, &I.tex_buffer_mem));
        VK_TRY(vkBindBufferMemory(I.device, I.tex_buffer, I.tex_buffer_mem, 0));

        void* mapped = nullptr;
        VK_TRY(vkMapMemory(I.device, I.tex_buffer_mem, 0, I.tex_buffer_sz,
                           0, &mapped));
        std::memcpy(mapped, texture_pixels.data(), I.tex_buffer_sz);
        vkUnmapMemory(I.device, I.tex_buffer_mem);

        VkDescriptorBufferInfo dbi{};
        dbi.buffer = I.tex_buffer;
        dbi.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = I.desc_set;
        w.dstBinding      = 2;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &dbi;
        vkUpdateDescriptorSets(I.device, 1, &w, 0, nullptr);
    }

    // 8.7. Runtime parameter buffer (binding = 3), kept mapped for cheap edits.
    if (I.has_param_buffer) {
        I.param_buffer_sz = params.size() * sizeof(float);
        VkBufferCreateInfo bci{};
        bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size        = I.param_buffer_sz;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_TRY(vkCreateBuffer(I.device, &bci, nullptr, &I.param_buffer));
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(I.device, I.param_buffer, &mr);
        int t = find_memory_type(I.physical_device, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t < 0) return std::unexpected("no host-visible memory for param buffer");
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = static_cast<std::uint32_t>(t);
        VK_TRY(vkAllocateMemory(I.device, &mai, nullptr, &I.param_buffer_mem));
        VK_TRY(vkBindBufferMemory(I.device, I.param_buffer, I.param_buffer_mem, 0));
        // Persistent mapping for the buffer's lifetime (host-coherent).
        VK_TRY(vkMapMemory(I.device, I.param_buffer_mem, 0, I.param_buffer_sz,
                           0, &I.param_mapped));
        std::memcpy(I.param_mapped, params.data(), I.param_buffer_sz);

        VkDescriptorBufferInfo dbi{};
        dbi.buffer = I.param_buffer;
        dbi.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = I.desc_set;
        w.dstBinding      = 3;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &dbi;
        vkUpdateDescriptorSets(I.device, 1, &w, 0, nullptr);
    }

    auto ts_buffers = std::chrono::steady_clock::now();  // descriptor/storage buffers done

    // 9. Command pool + buffer + fence.
    {
        VkCommandPoolCreateInfo cpi{};
        cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = I.compute_family;
        VK_TRY(vkCreateCommandPool(I.device, &cpi, nullptr, &I.cmd_pool));

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = I.cmd_pool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_TRY(vkAllocateCommandBuffers(I.device, &cbai, &I.cmd_buf));

        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_TRY(vkCreateFence(I.device, &fi, nullptr, &I.fence));
    }

    auto t1 = std::chrono::steady_clock::now();
    ctx->stats_.device_name = I.device_name;
    using ms = std::chrono::duration<double, std::milli>;
    ctx->stats_.init_ms          = ms(t1 - t0).count();
    ctx->stats_.init_device_ms   = ms(ts_device   - t0).count();
    ctx->stats_.init_shader_ms   = ms(ts_shader   - ts_device).count();
    ctx->stats_.init_pipeline_ms = ms(ts_pipeline - ts_shader).count();
    ctx->stats_.init_buffers_ms  = ms(ts_buffers  - ts_pipeline).count();
    ctx->stats_.init_misc_ms     = ms(t1          - ts_buffers).count();
    return ctx;
}

VulkanCtx::~VulkanCtx() = default;

// ─────────────────────────────────────────────────────────────────────────────
// render()
// ─────────────────────────────────────────────────────────────────────────────
std::expected<void, std::string>
VulkanCtx::render(const ShaderPush& push, std::vector<std::uint8_t>& out_rgba)
{
    auto t0 = std::chrono::steady_clock::now();
    Impl& I = *impl_;

    int w = push.width;
    int h = push.height;
    if (w <= 0 || h <= 0)
        return std::unexpected("width/height must be > 0");

    auto img_res = I.ensure_image(w, h);
    if (!img_res) return std::unexpected(img_res.error());
    // Always re-bind in case the image was recreated.
    I.update_descriptor();

    // Record the command buffer fresh each render.
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_TRY(vkBeginCommandBuffer(I.cmd_buf, &bi));

    // Image transition: UNDEFINED -> GENERAL (for storage image writes).
    {
        VkImageMemoryBarrier ib{};
        ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ib.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        ib.image         = I.image;
        ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.layerCount = 1;
        ib.srcAccessMask = 0;
        ib.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(I.cmd_buf,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &ib);
    }

    // Bind + push + dispatch.
    vkCmdBindPipeline(I.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, I.pipeline);
    vkCmdBindDescriptorSets(I.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
        I.pipe_layout, 0, 1, &I.desc_set, 0, nullptr);
    vkCmdPushConstants(I.cmd_buf, I.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(ShaderPush), &push);
    // Dispatch only enough workgroups to cover the tile (whole frame when
    // tile_x1/y1 == 0). The shader maps local invocation → tile-relative
    // pixel, so a sub-region launches proportionally fewer invocations
    // rather than the full frame's worth.
    int tile_w = (push.tile_x1 > 0 ? push.tile_x1 : w) - push.tile_x0;
    int tile_h = (push.tile_y1 > 0 ? push.tile_y1 : h) - push.tile_y0;
    if (tile_w <= 0) tile_w = w;
    if (tile_h <= 0) tile_h = h;
    vkCmdDispatch(I.cmd_buf, (tile_w + 7) / 8, (tile_h + 7) / 8, 1);

    // Transition GENERAL -> TRANSFER_SRC_OPTIMAL.
    {
        VkImageMemoryBarrier ib{};
        ib.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        ib.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        ib.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        ib.image         = I.image;
        ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ib.subresourceRange.levelCount = 1;
        ib.subresourceRange.layerCount = 1;
        ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(I.cmd_buf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &ib);
    }

    // Copy image -> staging buffer.
    VkBufferImageCopy bic{};
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent                 = { static_cast<std::uint32_t>(w),
                                        static_cast<std::uint32_t>(h), 1 };
    vkCmdCopyImageToBuffer(I.cmd_buf, I.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, I.staging, 1, &bic);

    VK_TRY(vkEndCommandBuffer(I.cmd_buf));

    // Submit + wait.
    vkResetFences(I.device, 1, &I.fence);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &I.cmd_buf;
    VK_TRY(vkQueueSubmit(I.compute_queue, 1, &si, I.fence));
    VK_TRY(vkWaitForFences(I.device, 1, &I.fence, VK_TRUE, ~0ULL));

    // Map and copy.
    std::size_t expected_size = static_cast<std::size_t>(w) * h * 4;
    out_rgba.resize(expected_size);
    void* mapped = nullptr;
    VK_TRY(vkMapMemory(I.device, I.staging_mem, 0, I.staging_sz, 0, &mapped));
    std::memcpy(out_rgba.data(), mapped, expected_size);
    vkUnmapMemory(I.device, I.staging_mem);

    auto t1 = std::chrono::steady_clock::now();
    stats_.width  = w;
    stats_.height = h;
    stats_.render_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────
// update_params — re-upload the runtime parameter buffer (binding 3). A cheap
// memcpy into the persistently-mapped, host-coherent buffer; no descriptor or
// pipeline change, so an interactive parameter edit reuses the compiled
// pipeline. No-op when the context has no parameter buffer.
void VulkanCtx::update_params(const std::vector<float>& values) {
    Impl& I = *impl_;
    if (!I.has_param_buffer || !I.param_mapped || values.empty()) return;
    VkDeviceSize n = static_cast<VkDeviceSize>(values.size() * sizeof(float));
    if (n > I.param_buffer_sz) n = I.param_buffer_sz;
    std::memcpy(I.param_mapped, values.data(), static_cast<std::size_t>(n));
}

} // namespace frep::gpu
