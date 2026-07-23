// core/gpu/rtx_query.cpp — see rtx_query.hpp.

#include "core/gpu/rtx_query.hpp"
#include "core/gpu/glsl_emitter.hpp"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstring>
#include <sstream>

namespace frep::gpu {

namespace {

using clk = std::chrono::steady_clock;
double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Lift the self-contained math/shading region (helpers + scene_sdf + scene_sdf_v
// + scene_normal + shade + sky_color_s) from a compute source, dropping the
// #version/bindings preamble and main(). Same rule as rtx_shaders' lift, including
// the instance/template subprogram start-adjustment so structured scenes work.
std::string lift_shared(const std::string& comp, std::string& err) {
    std::size_t main_pos = comp.find("void main()");
    std::size_t start    = comp.find("float scene_sdf");
    for (const char* helper : {"float sample_mesh_", "vec3 _unpack_rgb",
                               "vec3 sample_texture", "float sample_texture",
                               "float _inst_fn_", "Dual _inst_grad_fn_",
                               "float frep_tmpl_"}) {
        std::size_t h = comp.find(helper);
        if (h != std::string::npos && h < start) start = h;
    }
    if (start == std::string::npos || main_pos == std::string::npos ||
        start >= main_pos) {
        err = "rtx_query: could not locate shared region in compute source";
        return {};
    }
    return comp.substr(start, main_pos - start);
}

// Lift the exact push-constant block so the compute stage reads camera/lights at
// the same offsets the SBT stages do (and the same RtPushConstants layout).
std::string lift_push(const std::string& comp, std::string& err) {
    std::size_t p = comp.find("layout(push_constant)");
    if (p == std::string::npos) { err = "rtx_query: no push_constant block"; return {}; }
    std::size_t end = comp.find("} pc;", p);
    if (end == std::string::npos) { err = "rtx_query: malformed push block"; return {}; }
    return comp.substr(p, (end + 5) - p) + "\n";
}

int find_mem(VkPhysicalDevice phys, std::uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

}  // namespace

std::expected<std::string, std::string>
emit_ray_query_compute(const SceneGraph& scene, const TracerConfig& cfg) {
    auto comp = GlslEmitter::emit(scene, cfg);
    if (!comp) return std::unexpected("rtx_query: " + comp.error());
    if (comp->mesh_count > 0 || comp->texture_count > 0)
        return std::unexpected("rtx_query: mesh/texture scenes not supported yet");

    std::string err;
    std::string shared = lift_shared(comp->source, err);
    if (!err.empty()) return std::unexpected(err);
    std::string push = lift_push(comp->source, err);
    if (!err.empty()) return std::unexpected(err);

    std::ostringstream s;
    s << "#version 460\n"
      << "#extension GL_EXT_ray_query : require\n"
      << "layout(local_size_x = 8, local_size_y = 8) in;\n"
      << push
      << shared
      << "layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;\n"
      << "layout(set = 0, binding = 1, rgba32f) uniform image2D out_img;\n"
      << "void main() {\n"
      << "    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);\n"
      << "    if (pix.x >= pc.width || pix.y >= pc.height) return;\n"
      // Identical primary-ray formula to the SBT raygen / compute emitter.
      << "    float u = (2.0 * float(pix.x) / float(pc.width) - 1.0)\n"
      << "            * (float(pc.width) / float(pc.height));\n"
      << "    float v = 1.0 - 2.0 * float(pix.y) / float(pc.height);\n"
      << "    vec3 ro, rd;\n"
      << "    if (pc.projection_mode > 0.5) {\n"
      << "        ro = pc.cam_pos + pc.cam_right * (u * pc.ortho_size)\n"
      << "                        + pc.cam_up    * (v * pc.ortho_size);\n"
      << "        rd = normalize(pc.cam_fwd);\n"
      << "    } else {\n"
      << "        ro = pc.cam_pos;\n"
      << "        rd = normalize(pc.cam_fwd + pc.cam_right * (u * pc.fov_scale)\n"
      << "                                  + pc.cam_up    * (v * pc.fov_scale));\n"
      << "    }\n"
      << "    vec3 color = sky_color_s(0.5 + 0.5 * v);\n"
      << "    float tmin = 0.001;\n"
      << "    float tmax = " << cfg.max_dist << ";\n"
      // Hardware BVH broad phase: rayQueryEXT walks the TLAS. On each candidate
      // procedural AABB we sphere-trace scene_sdf INLINE (full compute occupancy,
      // no intersection-shader stage). Committing a hit shrinks the ray interval,
      // so with multiple boxes the nearest wins automatically.
      << "    rayQueryEXT rq;\n"
      << "    rayQueryInitializeEXT(rq, tlas, gl_RayFlagsNoneEXT, 0xFF,\n"
      << "                          ro, tmin, rd, tmax);\n"
      << "    while (rayQueryProceedEXT(rq)) {\n"
      << "        if (rayQueryGetIntersectionTypeEXT(rq, false)\n"
      << "            == gl_RayQueryCandidateIntersectionAABBEXT) {\n"
      << "            float t = tmin;\n"
      << "            bool hit = false; float last_d = 1e30;\n"
      << "            float step_len = 0.0; float omega = " << cfg.over_relax << ";\n"
      << "            for (int i = 0; i < " << cfg.max_steps << "; ++i) {\n"
      << "                if (t > tmax) break;\n"
      << "                vec3 p = ro + rd * t;\n"
      << "                float d = scene_sdf_v(p);\n"
      << "                float radius = d * " << cfg.safety_factor << ";\n"
      << "                bool sor_fail = (omega > 1.0) && ((radius + last_d) < step_len);\n"
      << "                step_len = sor_fail ? (step_len * (1.0 - omega)) : (radius * omega);\n"
      << "                omega = sor_fail ? 1.0 : omega;\n"
      << "                if (d < " << cfg.epsilon << " && !sor_fail) { hit = true; last_d = d; break; }\n"
      << "                last_d = d;\n"
      << "                t += step_len;\n"
      << "            }\n"
      << "            if (!hit && t <= tmax && last_d < " << (cfg.epsilon * 80.0f) << ") hit = true;\n"
      << "            if (hit && t <= tmax) rayQueryGenerateIntersectionEXT(rq, t);\n"
      << "        }\n"
      << "    }\n"
      << "    if (rayQueryGetIntersectionTypeEXT(rq, true)\n"
      << "        == gl_RayQueryCommittedIntersectionGeneratedEXT) {\n"
      << "        float th = rayQueryGetIntersectionTEXT(rq, true);\n"
      << "        vec3 p = ro + rd * th;\n"
      << "        vec3 n = scene_normal(p);\n"
      << "        color = shade(p, n, -rd);\n"
      << "    }\n"
      << "    imageStore(out_img, pix, vec4(color, 1.0));\n"
      << "}\n";
    return s.str();
}

std::expected<RtRenderResult, std::string>
rtx_query_trace(const RtxCtx& ctx, const RtAccel& accel,
                const std::vector<std::uint32_t>& comp,
                const RtPushConstants& pc, int width, int height) {
    if (!ctx.has_ray_query())
        return std::unexpected("rtx_query: VK_KHR_ray_query not enabled on device");
    VkDevice dev = ctx.device();
    auto t_pipe0 = clk::now();

    // ── Compute pipeline: shader module + descriptor layout + pipeline ──────
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = comp.size() * sizeof(std::uint32_t);
    smci.pCode = comp.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &smci, nullptr, &module) != VK_SUCCESS)
        return std::unexpected("rtx_query: vkCreateShaderModule failed");

    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2; dslci.pBindings = binds;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0; pcr.size = sizeof(RtPushConstants);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(dev, &plci, nullptr, &playout);

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = playout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline)
        != VK_SUCCESS)
        return std::unexpected("rtx_query: vkCreateComputePipelines failed");

    // ── Output image (rgba32f, storage) ─────────────────────────────────────
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ici.extent = { (std::uint32_t)width, (std::uint32_t)height, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(dev, &ici, nullptr, &image) != VK_SUCCESS)
        return std::unexpected("rtx_query: vkCreateImage failed");
    VkMemoryRequirements imr{};
    vkGetImageMemoryRequirements(dev, image, &imr);
    int it = find_mem(ctx.physical_device(), imr.memoryTypeBits,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo imai{};
    imai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imai.allocationSize = imr.size; imai.memoryTypeIndex = (std::uint32_t)it;
    VkDeviceMemory image_mem = VK_NULL_HANDLE;
    vkAllocateMemory(dev, &imai, nullptr, &image_mem);
    vkBindImageMemory(dev, image, image_mem, 0);

    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = image; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView image_view = VK_NULL_HANDLE;
    vkCreateImageView(dev, &ivci, nullptr, &image_view);

    // ── Descriptor pool + set: TLAS (0) + storage image (1) ─────────────────
    VkDescriptorPoolSize ps[2]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; ps[0].descriptorCount = 1;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; ps[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &dsai, &dset);

    VkAccelerationStructureKHR tlas = accel.tlas();
    VkWriteDescriptorSetAccelerationStructureKHR was{};
    was.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    was.accelerationStructureCount = 1; was.pAccelerationStructures = &tlas;
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.pNext = &was; w0.dstSet = dset; w0.dstBinding = 0;
    w0.descriptorCount = 1; w0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    VkDescriptorImageInfo dii{};
    dii.imageView = image_view; dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w1{};
    w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet = dset; w1.dstBinding = 1;
    w1.descriptorCount = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w1.pImageInfo = &dii;
    VkWriteDescriptorSet writes[2] = {w0, w1};
    vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);

    // ── Command buffer: barrier, dispatch, copy to host buffer ──────────────
    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = ctx.queue_family();
    VkCommandPool cpool = VK_NULL_HANDLE;
    vkCreateCommandPool(dev, &cpi, nullptr, &cpool);
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev, &cbai, &cmd);

    VkDeviceSize img_bytes = (VkDeviceSize)width * height * 4 * sizeof(float);
    VkBuffer rb_buf = VK_NULL_HANDLE; VkDeviceMemory rb_mem = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = img_bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(dev, &bci, nullptr, &rb_buf) != VK_SUCCESS)
            return std::unexpected("rtx_query: readback vkCreateBuffer failed");
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, rb_buf, &mr);
        // Prefer HOST_CACHED for the readback (write-combined memory reads at
        // ~300 MB/s on NVIDIA; see mk_readback_buf in rtx_pipeline.cpp).
        int t = find_mem(ctx.physical_device(), mr.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t < 0) t = find_mem(ctx.physical_device(), mr.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size; mai.memoryTypeIndex = (std::uint32_t)t;
        vkAllocateMemory(dev, &mai, nullptr, &rb_mem);
        vkBindBufferMemory(dev, rb_buf, rb_mem, 0);
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.image = image;
    toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toGeneral.srcAccessMask = 0;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toGeneral);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            playout, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(RtPushConstants), &pc);
    std::uint32_t gx = ((std::uint32_t)width  + 7) / 8;
    std::uint32_t gy = ((std::uint32_t)height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    VkImageMemoryBarrier toSrc = toGeneral;
    toSrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toSrc);

    VkBufferImageCopy copy{};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { (std::uint32_t)width, (std::uint32_t)height, 1 };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           rb_buf, 1, &copy);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(dev, &fci, nullptr, &fence);
    auto t_trace0 = clk::now();
    vkQueueSubmit(ctx.queue(), 1, &si, fence);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull);
    auto t_trace1 = clk::now();

    RtRenderResult out;
    out.width = width; out.height = height;
    out.rgba.resize((std::size_t)width * height * 4);
    auto t_rb0 = clk::now();
    {
        void* p = nullptr;
        vkMapMemory(dev, rb_mem, 0, img_bytes, 0, &p);
        std::memcpy(out.rgba.data(), p, (size_t)img_bytes);
        vkUnmapMemory(dev, rb_mem);
    }
    auto t_rb1 = clk::now();
    out.pipeline_ms = ms(t_pipe0, t_trace0);
    out.trace_ms    = ms(t_trace0, t_trace1);
    out.readback_ms = ms(t_rb0, t_rb1);

    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyImageView(dev, image_view, nullptr);
    vkDestroyImage(dev, image, nullptr);
    vkFreeMemory(dev, image_mem, nullptr);
    vkDestroyBuffer(dev, rb_buf, nullptr);
    vkFreeMemory(dev, rb_mem, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyPipelineLayout(dev, playout, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, module, nullptr);
    return out;
}

}  // namespace frep::gpu
