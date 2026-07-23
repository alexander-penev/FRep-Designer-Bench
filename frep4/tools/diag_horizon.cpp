#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/multipath.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/gpu/glsl_compile.hpp"
#include <cstdio>
#include <vector>
using namespace frep;
static inline int u8(float f){int v=int(f*255.0f+0.5f);return v<0?0:(v>255?255:v);}
int main(){
    const int W=200,H=150;
    auto run=[&](int max_steps, std::vector<std::uint8_t>& out){
        SceneGraph s;
        Material mp{{0.5f,0.5f,0.5f}};
        s.add_object(std::make_shared<PlaneNode>(0,1,0,1.0f,"floor"),mp);
        Material msp{{0.85f,0.25f,0.20f}};
        s.add_object(std::make_shared<SphereNode>(1.0f,"ball"),msp);
        s.camera().position={0,0.0f,1.85f}; s.camera().target={0,0.0f,0};
        auto& L=s.lights(); L.clear(); L.push_back({{4,6,4},{1,1,1},1.0f});
        TracerConfig cfg{}; cfg.enable_shadows=false; cfg.enable_ao=false; cfg.max_steps=max_steps;
        exec::CpuIrExecutor ex(SceneCodegen::SceneSdfMode::Inlined, cfg);
        auto r=ex.render(s,W,H,exec::Tile{0,0,W,H});
        if(r.rgba.empty())return false;
        out.resize((size_t)W*H*4);
        for(size_t i=0;i<out.size();++i) out[i]=u8(r.rgba[i]);
        return true;
    };
    auto run_gpu=[&](std::vector<std::uint8_t>& out){
        if(!gpu::VulkanCtx::available())return false;
        SceneGraph s; Material mp{{0.5f,0.5f,0.5f}};
        s.add_object(std::make_shared<PlaneNode>(0,1,0,1.0f,"floor"),mp);
        Material msp{{0.85f,0.25f,0.20f}}; s.add_object(std::make_shared<SphereNode>(1.0f,"ball"),msp);
        s.camera().position={0,0.0f,1.85f}; s.camera().target={0,0.0f,0};
        auto& L=s.lights(); L.clear(); L.push_back({{4,6,4},{1,1,1},1.0f});
        TracerConfig cfg{}; cfg.enable_shadows=false; cfg.enable_ao=false;
        auto e=gpu::GlslEmitter::emit(s,cfg); if(!e)return false;
        auto sp=gpu::compile_glsl_to_spv_managed(e->source); if(!sp)return false;
        auto cx=gpu::VulkanCtx::create(sp->path(),e->mesh_voxels,e->texture_pixels); if(!cx)return false;
        std::vector<std::uint8_t> px; if(!(**cx).render(gpu::build_push_from_scene(s,W,H),px))return false;
        out=px; return true;
    };
    auto sky_in_window=[&](std::vector<std::uint8_t>&px){int holes=0,tot=0;
        for(int y=int(0.45*H);y<int(0.78*H);++y)for(int x=int(0.28*W);x<int(0.72*W);++x){
            int i=(y*W+x)*4;int r=px[i],g=px[i+1],b=px[i+2];++tot; if(b>r+15&&b>g)++holes;}
        return std::pair<int,int>{holes,tot};};
    std::vector<std::uint8_t> a,bb;
    if(!run(192,a)){printf("render fail\n");return 1;}
    run(2048,bb);
    auto[ha,ta]=sky_in_window(a); auto[hb,tb]=sky_in_window(bb);
    printf("max_steps=192 : sky-in-window = %d/%d (%.1f%%)\n",ha,ta,100.0*ha/ta);
    printf("max_steps=2048: sky-in-window = %d/%d (%.1f%%)\n",hb,tb,100.0*hb/tb);
    const char* ramp=" .:-=+*#%@";
    printf("\nmap (max_steps=192): 'O'=sphere, mid chars=floor, ' '=sky\n");
    for(int y=int(0.40*H);y<int(0.80*H);++y){printf("y%3d ",y);
        for(int x=int(0.0*W);x<int(0.40*W);++x){int i=(y*W+x)*4;int r=a[i],g=a[i+1],b=a[i+2];
            int mx=r>g?(r>b?r:b):(g>b?g:b);char ch;
            if(b>r+15&&b>g)ch=' '; else if(r>g+30&&r>b+30)ch='O'; else ch=ramp[mx*9/255];
            putchar(ch);} putchar('\n');}
    // Compare CPU vs GPU brightness along a horizontal scanline through the
    // halo (just right of the sphere edge, on the floor).
    std::vector<std::uint8_t> gp;
    if(run_gpu(gp)){
        int yline=int(0.60*H);
        printf("\nfloor brightness along y=%d, columns near sphere edge (CPU vs GPU):\n",yline);
        printf(" col |");for(int x=int(0.50*W);x<int(0.78*W);x+=2)printf("%4d",x);printf("\n");
        printf(" CPU |");for(int x=int(0.50*W);x<int(0.78*W);x+=2){int i=(yline*W+x)*4;int mx=a[i];printf("%4d",mx);}printf("\n");
        printf(" GPU |");for(int x=int(0.50*W);x<int(0.78*W);x+=2){int i=(yline*W+x)*4;int mx=gp[i];printf("%4d",mx);}printf("\n");
    } else printf("\n(GPU unavailable)\n");
    return 0;
}
