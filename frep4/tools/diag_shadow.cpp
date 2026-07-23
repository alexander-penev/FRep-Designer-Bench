#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"
#include "core/gpu/glsl_compile.hpp"
#include <cstdio>
#include <vector>
using namespace frep;
int main(){
    if(!gpu::VulkanCtx::available()){printf("no vk\n");return 0;}
    SceneGraph s;
    Material mfloor{{0.5f,0.5f,0.5f}};
    s.add_object(std::make_shared<PlaneNode>(0,1,0,1.0f,"floor"),mfloor);
    Material mcube{{0.2f,0.4f,0.45f}};
    s.add_object(std::make_shared<TranslateNode>(std::make_shared<BoxNode>(0.6f,0.7f,0.6f,"cube"),-2.0f,0.3f,0,"cube_t"),mcube);
    Material msph{{0.55f,0.30f,0.25f}};
    s.add_object(std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.85f,"ball"),0.0f,0.5f,0,"ball_t"),msph);
    s.camera().position={0,1.4f,5.5f}; s.camera().target={0,0.4f,0};
    auto& L=s.lights(); L.clear(); L.push_back({{5,7,4},{1,1,0.95f},1.0f});
    int W=320,H=240;
    auto render_with=[&](TracerConfig cfg, std::vector<std::uint8_t>& px)->bool{
        auto emit=gpu::GlslEmitter::emit(s,cfg); if(!emit)return false;
        auto spv=gpu::compile_glsl_to_spv_managed(emit->source); if(!spv)return false;
        auto ctx=gpu::VulkanCtx::create(spv->path(),emit->mesh_voxels,emit->texture_pixels); if(!ctx)return false;
        return (bool)(**ctx).render(gpu::build_push_from_scene(s,W,H),px);
    };
    auto darkfrac=[&](std::vector<std::uint8_t>&px){int total=0,dark=0;for(int y=int(0.55*H);y<int(0.65*H);++y)for(int x=0;x<W;++x){int i=(y*W+x)*4;int r=px[i],g=px[i+1],b=px[i+2];int mx=r>g?(r>b?r:b):(g>b?g:b),mn=r<g?(r<b?r:b):(g<b?g:b);if(mx-mn>30)continue;if(b>r+20)continue;++total;if(mx<30)++dark;}return total?100.0*dark/total:0.0;};
    std::vector<std::uint8_t> px;
    TracerConfig cfg_on{}; if(!render_with(cfg_on,px)){printf("render fail\n");return 1;}
    printf("shadows ON:  dark floor = %.1f%%\n", darkfrac(px));
    std::vector<std::uint8_t> px2; TracerConfig cfg_off{}; cfg_off.enable_shadows=false;
    if(render_with(cfg_off,px2)) printf("shadows OFF: dark floor = %.1f%%\n", darkfrac(px2));
    TracerConfig cfg_nao{}; cfg_nao.enable_ao=false; std::vector<std::uint8_t> px3;
    if(render_with(cfg_nao,px3)) printf("AO OFF:      dark floor = %.1f%%\n", darkfrac(px3));
    TracerConfig cfg_hard{}; cfg_hard.shadow_samples=4; std::vector<std::uint8_t> px4;
    if(render_with(cfg_hard,px4)) printf("HARD shadows(samples=4): dark floor = %.1f%%\n", darkfrac(px4));
    {   // objects RESTING on the floor (contact shadows): cube center y=-0.3 (bottom=-1), sphere center y=-0.15 (bottom=-1).
        SceneGraph rs; Material mf{{0.5f,0.5f,0.5f}};
        rs.add_object(std::make_shared<PlaneNode>(0,1,0,1.0f,"floor"),mf);
        Material mc{{0.2f,0.4f,0.45f}};
        rs.add_object(std::make_shared<TranslateNode>(std::make_shared<BoxNode>(0.6f,0.7f,0.6f,"cube"),-2.0f,-0.3f,0,"cube_t"),mc);
        Material ms{{0.55f,0.30f,0.25f}};
        rs.add_object(std::make_shared<TranslateNode>(std::make_shared<SphereNode>(0.85f,"ball"),0.0f,-0.15f,0,"ball_t"),ms);
        rs.camera().position={0,1.4f,5.5f}; rs.camera().target={0,0.4f,0};
        auto& rl=rs.lights(); rl.clear(); rl.push_back({{5,7,4},{1,1,0.95f},1.0f});
        TracerConfig c{}; auto e=gpu::GlslEmitter::emit(rs,c);
        if(e){auto sp=gpu::compile_glsl_to_spv_managed(e->source); if(sp){auto cx=gpu::VulkanCtx::create(sp->path(),e->mesh_voxels,e->texture_pixels);
            if(cx){std::vector<std::uint8_t> rp; if((**cx).render(gpu::build_push_from_scene(rs,W,H),rp)) printf("OBJECTS-ON-FLOOR (contact): dark floor = %.1f%%\n", darkfrac(rp));}}}
    }
    {   // floor-ONLY scene: zero occluders. If still dark -> floor self-shadows.
        SceneGraph fs; Material mf{{0.5f,0.5f,0.5f}};
        fs.add_object(std::make_shared<PlaneNode>(0,1,0,1.0f,"floor"),mf);
        fs.camera().position={0,1.4f,5.5f}; fs.camera().target={0,0.4f,0};
        auto& fl=fs.lights(); fl.clear(); fl.push_back({{5,7,4},{1,1,0.95f},1.0f});
        TracerConfig c{}; auto e=gpu::GlslEmitter::emit(fs,c);
        if(e){auto sp=gpu::compile_glsl_to_spv_managed(e->source);
            if(sp){auto cx=gpu::VulkanCtx::create(sp->path(),e->mesh_voxels,e->texture_pixels);
                if(cx){std::vector<std::uint8_t> fp; if((**cx).render(gpu::build_push_from_scene(fs,W,H),fp))
                    printf("FLOOR-ONLY (no occluders): dark floor = %.1f%%\n", darkfrac(fp));}}}
    }
    // ASCII brightness map of the floor band (y 0.50..0.68), downsampled to ~80 cols
    int y0=int(0.50*H), y1=int(0.68*H);
    const char* ramp=" .:-=+*#%@";
    printf("floor band brightness map (rows y=%d..%d):\n",y0,y1);
    for(int y=y0;y<y1;y+=1){
        printf("y%3d ",y);
        for(int x=0;x<W;x+=4){
            int i=(y*W+x)*4; int r=px[i],g=px[i+1],b=px[i+2];
            int mx=r>g?(r>b?r:b):(g>b?g:b), mn=r<g?(r<b?r:b):(g<b?g:b);
            char ch;
            if(b>r+20) ch=' ';                 // sky
            else if(mx-mn>30) ch='O';          // object (saturated)
            else ch=ramp[mx*9/255];            // floor brightness 0..9
            putchar(ch);
        }
        putchar('\n');
    }
    // count dark floor pixels per column-third to see distribution
    int total=0,dark=0; for(int y=int(0.55*H);y<int(0.65*H);++y)for(int x=0;x<W;++x){
        int i=(y*W+x)*4;int r=px[i],g=px[i+1],b=px[i+2];int mx=r>g?(r>b?r:b):(g>b?g:b),mn=r<g?(r<b?r:b):(g<b?g:b);
        if(mx-mn>30)continue; if(b>r+20)continue; ++total; if(mx<30)++dark;}
    printf("dark floor pixels: %d/%d = %.1f%%\n",dark,total,100.0*dark/total);

    // Difference map: where does the shadow ADD darkness? '#'=dark only w/shadows
    {
        std::vector<std::uint8_t> on, off; TracerConfig c_on{}, c_off{}; c_off.enable_shadows=false;
        render_with(c_on,on); render_with(c_off,off);
        auto isfloor=[&](std::vector<std::uint8_t>&px,int i){int r=px[i],g=px[i+1],b=px[i+2];int mx=r>g?(r>b?r:b):(g>b?g:b),mn=r<g?(r<b?r:b):(g<b?g:b);return !(mx-mn>30)&&!(b>r+20);};
        auto bright=[&](std::vector<std::uint8_t>&px,int i){int r=px[i],g=px[i+1],b=px[i+2];return r>g?(r>b?r:b):(g>b?g:b);};
        printf("\nshadow-difference map ('#'=dark only w/ shadows, '.'=dark both, '+'=lit):\n");
        for(int y=int(0.50*H);y<int(0.68*H);++y){printf("y%3d ",y);
            for(int x=0;x<W;x+=4){int i=(y*W+x)*4; char ch;
                bool fon=isfloor(on,i), foff=isfloor(off,i);
                if(!fon&&!foff) ch=' ';
                else { int bon=bright(on,i), boff=bright(off,i);
                    if(bon<30 && boff>=30) ch='#';        // shadow turned it dark
                    else if(bon<30 && boff<30) ch='.';    // dark regardless
                    else ch='+'; }                        // lit
                putchar(ch);} putchar('\n');}
    }
    return 0;
}
