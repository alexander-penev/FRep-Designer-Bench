// libfive benchmark + scene import/export.
//   --emit-frep DIR                    write canonical scenes as archives
//   --grid N   [extra.frep ...]        grid-eval (canonical + imported)
//   --render R [threads] [extra.frep..]heightmap render
//   --export-hf in.frep out.hf         port an archive to HyperFun (DAG -> assignments)
//   --export-expr in.frep out.txt      port to a single infix expression (frep4 CustomExpr)
#include "scenes_trees.hpp"
#include "../common/timing.hpp"
#include <libfive/tree/archive.hpp>
#include <libfive/tree/data.hpp>
#include <libfive/eval/eval_array.hpp>
#include "../common/field_dump.hpp"
#include <libfive/render/discrete/heightmap.hpp>
#include <libfive/render/discrete/voxels.hpp>
#include <atomic>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
using libfive::Tree; using libfive::Opcode::Opcode;
static std::string base(std::string p){auto s=p.find_last_of('/');if(s!=std::string::npos)p=p.substr(s+1);return p.substr(0,p.find(".frep"));}
static Tree load(const char* p){
    std::ifstream f(p,std::ios::binary);
    try { auto a=libfive::Archive::deserialize(f); return a.shapes.front().tree; }
    catch (std::exception& e){ fprintf(stderr,"skip %s: %s\n",p,e.what()); exit(4);} }
struct Ex { // DAG exporter: id -> tK
    std::map<const void*,std::string> memo; int n=0; std::ostringstream body; int mode=0; // 0=infix flat,1=hf,2=let
    std::string go(Tree t){
        auto key=(const void*)t.id();
        if (auto it=memo.find(key); it!=memo.end()) return it->second;
        auto op=t->op(); std::string a,b;
        const size_t na=libfive::Opcode::args(op);
        if (na>=1) a=go(t->lhs());
        if (na>=2) b=go(t->rhs());
        std::string e;
        switch(op){
        case libfive::Opcode::CONSTANT:{
            // HF parser rejects exponent notation -> plain decimal, trimmed
            char buf[64];snprintf(buf,64,"%.12f",(double)t->value());
            std::string v=buf; auto d=v.find('.');
            auto z=v.find_last_not_of('0'); if(z>d) v.erase(z+1); if(v.back()=='.') v+='0';
            e=v;break;}
        case libfive::Opcode::VAR_X:e=(mode!=1)?"x":"x[1]";break;
        case libfive::Opcode::VAR_Y:e=(mode!=1)?"y":"x[2]";break;
        case libfive::Opcode::VAR_Z:e=(mode!=1)?"z":"x[3]";break;
        case libfive::Opcode::OP_SQUARE:e="("+a+"*"+a+")";break;
        case libfive::Opcode::OP_SQRT:e="sqrt("+a+")";break;
        case libfive::Opcode::OP_NEG:e="(0.0-"+a+")";break;
        case libfive::Opcode::OP_SIN:e="sin("+a+")";break;
        case libfive::Opcode::OP_COS:e="cos("+a+")";break;
        case libfive::Opcode::OP_TAN:e="tan("+a+")";break;
        case libfive::Opcode::OP_ASIN:e="asin("+a+")";break;
        case libfive::Opcode::OP_ACOS:e="acos("+a+")";break;
        case libfive::Opcode::OP_ATAN:e="atan("+a+")";break;
        case libfive::Opcode::OP_EXP:e="exp("+a+")";break;
        case libfive::Opcode::OP_LOG:e="log("+a+")";break;
        case libfive::Opcode::OP_ABS:e="abs("+a+")";break;
        case libfive::Opcode::OP_RECIP:e="(1.0/"+a+")";break;
        case libfive::Opcode::OP_ADD:e="("+a+"+"+b+")";break;
        case libfive::Opcode::OP_SUB:e="("+a+"-"+b+")";break;
        case libfive::Opcode::OP_MUL:e="("+a+"*"+b+")";break;
        case libfive::Opcode::OP_DIV:e="("+a+"/"+b+")";break;
        case libfive::Opcode::OP_MIN:e="min("+a+","+b+")";break;
        case libfive::Opcode::OP_MAX:e="max("+a+","+b+")";break;
        case libfive::Opcode::OP_ATAN2:e="atan2("+a+","+b+")";break;
        case libfive::Opcode::OP_POW:e=(mode!=1)?"pow("+a+","+b+")":"("+a+"^"+b+")";break;
        case libfive::Opcode::OP_NTH_ROOT:e=(mode!=1)?("nth_root("+a+","+b+")"):("nth_root("+a+","+b+")");break;
        case libfive::Opcode::OP_MOD:e="mod("+a+","+b+")";break;
        case libfive::Opcode::OP_NANFILL:e=a;break; // best-effort
        default: fprintf(stderr,"unsupported op %s\n",libfive::Opcode::toString(op).c_str()); exit(3);
        }
        if (mode==0) { // flat infix
            if (e.size()>200000){fprintf(stderr,"expr too large (DAG sharing)\n");exit(5);}
            memo[key]=e; return e; }
        // mode 1 (HF) and 2 (let): bind every non-leaf to a temporary (preserves DAG)
        std::string v="t"+std::to_string(n++);
        body<<"  "<<v<<" = "<<e<<";\n";
        memo[key]=v; return v;
    }
};
int main(int argc, char** argv) {
    auto scenes = canonical_scenes();
    { auto c = complex_scenes(); for (auto& s : c) scenes.push_back(s); }  // valid complex SDFs
    auto add_files=[&](int from){for(int i=from;i<argc;++i)scenes.emplace_back(base(argv[i]),load(argv[i]));};
    // Cross-system visual parity: min-over-Z orthographic SDF field on the shared
    // grid, same projection as frep4/hyperfun -> pixel-comparable images.
    //   bench_libfive --dump-field R Z outdir [extra.frep ...]
    if (argc>=5 && !strcmp(argv[1],"--dump-field")) {
        int R=atoi(argv[2]), Z=atoi(argv[3]); const char* dir=argv[4];
        add_files(5);
        for (auto& [n,t]:scenes){
            libfive::ArrayEvaluator e(t);
            fdump::dump_field([&e](float x,float y,float z){
                e.set({x,y,z},0); return e.values(1)(0); }, R, Z,
                std::string(dir)+"/"+n+"_libfive");
            std::fprintf(stderr,"dumped %s_libfive\n", n.c_str());
        }
        return 0;
    }
    // Ground-truth heightmap render (libfive's native interval evaluator) dumped
    // to PPM, to see whether a .frep archive is a valid SDF at all.
    //   bench_libfive --render-img R outdir [extra.frep ...]
    if (argc>=4 && !strcmp(argv[1],"--render-img")) {
        int R=atoi(argv[2]); const char* dir=argv[3]; add_files(4);
        for (auto& [n,t]:scenes){
            libfive::Voxels vox({-1.6f,-1.6f,-1.6f},{1.6f,1.6f,1.6f},R/3.2f);
            std::atomic_bool ab(false);
            auto h=libfive::Heightmap::render(t,vox,ab,8);
            const auto& d=h->depth; int H=d.rows(), W=d.cols();
            float lo=1e30f,hi=-1e30f; long hit=0;
            for(int i=0;i<H;++i)for(int j=0;j<W;++j){float v=d(i,j);
                if(std::isfinite(v)){lo=std::min(lo,v);hi=std::max(hi,v);++hit;}}
            std::string p=std::string(dir)+"/"+n+"_lfheight.ppm";
            FILE* fp=std::fopen(p.c_str(),"wb"); std::fprintf(fp,"P6\n%d %d\n255\n",W,H);
            for(int i=0;i<H;++i)for(int j=0;j<W;++j){float v=d(i,j);
                unsigned char c = std::isfinite(v)&&hi>lo ? (unsigned char)(40+215*(v-lo)/(hi-lo)) : 0;
                std::fputc(c,fp);std::fputc(c,fp);std::fputc(std::isfinite(v)?c:60,fp);}
            std::fclose(fp);
            std::fprintf(stderr,"%s: %ld/%d surface px (%.1f%%)\n",n.c_str(),hit,H*W,100.0*hit/(H*W));
        }
        return 0;
    }
    if (argc>=3 && !strcmp(argv[1],"--emit-frep")) {
        for (auto& [n,t]:scenes){std::ofstream f(std::string(argv[2])+"/"+n+".frep",std::ios::binary);libfive::Archive a(t);a.serialize(f);}
        return 0;
    }
    if (argc>=4 && !strcmp(argv[1],"--export-hf")) {
        Ex ex; ex.mode=1; auto root=ex.go(load(argv[2]));
        std::ofstream o(argv[3]);
        o<<"my_model(x[3], a[1])\n{\n"<<ex.body.str()<<"  my_model = -("<<root<<");\n}\n";
        printf("hf: %d nodes -> %s\n",ex.n,argv[3]); return 0;
    }
    if (argc>=4 && !strcmp(argv[1],"--export-expr")) {
        Ex ex; ex.mode=0; auto e=ex.go(load(argv[2]));
        std::ofstream o(argv[3]); o<<e<<"\n";
        printf("expr: %zu chars -> %s\n",e.size(),argv[3]); return 0;
    }
    if (argc>=4 && !strcmp(argv[1],"--export-let")) {
        Ex ex; ex.mode=2; auto root=ex.go(load(argv[2]));
        std::ofstream o(argv[3]); o<<ex.body.str()<<root<<"\n";
        printf("let: %d bindings -> %s\n",ex.n,argv[3]); return 0;
    }
    if (argc>=3 && !strcmp(argv[1],"--grid")) {
        const long N=atol(argv[2]); add_files(3);
        for (auto& [n,t]:scenes){
            libfive::ArrayEvaluator e(t);
            const long total=N*N*N;
            auto [ms,J]=median_ms_energy([&]{
                const int B=LIBFIVE_EVAL_ARRAY_SIZE; volatile float sink=0;
                for(long i=0;i<N;++i)for(long j=0;j<N;++j){long k=0;
                    while(k<N){int m=(int)std::min<long>(B,N-k);
                        for(int q=0;q<m;++q)e.set({-1.6f+3.2f*i/(N-1),-1.6f+3.2f*j/(N-1),-1.6f+3.2f*(k+q)/(N-1)},q);
                        auto out=e.values(m);sink+=out(0);k+=m;}}
                (void)sink;});
            csv_row("libfive","cpu-array",n.c_str(),"grid",N,ms,total/ms/1e3,J,J<0?-1:J*1e6/total);
        }
        return 0;
    }
    if (argc>=3 && !strcmp(argv[1],"--render")) {
        const int R=atoi(argv[2]); size_t th=8; int from=3;
        if (argc>=4 && argv[3][0]!='-' && !strstr(argv[3],".frep")){th=atoi(argv[3]);from=4;}
        add_files(from);
        for (auto& [n,t]:scenes){
            libfive::Voxels vox({-1.6f,-1.6f,-1.6f},{1.6f,1.6f,1.6f},R/3.2f);
            std::atomic_bool ab(false);
            auto [ms,J]=median_ms_energy([&]{auto h=libfive::Heightmap::render(t,vox,ab,th);(void)h;});
            double px=(double)R*R;
            csv_row("libfive","cpu-heightmap",n.c_str(),"render3d",R,ms,px/ms/1e3,J,J<0?-1:J*1e6/px);
        }
        return 0;
    }
    fprintf(stderr,"see header for usage\n"); return 1;
}
