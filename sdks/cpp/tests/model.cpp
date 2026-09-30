#include <l2s1/l2s1.hpp>
#include <iostream>
using namespace l2s1;
void check(bool value) { if (!value) throw std::runtime_error("native reuse assertion failed"); }
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: model-test BINARY MODEL cpu|cuda\n"; return 2; }
    try {
        for (bool fresh : {false, true}) {
            LoadOptions options; options.binary_path=argv[1]; options.model=argv[2]; options.device=argv[3];
            if (fresh) options.fixed_schema=false;
            auto engine=Engine::load(options);
            auto caps=engine.capabilities();
            if (!fresh) check(caps["prefix_reuse"]["schema_change"]=="clear_all");
            std::vector<Decision> decisions{{"cold","Is temperature_c below 10?",Binary{"At least 10.","Below 10."},std::nullopt}};
            auto plan=engine.prepare(decisions);
            auto cold=plan.decide({{"temperature_c",6}});
            auto warm=plan.decide({{"temperature_c",6}});
            check(cold.results[0].evidence==warm.results[0].evidence);
            auto reused=warm.results[0].usage.at("reused_prefix_tokens").get<unsigned>();
            check((reused>0)==!fresh);
            check((plan.decide({{"temperature_c",20}}).results[0].usage["reused_prefix_tokens"].get<unsigned>()>0)==!fresh);
            decisions[0].id="changed_schema";
            auto other=engine.prepare(decisions);
            check(other.decide({{"temperature_c",6}}).results[0].usage["reused_prefix_tokens"]==0);
            check(plan.decide({{"temperature_c",6}}).results[0].usage["reused_prefix_tokens"]==0);
            std::string oversized; for (unsigned i=0;i<4096;++i) oversized+="word ";
            bool failed=false; try { plan.decide({{"oversized",oversized}}); } catch (const Error& e) { failed=true; check(e.code=="invalid_request" || e.code=="backend_error"); }
            check(failed);
            check(plan.decide({{"temperature_c",6}}).results[0].usage["reused_prefix_tokens"]==0);
            std::cout << Json{{"sdk","cpp"},{"transport","stdio"},{"fresh",fresh},{"reused_prefix_tokens",reused},{"schema_return_tokens",0},{"recovery_tokens",0}}.dump() << '\n';
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
