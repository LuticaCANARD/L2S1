#include <l2s1/l2s1.hpp>
#include <iostream>
int main(int argc,char** argv) {
    if (argc!=3) { std::cerr<<"Usage: l2s1_cpp_example /path/to/l2s1 /path/to/model.gguf\n"; return 2; }
    try {
        l2s1::LoadOptions options; options.binary_path=argv[1]; options.model=argv[2]; options.execution_mode="parallel"; options.parallel_width=2;
        auto engine=l2s1::Engine::load(options);
        auto plan=engine.prepare({{"cold","Is temperature_c below 10?",l2s1::Binary{"At least 10.","Below 10."},std::nullopt}});
        auto first=plan.decide({{"temperature_c",6}});
        auto second=plan.decide({{"temperature_c",15}});
        auto batch=plan.decide_batch({{{"temperature_c",2}},{{"temperature_c",20}}});
        for (const auto* response : {&first,&second,&batch[0],&batch[1]}) {
            const auto value=std::get<l2s1::BinaryValue>(response->results.at(0).value).value;
            std::cout<<(value ? (*value ? "true" : "false") : "abstained")<<'\n';
        }
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
