#include <l2s1/l2s1.hpp>
#include <algorithm>
#include <filesystem>
#include <iostream>
using namespace l2s1;
void check(bool value) { if(!value) throw std::runtime_error("test assertion failed"); }
template<class F> void fails(const std::string& code,F&& f) { try { f(); } catch(const Error& e) { check(e.code==code); return; } throw std::runtime_error("expected error: "+code); }
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {
        LoadOptions options; options.binary_path=argv[1]; options.model="model with spaces;$(not-a-shell)";
        auto launch=[&]() { auto e=Engine::load(options); return e.capabilities()["launch_args"].get<std::vector<std::string>>(); };
        auto mode=[&](const std::string& expected, bool fixed) {
            auto args=launch(); auto it=std::find(args.begin(),args.end(),"--execution-mode");
            check(std::count(args.begin(),args.end(),"--execution-mode")==1 && it+1!=args.end() && *(it+1)==expected);
            check((std::find(args.begin(),args.end(),"--fixed-schema")!=args.end())==fixed);
        };
        auto defaults=launch(); check(std::find(defaults.begin(),defaults.end(),"--execution-mode")==defaults.end());
        options.fixed_schema=false; mode("fresh",false);
        options.execution_mode="parallel"; mode("parallel",false);
        options.fixed_schema=true; fails("invalid_options",[&]{Engine::load(options);});
        options.execution_mode.reset(); mode("prefix-reuse",true);
        options.execution_mode="prefix-reuse"; mode("prefix-reuse",true);
        options.execution_mode="fresh"; fails("invalid_options",[&]{Engine::load(options);});
        options.fixed_schema.reset(); mode("fresh",false);
        options.execution_mode="parallel";
        options.policy=Policy{0.8123456789123456,0.05123456789123456};
        Request request{{{"temperature_c",6}},{{"cold","Temperature?",Binary{"warm","cold"},std::nullopt}}};
        auto engine=Engine::load(options); check(engine.capabilities()["api_version"]==1);
        check(std::stod(engine.capabilities()["launch_policy"].get<std::string>())==options.policy->min_top_probability);
        auto plan=engine.prepare(request.decisions);
        check(std::get<BinaryValue>(plan.decide({{"temperature_c",6}}).results[0].value).value==true);
        check(std::get<BinaryValue>(plan.decide({{"temperature_c",20}}).results[0].value).value==false);
        auto abstained=plan.decide({{"abstain",true}}); check(!std::get<BinaryValue>(abstained.results[0].value).value);
        check(abstained.results[0].evidence["custom_field"]=="preserved");
        auto batch=plan.decide_batch({{{"temperature_c",2}},{{"temperature_c",20}}}); check(batch.size()==2);
        check(std::get<BinaryValue>(batch[0].results[0].value).value==true); check(std::get<BinaryValue>(batch[1].results[0].value).value==false);
        auto kinds=engine.prepare({{"choice","Pick",Choice{{{"a","A"},{"b","B"}}},std::nullopt},{"level","Rate",Ordinal{{{"low","Low",1.0},{"high","High",2.0}}},std::nullopt}});
        auto values=kinds.decide(Json::object()); check(std::get<ChoiceValue>(values.results[0].value).selected=="a"); check(std::get<OrdinalValue>(values.results[1].value).level_value==1.0);
        check(engine.decide_batch({}).empty()); engine.close(); engine.close(); fails("backend_closed",[&]{engine.health();});
        for(const auto& mode:{"wrong-id","wrong-result","malformed"}) {options.model=mode; auto bad=Engine::load(options); fails("invalid_response",[&]{bad.decide(request);});}
        options.model="error"; auto error=Engine::load(options);
        try {error.decide(request);check(false);} catch(const Error& e) {check(e.code=="reasoning_limit" && e.user_reason=="configured reason");}
        error.health();
        options.model="timeout"; options.timeout=std::chrono::milliseconds(30); auto timeout=Engine::load(options);
        fails("timeout",[&]{timeout.decide(request);}); fails("backend_closed",[&]{timeout.health();});
        options.model="startup-timeout"; options.startup_timeout=std::chrono::milliseconds(30); fails("timeout",[&]{Engine::load(options);});
        options.model="exit"; options.startup_timeout=std::chrono::seconds(2); fails("process_closed",[&]{Engine::load(options);});
        options.extra_args={"--listen=127.0.0.1:0"}; fails("invalid_options",[&]{Engine::load(options);}); options.extra_args.clear();
        options.model=std::string("x\0y",3); fails("invalid_options",[&]{Engine::load(options);});
        options.model="ok"; options.binary_path+="-does-not-exist"; fails("spawn_failed",[&]{Engine::load(options);});
        auto path=std::filesystem::temp_directory_path()/std::filesystem::path("l2s1 cpp spaces $ test"); std::filesystem::create_directories(path);
#ifdef _WIN32
        auto binary=path/"fake engine.exe";
#else
        auto binary=path/"fake engine";
#endif
        std::filesystem::copy_file(argv[1],binary,std::filesystem::copy_options::overwrite_existing);
        options.binary_path=binary.string(); {auto spaces=Engine::load(options);spaces.health();}
        std::filesystem::remove_all(path);
        std::cout<<"C++ SDK contract tests passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
