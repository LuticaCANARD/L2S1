#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
using Json=nlohmann::json;
Json decide(const Json& request,const std::string& id) {
    Json results=Json::array();
    for (const auto& d:request.at("decisions")) {
        auto type=d.at("kind").at("type").get<std::string>(); bool abstain=request.at("state").value("abstain",false);
        Json value={{"type",type}};
        if (type=="binary") value["value"]=abstain?Json(nullptr):Json(request.at("state").value("temperature_c",0)<10);
        else { const auto& options=d.at("kind").at(type=="choice"?"options":"levels"); value["selected"]=abstain?Json(nullptr):options.at(0).at("id"); if(type=="ordinal") value["level_value"]=abstain?Json(nullptr):options.at(0).at("value"); }
        results.push_back({{"id",d.at("id")},{"value",value},{"status",abstain?"abstained":"selected"},{"abstention_reasons",abstain?Json::array({"low_candidate_mass"}):Json::array()},{"evidence",{{"type","model_scored"},{"candidate_mass",0.5},{"custom_field","preserved"}}},{"usage",{{"input_tokens",10}}}});
    }
    return {{"api_version",1},{"request_id",id},{"backend",{{"runtime","fixture"},{"model","fixture"}}},{"policy",nullptr},{"results",results}};
}
int main(int argc,char** argv) {
    Json launch_args=Json::array(); for(int i=1;i<argc;++i) launch_args.push_back(argv[i]);
    std::string mode; for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--model") mode=argv[i+1];
    std::string policy_text; for(int i=1;i+1<argc;++i) if(std::string(argv[i])=="--min-top-probability") policy_text=argv[i+1];
    if(mode=="exit") return 7;
    if(mode=="startup-timeout") std::this_thread::sleep_for(std::chrono::seconds(5));
    std::string line;
    while(std::getline(std::cin,line)) {
        auto call=Json::parse(line); auto id=call.at("id").get<std::string>(); auto op=call.at("op").get<std::string>(); Json result;
        if(op=="health") result={{"status","ok"}};
        else if(op=="capabilities") result={{"api_version",1},{"backend",{{"runtime","fixture"},{"model","fixture"}}},{"decision_types",{"binary","choice","ordinal"}},{"media",Json::object()},{"limits",Json::object()},{"launch_policy",policy_text},{"launch_args",launch_args}};
        else {
            if(mode=="timeout") std::this_thread::sleep_for(std::chrono::seconds(5));
            if(mode=="malformed") {std::cout<<"not json\n"<<std::flush;continue;}
            if(mode=="error") {std::cout<<Json{{"id",id},{"error",{{"message","budget reached"},{"code","reasoning_limit"},{"user_reason","configured reason"}}}}.dump()<<'\n'<<std::flush;continue;}
            if(op=="decide_batch") {Json responses=Json::array(); for(const auto& r:call.at("body").at("requests")) responses.push_back(decide(r,id)); result={{"api_version",1},{"request_id",id},{"execution","native_parallel"},{"responses",responses}};}
            else result=decide(call.at("body"),id);
            if(mode=="wrong-id") id="other";
            if(mode=="wrong-result") result["results"][0]["id"]="other";
        }
        std::cout<<Json{{"id",id},{"result",result}}.dump()<<'\n'<<std::flush;
    }
}
