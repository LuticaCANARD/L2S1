#include <l2s1/l2s1.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <future>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <thread>
#include <type_traits>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace l2s1 {
Error::Error(std::string message, std::string error_code, std::string id, std::string reason)
    : std::runtime_error(std::move(message)), code(std::move(error_code)), request_id(std::move(id)), user_reason(std::move(reason)) {}
Json to_json(const Decision& d) {
    Json kind = std::visit([](const auto& k) -> Json {
        using T = std::decay_t<decltype(k)>;
        if constexpr (std::is_same_v<T,Binary>) return {{"type","binary"},{"false_label",k.false_label},{"true_label",k.true_label}};
        else if constexpr (std::is_same_v<T,Choice>) {
            Json options = Json::array(); for (const auto& o : k.options) options.push_back({{"id",o.id},{"criterion",o.criterion}});
            return {{"type","choice"},{"options",options}};
        } else {
            Json levels = Json::array(); for (const auto& l : k.levels) levels.push_back({{"id",l.id},{"criterion",l.criterion},{"value",l.value}});
            return {{"type","ordinal"},{"levels",levels}};
        }
    }, d.kind);
    Json value = {{"id",d.id},{"instruction",d.instruction},{"kind",kind}};
    if (d.media_ids) value["media_ids"] = *d.media_ids;
    return value;
}
Json to_json(const Request& request) {
    if (!request.options.is_object()) throw Error("request options must be an object","invalid_request");
    Json value = request.options; value["state"] = request.state; value["decisions"] = Json::array();
    for (const auto& decision : request.decisions) value["decisions"].push_back(to_json(decision));
    return value;
}
namespace {
constexpr std::size_t max_request = 44*1024*1024+1024, max_response = 64*1024*1024;
void valid_timeout(std::chrono::milliseconds timeout) {
    if (timeout.count() < 1 || timeout.count() > INT_MAX) throw Error("timeout must be in 1..2147483647 ms","invalid_options");
}
void valid_argument(const std::string& arg) {
    if (arg.find('\0') != std::string::npos) throw Error("process arguments must not contain NUL","invalid_options");
}
std::string precise(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return out.str();
}
std::vector<std::string> arguments(const LoadOptions& o) {
    if (o.model.empty() || o.binary_path.empty()) throw Error("model and binary_path are required","invalid_options");
    valid_timeout(o.startup_timeout); valid_timeout(o.timeout);
    if (o.device != "cpu" && o.device != "cuda" && o.device != "metal") throw Error("invalid device","invalid_options");
    std::vector<std::string> args{o.binary_path,"--model",o.model,"--device",o.device};
    if (o.fixed_schema.value_or(false)) {
        if (o.execution_mode && *o.execution_mode != "prefix-reuse") throw Error("fixed_schema requires prefix-reuse","invalid_options");
        args.emplace_back("--fixed-schema");
        if (!o.execution_mode) args.insert(args.end(),{"--execution-mode","prefix-reuse"});
    }
    if (o.fixed_schema == false && !o.execution_mode) args.insert(args.end(),{"--execution-mode","fresh"});
    if (o.execution_mode) args.insert(args.end(),{"--execution-mode",*o.execution_mode});
    auto number = [&](const char* flag, auto value) { if (value) { args.emplace_back(flag); args.push_back(std::to_string(*value)); } };
    auto string = [&](const char* flag, const auto& value) { if (value) { args.emplace_back(flag); args.push_back(*value); } };
    string("--mmproj",o.mmproj); string("--lora",o.lora);
    number("--context",o.context); number("--batch",o.batch); number("--ubatch",o.ubatch); number("--threads",o.threads); number("--parallel-width",o.parallel_width); number("--gpu-layers",o.gpu_layers);
    if (o.policy) { args.insert(args.end(),{"--min-top-probability",precise(o.policy->min_top_probability),"--min-candidate-mass",precise(o.policy->min_candidate_mass)}); }
    for (const auto& arg : o.extra_args) {
        if (arg == "--" || arg == "--stdio" || arg.rfind("--stdio=",0) == 0 || arg == "--listen" || arg.rfind("--listen=",0) == 0) throw Error("--stdio/--listen are managed by Engine::load","invalid_options");
        args.push_back(arg);
    }
    args.emplace_back("--stdio"); for (const auto& arg : args) valid_argument(arg); return args;
}
#ifdef _WIN32
std::wstring wide(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),static_cast<int>(s.size()),nullptr,0);
    if (!n && !s.empty()) throw Error("invalid UTF-8 argument","invalid_options");
    std::wstring out(n,L'\0'); MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),static_cast<int>(s.size()),out.data(),n); return out;
}
std::wstring quote(const std::string& value) {
    auto s = wide(value); std::wstring out=L"\""; std::size_t slashes=0;
    for (wchar_t c : s) {
        if (c == L'\\') { ++slashes; continue; }
        out.append(c == L'"' ? slashes*2+1 : slashes,L'\\'); slashes=0; out += c;
    }
    out.append(slashes*2,L'\\'); return out+L'"';
}
#endif
struct Process {
#ifdef _WIN32
    HANDLE process=nullptr, input=nullptr, output=nullptr;
#else
    pid_t pid=-1;
    int socket=-1;
#endif
    Process()=default; Process(const Process&)=delete; Process& operator=(const Process&)=delete;
    ~Process() { close(); }
    void start(const std::vector<std::string>& args) {
#ifdef _WIN32
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE}; HANDLE child_in=nullptr, child_out=nullptr;
        if (!CreatePipe(&child_in,&input,&security,65536)) throw Error("CreatePipe failed","spawn_failed");
        if (!CreatePipe(&output,&child_out,&security,65536)) { CloseHandle(child_in); close(); throw Error("CreatePipe failed","spawn_failed"); }
        SetHandleInformation(input,HANDLE_FLAG_INHERIT,0); SetHandleInformation(output,HANDLE_FLAG_INHERIT,0);
        STARTUPINFOEXW startup{}; startup.StartupInfo.cb=sizeof(startup); startup.StartupInfo.dwFlags=STARTF_USESTDHANDLES;
        // Duplicate stderr so its lifetime is explicit and only these three handles are inherited.
        HANDLE child_err=nullptr;
        if (!DuplicateHandle(GetCurrentProcess(),GetStdHandle(STD_ERROR_HANDLE),GetCurrentProcess(),&child_err,0,TRUE,DUPLICATE_SAME_ACCESS)) {
            child_err=CreateFileW(L"NUL",GENERIC_WRITE,FILE_SHARE_WRITE,&security,OPEN_EXISTING,0,nullptr);
        }
        startup.StartupInfo.hStdInput=child_in; startup.StartupInfo.hStdOutput=child_out; startup.StartupInfo.hStdError=child_err;
        SIZE_T size=0; InitializeProcThreadAttributeList(nullptr,1,0,&size); std::vector<unsigned char> attributes(size);
        startup.lpAttributeList=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
        HANDLE handles[]{child_in,child_out,child_err};
        bool initialized=InitializeProcThreadAttributeList(startup.lpAttributeList,1,0,&size)!=0;
        bool prepared=initialized && UpdateProcThreadAttribute(startup.lpAttributeList,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,handles,sizeof(handles),nullptr,nullptr)!=0;
        std::wstring command; for (const auto& arg : args) { if (!command.empty()) command+=L' '; command+=quote(arg); }
        PROCESS_INFORMATION info{};
        bool ok=prepared && CreateProcessW(nullptr,command.data(),nullptr,nullptr,TRUE,EXTENDED_STARTUPINFO_PRESENT|CREATE_NO_WINDOW,nullptr,nullptr,&startup.StartupInfo,&info)!=0;
        if (initialized) DeleteProcThreadAttributeList(startup.lpAttributeList);
        CloseHandle(child_in); CloseHandle(child_out); if (child_err && child_err!=INVALID_HANDLE_VALUE) CloseHandle(child_err);
        if (!ok) { close(); throw Error("could not launch Rust executable","spawn_failed"); }
        process=info.hProcess; CloseHandle(info.hThread);
#else
        int sockets[2];
        if (::socketpair(AF_UNIX,SOCK_STREAM,0,sockets) != 0) throw Error(std::strerror(errno),"spawn_failed");
        // Move descriptors above stdio, including in applications with closed fd 0/1.
        for (int& fd : sockets) { int moved=fcntl(fd,F_DUPFD_CLOEXEC,3); if (moved<0) { for (int f:sockets) ::close(f); throw Error("could not reserve process descriptor","spawn_failed"); } ::close(fd); fd=moved; }
        socket=sockets[0];
#ifdef __APPLE__
        int one=1; setsockopt(socket,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
#endif
        posix_spawn_file_actions_t actions;
        int rc=posix_spawn_file_actions_init(&actions);
        if (rc!=0) { ::close(sockets[1]); close(); throw Error(std::strerror(rc),"spawn_failed"); }
        rc=posix_spawn_file_actions_adddup2(&actions,sockets[1],STDIN_FILENO);
        if (!rc) rc=posix_spawn_file_actions_adddup2(&actions,sockets[1],STDOUT_FILENO);
        if (!rc) rc=posix_spawn_file_actions_addclose(&actions,sockets[0]);
        if (!rc) rc=posix_spawn_file_actions_addclose(&actions,sockets[1]);
        std::vector<char*> argv; for (const auto& arg:args) argv.push_back(const_cast<char*>(arg.c_str())); argv.push_back(nullptr);
        if (!rc) rc=posix_spawnp(&pid,argv[0],&actions,nullptr,argv.data(),environ);
        posix_spawn_file_actions_destroy(&actions); ::close(sockets[1]);
        if (rc) { pid=-1; close(); throw Error(std::strerror(rc),"spawn_failed"); }
#endif
    }
    void interrupt() noexcept {
#ifdef _WIN32
        if (process) TerminateProcess(process,1);
#else
        if (socket>=0) shutdown(socket,SHUT_RDWR);
        if (pid>0) kill(pid,SIGKILL);
#endif
    }
    void close() noexcept {
        interrupt();
#ifdef _WIN32
        if (input) CloseHandle(input); if (output) CloseHandle(output);
        if (process) { WaitForSingleObject(process,INFINITE); CloseHandle(process); }
        process=input=output=nullptr;
#else
        if (socket>=0) ::close(socket);
        if (pid>0) { int status; while (waitpid(pid,&status,0)<0 && errno==EINTR) {} }
        socket=-1; pid=-1;
#endif
    }
    void write_all(const std::string& text) {
        std::size_t offset=0;
        while (offset<text.size()) {
#ifdef _WIN32
            DWORD written=0; if (!WriteFile(input,text.data()+offset,static_cast<DWORD>(std::min<std::size_t>(text.size()-offset,65536)),&written,nullptr) || !written) throw Error("Rust stdin closed","process_closed");
            offset+=written;
#else
#ifdef MSG_NOSIGNAL
            constexpr int flags=MSG_NOSIGNAL;
#else
            constexpr int flags=0;
#endif
            auto n=send(socket,text.data()+offset,text.size()-offset,flags);
            if (n<0 && errno==EINTR) continue;
            if (n<=0) throw Error("Rust stdin closed","process_closed");
            offset+=static_cast<std::size_t>(n);
#endif
        }
    }
    std::string read_line() {
        std::string line; std::array<char,8192> buffer{};
        for (;;) {
#ifdef _WIN32
            DWORD bytes=0; if (!ReadFile(output,buffer.data(),static_cast<DWORD>(buffer.size()),&bytes,nullptr) || !bytes) throw Error("Rust stdout closed","process_closed");
            std::size_t n=bytes;
#else
            auto bytes=recv(socket,buffer.data(),buffer.size(),0);
            if (bytes<0 && errno==EINTR) continue;
            if (bytes<=0) throw Error("Rust stdout closed","process_closed");
            auto n=static_cast<std::size_t>(bytes);
#endif
            if (line.size()+n>max_response) throw Error("stdio response exceeds 64 MiB","invalid_response");
            line.append(buffer.data(),n); auto newline=line.find('\n');
            if (newline!=std::string::npos) {
                if (newline+1!=line.size()) throw Error("unsolicited stdio data","invalid_response");
                line.resize(newline); return line;
            }
        }
    }
};
Response response(const Json& value,const Json& request) {
    try {
        if (value.at("api_version")!=1 || !value.at("request_id").is_string() || !value.at("backend").at("runtime").is_string() || !value.at("backend").at("model").is_string() || !value.contains("policy") || !value.at("results").is_array() || value.at("results").size()!=request.at("decisions").size()) throw std::runtime_error("envelope");
        Response out{value.at("request_id").get<std::string>(),{},value};
        for (std::size_t i=0;i<value.at("results").size();++i) {
            const auto& r=value.at("results").at(i); const auto& d=request.at("decisions").at(i); const auto& v=r.at("value");
            auto status=r.at("status").get<std::string>(); auto type=v.at("type").get<std::string>();
            if (r.at("id")!=d.at("id") || type!=d.at("kind").at("type") || (status!="selected" && status!="abstained") || !r.at("usage").is_object()) throw std::runtime_error("result");
            auto evidence_type=r.at("evidence").at("type").get<std::string>();
            if (evidence_type!="model_scored" && evidence_type!="selection_only") throw std::runtime_error("evidence");
            Value typed; bool abstained=status=="abstained";
            if (type=="binary") { const auto& selected=v.at("value"); if (selected.is_null()!=abstained) throw std::runtime_error("binary abstention"); typed=BinaryValue{selected.is_null()?std::nullopt:std::make_optional(selected.get<bool>())}; }
            else if (type=="choice" || type=="ordinal") {
                const auto& selected=v.at("selected"); if (selected.is_null()!=abstained) throw std::runtime_error("selection abstention");
                auto id=selected.is_null()?std::nullopt:std::make_optional(selected.get<std::string>());
                if (id) { const auto& options=d.at("kind").at(type=="choice"?"options":"levels"); if (std::none_of(options.begin(),options.end(),[&](const Json& o){return o.at("id")==*id;})) throw std::runtime_error("unknown selection"); }
                if (type=="choice") typed=ChoiceValue{id};
                else typed=OrdinalValue{id,(!v.contains("level_value") || v.at("level_value").is_null())?std::nullopt:std::make_optional(v.at("level_value").get<double>())};
            } else throw std::runtime_error("unknown kind");
            out.results.push_back({r.at("id").get<std::string>(),status,std::move(typed),r.at("abstention_reasons").get<std::vector<std::string>>(),r.at("evidence"),r.at("usage"),r.value("reason_messages",Json::array())});
        } return out;
    } catch (const std::exception&) { throw Error("response does not match the v1 decision contract","invalid_response"); }
}
} // namespace
struct Engine::Impl {
    Process process;
    std::chrono::milliseconds timeout;
    std::uint64_t counter=0;
    bool closed=false;
    Json call(const std::string& op,const Json& body) {
        if (closed) throw Error("engine is closed","backend_closed");
        auto id="cpp-"+std::to_string(++counter);
        auto line=Json{{"id",id},{"op",op},{"body",body}}.dump()+"\n";
        if (line.size()>max_request) throw Error("stdio request exceeds limit","invalid_request");
        std::packaged_task<Json()> task([&]() {
            process.write_all(line);
            try { return Json::parse(process.read_line()); }
            catch (const Json::exception&) { throw Error("Rust returned invalid JSON","invalid_response"); }
        });
        auto future=task.get_future(); std::thread worker(std::move(task));
        if (future.wait_for(timeout)!=std::future_status::ready) {
            // Terminate on timeout; a later call must never consume this call's reply.
            closed=true; process.interrupt();
#ifdef _WIN32
            CancelSynchronousIo(worker.native_handle());
#endif
            worker.join(); process.close(); throw Error("Rust call timed out; engine closed","timeout",id);
        }
        worker.join(); Json envelope;
        try { envelope=future.get(); }
        catch (...) { closed=true; process.close(); throw; }
        if (!envelope.is_object() || envelope.value("id",Json())!=id) { closed=true; process.close(); throw Error("invalid stdio response ID","invalid_response",id); }
        if (envelope.contains("error")) {
            const auto& e=envelope["error"];
            try { throw Error(e.at("message").get<std::string>(),e.at("code").get<std::string>(),id,e.value("user_reason",std::string())); }
            catch (const Json::exception&) { throw Error("invalid error envelope","invalid_response",id); }
        }
        if (!envelope.contains("result")) throw Error("missing result envelope","invalid_response",id);
        return envelope["result"];
    }
};
Engine::Engine(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
Engine::~Engine()=default;
Engine::Engine(Engine&&) noexcept=default;
Engine& Engine::operator=(Engine&&) noexcept=default;
Engine Engine::load(const LoadOptions& options) {
    auto args=arguments(options); auto impl=std::make_unique<Impl>(); impl->timeout=options.startup_timeout;
    impl->process.start(args); Engine engine(std::move(impl)); engine.health(); engine.impl_->timeout=options.timeout; return engine;
}
Json Engine::call(const std::string& op,const Json& body) { if (!impl_) throw Error("engine is closed","backend_closed"); return impl_->call(op,body); }
void Engine::health() { auto value=call("health",nullptr); if (!value.is_object() || value.value("status",Json())!="ok") throw Error("invalid health response","invalid_response"); }
Json Engine::capabilities() {
    auto v=call("capabilities",nullptr);
    if (!v.is_object() || v.value("api_version",Json())!=1 || !v.contains("backend") || !v["backend"].is_object() || !v.contains("decision_types") || !v["decision_types"].is_array() || !v.contains("media") || !v.contains("limits")) throw Error("invalid capabilities response","invalid_response");
    return v;
}
Response Engine::decide(const Request& request) { return decide_json(to_json(request)); }
Response Engine::decide_json(const Json& request) { return response(call("decide",request),request); }
std::vector<Response> Engine::decide_batch(const std::vector<Request>& requests) { std::vector<Json> values; for (const auto& r:requests) values.push_back(to_json(r)); return decide_batch_json(values); }
std::vector<Response> Engine::decide_batch_json(const std::vector<Json>& requests) {
    if (!impl_ || impl_->closed) throw Error("engine is closed","backend_closed");
    if (requests.empty()) return {};
    auto v=call("decide_batch",{{"requests",requests}});
    if (!v.is_object() || v.value("api_version",Json())!=1 || v.value("execution",Json())!="native_parallel" || !v.contains("responses") || !v["responses"].is_array() || v["responses"].size()!=requests.size()) throw Error("invalid native batch response","invalid_response");
    std::vector<Response> out; for (std::size_t i=0;i<requests.size();++i) out.push_back(response(v["responses"][i],requests[i])); return out;
}
void Engine::close() noexcept { if (impl_) { impl_->closed=true; impl_->process.close(); } }
PreparedDecision Engine::prepare(std::vector<Decision> decisions,Json options) { return PreparedDecision(*this,std::move(decisions),std::move(options)); }
PreparedDecision::PreparedDecision(Engine& engine,std::vector<Decision> decisions,Json options):engine_(&engine),decisions_(std::move(decisions)),options_(std::move(options)) {}
Response PreparedDecision::decide(Json state) const { return engine_->decide({std::move(state),decisions_,options_}); }
std::vector<Response> PreparedDecision::decide_batch(const std::vector<Json>& states) const { std::vector<Request> requests; for (const auto& state:states) requests.push_back({state,decisions_,options_}); return engine_->decide_batch(requests); }
} // namespace l2s1
