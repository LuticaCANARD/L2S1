#pragma once
#include <nlohmann/json.hpp>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace l2s1 {
using Json = nlohmann::json;
struct Error : std::runtime_error {
    std::string code, request_id, user_reason;
    Error(std::string message, std::string code, std::string request_id = {}, std::string user_reason = {});
};
struct Option { std::string id, criterion; };
struct Level { std::string id, criterion; double value; };
struct Binary { std::string false_label, true_label; };
struct Choice { std::vector<Option> options; };
struct Ordinal { std::vector<Level> levels; };
using Kind = std::variant<Binary, Choice, Ordinal>;
struct Decision {
    std::string id, instruction;
    Kind kind;
    std::optional<std::vector<std::string>> media_ids;
};
Json to_json(const Decision& decision);
struct Policy { double min_top_probability = 0.8, min_candidate_mass = 0.05; };
struct Request {
    Json state;
    std::vector<Decision> decisions;
    // Advanced v1 fields (media, reasoning, target_error_rate, failure_reasons,
    // policy) are preserved verbatim and validated by Rust before inference.
    Json options = Json::object();
};
Json to_json(const Request& request);
struct BinaryValue { std::optional<bool> value; };
struct ChoiceValue { std::optional<std::string> selected; };
struct OrdinalValue { std::optional<std::string> selected; std::optional<double> level_value; };
using Value = std::variant<BinaryValue, ChoiceValue, OrdinalValue>;
struct Result {
    std::string id, status;
    Value value;
    std::vector<std::string> abstention_reasons;
    Json evidence, usage, reason_messages;
};
struct Response {
    std::string request_id;
    std::vector<Result> results;
    // The complete envelope retains backend identity, policy and future fields.
    Json raw;
};
struct LoadOptions {
    std::string model;
    std::string binary_path = "l2s1";
    std::string device = "cpu";
    std::optional<std::string> mmproj, lora;
    std::optional<unsigned> context, batch, ubatch, threads, parallel_width;
    std::optional<int> gpu_layers;
    // Omit to use automatic resident schema-prefix reuse when compatible.
    std::optional<std::string> execution_mode;
    // true requires reuse; false opts out unless an explicit mode is supplied.
    std::optional<bool> fixed_schema;
    std::optional<Policy> policy;
    std::chrono::milliseconds startup_timeout{120000}, timeout{180000};
    std::vector<std::string> extra_args;
};
class PreparedDecision;
// Owns one resident Rust child. Synchronous, move-only, RAII. Calls on the same
// instance must be serialized by the caller. Native inference batches are one RPC.
class Engine {
public:
    static Engine load(const LoadOptions& options);
    ~Engine();
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    void health();
    Json capabilities();
    Response decide(const Request& request);
    Response decide_json(const Json& request);
    std::vector<Response> decide_batch(const std::vector<Request>& requests);
    std::vector<Response> decide_batch_json(const std::vector<Json>& requests);
    PreparedDecision prepare(std::vector<Decision> decisions, Json options = Json::object());
    void close() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit Engine(std::unique_ptr<Impl> impl);
    Json call(const std::string& op, const Json& body);
};
// The engine must outlive its prepared plans and must not move while in use.
class PreparedDecision {
public:
    Response decide(Json state) const;
    std::vector<Response> decide_batch(const std::vector<Json>& states) const;
private:
    friend class Engine;
    PreparedDecision(Engine& engine, std::vector<Decision> decisions, Json options);
    Engine* engine_;
    std::vector<Decision> decisions_;
    Json options_;
};
} // namespace l2s1
