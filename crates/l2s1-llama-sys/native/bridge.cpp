#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <chrono>
#if defined(__linux__)
#include <link.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <climits>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
struct log_state {
    std::mutex mutex;
    std::string last_error;
    ggml_log_level last_level = GGML_LOG_LEVEL_NONE;
    ggml_log_level threshold = GGML_LOG_LEVEL_WARN;
};

log_state native_log;
// llama.cpp uses one process-wide logger, so model load diagnostics are captured
// one load at a time, including messages emitted by its worker threads.
std::mutex model_load_mutex;

void llama_log_callback(ggml_log_level level, const char * message, void *) noexcept {
    if (!message) return;
    try {
        std::lock_guard<std::mutex> lock(native_log.mutex);
        if (level == GGML_LOG_LEVEL_CONT) {
            if (native_log.last_level == GGML_LOG_LEVEL_ERROR) native_log.last_error += message;
            if (native_log.last_level >= native_log.threshold)
                std::fputs(message, stderr);
            return;
        }
        native_log.last_level = level;
        if (level == GGML_LOG_LEVEL_ERROR) {
            const std::string text(message);
            // llama.cpp emits this after the useful error, such as a tensor
            // count mismatch. Do not replace the actual cause with it.
            if (native_log.last_error.empty() || text.find("failed to load model") == std::string::npos)
                native_log.last_error = text;
        }
        if (level >= native_log.threshold && level != GGML_LOG_LEVEL_NONE)
            std::fputs(message, stderr);
    } catch (...) {
        // A log callback must not throw through llama.cpp's C API.
    }
}

void clear_load_error() {
    std::lock_guard<std::mutex> lock(native_log.mutex);
    native_log.last_error.clear();
    native_log.last_level = GGML_LOG_LEVEL_NONE;
}

std::string load_error(const char * fallback) {
    std::lock_guard<std::mutex> lock(native_log.mutex);
    auto message = native_log.last_error;
    const auto last = message.find_last_not_of("\r\n \t");
    if (last == std::string::npos) return fallback;
    message.erase(last + 1);
    return message;
}

bool log_info_enabled() {
    return native_log.threshold <= GGML_LOG_LEVEL_INFO;
}
} // namespace

struct vision_batch_metrics {
    size_t projector_encode_calls = 0;
    size_t projector_batch_max = 0;
    size_t decoder_calls = 0;
    size_t decoder_batch_max_sequences = 0;
    size_t projector_reused_chunks = 0;
    size_t kv_clear_calls = 0;
    size_t kv_clear_skipped = 0;
};

struct engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_adapter_lora * adapter = nullptr; // owned by model
    mtmd_context * vision = nullptr;
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    bool features_enabled = false;
    int32_t last_feature_row = -1;
    uint32_t batch_size = 0;
    uint32_t context_size = 0;
    uint32_t sequence_capacity = 1;
    uint32_t allocated_context_size = 0;
    bool parallel_context_dynamic = false;
    bool parallel_shared_kv = true;
    llama_context_params context_params = {};
    std::string description;
    std::string architecture;
    std::string runtime_libraries;
    std::vector<int32_t> cached_tokens;
    vision_batch_metrics vision_metrics;
    bool memory_dirty = false;
    // Benchmark diagnostic: reproduce unconditional physical clearing while
    // retaining the same bridge, wrapper calls and numerical execution path.
    bool force_kv_clear = false;
    bool vision_projector_reuse = false;
    // Own pattern strings for at least the model lifetime; no static leaks.
    std::vector<std::string> cpu_moe_patterns;
    std::vector<llama_model_tensor_buft_override> placement_overrides;
    ~engine() {
        if (vision) mtmd_free(vision);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

static void report(char * out, size_t cap, const char * message) {
    if (cap) std::snprintf(out, cap, "%s", message);
}

extern "C" engine * sd_open_loading(const char * path, uint32_t context, uint32_t batch,
        uint32_t ubatch, int32_t flash_attention, int32_t threads, int32_t device_kind,
        int32_t gpu_layers, int32_t cpu_moe_layers, int32_t model_load_mode,
        char * error, size_t error_cap) noexcept {
    try {
        if (ubatch == 0 || ubatch > batch || flash_attention < -1 || flash_attention > 1)
            throw std::runtime_error("invalid microbatch or FlashAttention option");
        if (device_kind < 0 || device_kind > 2 || gpu_layers < -1 || cpu_moe_layers < 0 ||
                static_cast<size_t>(cpu_moe_layers) >= llama_max_tensor_buft_overrides() ||
                (device_kind == 0 && (gpu_layers != 0 || cpu_moe_layers != 0)))
            throw std::runtime_error("invalid CPU/GPU placement options");
        if (model_load_mode != -1 && model_load_mode != 0)
            throw std::runtime_error("invalid model loading mode");
        static std::once_flag init;
        std::call_once(init, [] {
            if (const char * setting = std::getenv("L2S1_LOG")) {
                const std::string level(setting);
                if (level == "off" || level == "none") native_log.threshold = static_cast<ggml_log_level>(6);
                else if (level == "error") native_log.threshold = GGML_LOG_LEVEL_ERROR;
                else if (level == "info") native_log.threshold = GGML_LOG_LEVEL_INFO;
                else if (level == "debug") native_log.threshold = GGML_LOG_LEVEL_DEBUG;
            }
            llama_log_set(llama_log_callback, nullptr);
            // Vision/helper/CLIP logging has a separate upstream callback.
            mtmd_helper_log_set(llama_log_callback, nullptr);
            ggml_backend_load_all();
            llama_backend_init();
        });
        auto e = std::make_unique<engine>();
        if (const char * setting = std::getenv("L2S1_FORCE_KV_CLEAR"))
            e->force_kv_clear = std::strcmp(setting, "1") == 0;
        auto mp = llama_model_default_params();
        if (device_kind != 0) {
            const char * requested = device_kind == 1 ? "CUDA" : "MTL";
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto dev = ggml_backend_dev_get(i);
                auto reg = ggml_backend_dev_backend_reg(dev);
                if (std::strcmp(ggml_backend_reg_name(reg), requested) == 0 &&
                    ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    e->devices[0] = dev;
                    break;
                }
            }
            if (!e->devices[0]) throw std::runtime_error(std::string(requested) + " device unavailable; select CPU explicitly to use CPU");
        }
        mp.devices = e->devices;
        mp.n_gpu_layers = gpu_layers;
        // Use llama.cpp's bounded read/upload path. Do not change weight placement,
        // quantization, lazy-tensor policy, or inference context parameters.
        if (model_load_mode == 0) mp.load_mode = LLAMA_LOAD_MODE_NONE;
        if (log_info_enabled()) std::fprintf(stderr, "l2s1 model loading: %s\n", model_load_mode == 0 ? "read" : "auto");
        if (cpu_moe_layers > 0) {
            auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!cpu) throw std::runtime_error("CPU backend unavailable for MoE split");
            auto cpu_buffer = ggml_backend_dev_buffer_type(cpu);
            e->cpu_moe_patterns.reserve(cpu_moe_layers);
            // Same expert tensor family as llama.cpp's --n-cpu-moe option.
            for (int32_t layer = 0; layer < cpu_moe_layers; ++layer)
                e->cpu_moe_patterns.push_back("^blk\\." + std::to_string(layer) +
                    "\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.");
            for (const auto & pattern : e->cpu_moe_patterns)
                e->placement_overrides.push_back({pattern.c_str(), cpu_buffer});
            e->placement_overrides.push_back({nullptr, nullptr});
            mp.tensor_buft_overrides = e->placement_overrides.data();
        }
        if (log_info_enabled()) std::fprintf(stderr, "l2s1 placement: gpu_layers=%d cpu_moe_layers=%d\n", gpu_layers, cpu_moe_layers);
        {
            std::lock_guard<std::mutex> load_lock(model_load_mutex);
            clear_load_error();
            e->model = llama_model_load_from_file(path, mp);
            if (!e->model) {
                auto cause = load_error("llama.cpp did not report a cause");
                if (cause.find("wrong number of tensors") != std::string::npos)
                    cause += "; if this is a bundled vision GGUF, model and projector must be separate GGUF files";
                throw std::runtime_error("model load failed: " + cause);
            }
        }
        char arch[64] = {};
        llama_model_meta_val_str(e->model, "general.architecture", arch, sizeof(arch));
        e->architecture = arch;
        if (cpu_moe_layers > 0) {
            char experts[32] = {};
            const std::string key = e->architecture + ".expert_count";
            llama_model_meta_val_str(e->model, key.c_str(), experts, sizeof(experts));
            if (std::strtol(experts, nullptr, 10) <= 0 || cpu_moe_layers > llama_model_n_layer(e->model))
                throw std::runtime_error("CPU MoE split requires an MoE model and a layer count within the model");
        }
        if (!llama_model_has_decoder(e->model) || llama_model_has_encoder(e->model))
            throw std::runtime_error("decision scoring requires a decoder-only language model");
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = batch;
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.embeddings = false;
        cp.offload_kqv = device_kind != 0;
        cp.op_offload = device_kind != 0;
        cp.flash_attn_type = static_cast<llama_flash_attn_type>(flash_attention);
        e->context_params = cp;
        e->context_size = context;
        e->ctx = llama_init_from_model(e->model, cp);
        if (!e->ctx) throw std::runtime_error("context allocation failed");
        e->allocated_context_size = context;
        e->batch_size = llama_n_batch(e->ctx);
        char desc[512] = {};
        llama_model_desc(e->model, desc, sizeof(desc));
        e->description = desc;
#if defined(__linux__)
        std::vector<std::string> libraries;
        dl_iterate_phdr([](dl_phdr_info * info, size_t, void * data) {
            const std::string path = info->dlpi_name;
            const auto name = path.substr(path.find_last_of('/') + 1);
            if (name.rfind("libllama.so", 0) == 0 || name.rfind("libmtmd.so", 0) == 0 || name.rfind("libggml", 0) == 0)
                static_cast<std::vector<std::string> *>(data)->push_back(path);
            return 0;
        }, &libraries);
        std::sort(libraries.begin(), libraries.end());
        for (const auto & library : libraries) e->runtime_libraries += library + "\n";
#elif defined(__APPLE__)
        std::vector<std::string> libraries;
        for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
            const char * image = _dyld_get_image_name(i);
            if (!image) continue;
            const std::string path = image;
            const auto name = path.substr(path.find_last_of('/') + 1);
            if ((name.rfind("libllama", 0) == 0 || name.rfind("libmtmd", 0) == 0 ||
                 name.rfind("libggml", 0) == 0) && name.find(".dylib") != std::string::npos)
                libraries.push_back(path);
        }
        std::sort(libraries.begin(), libraries.end());
        for (const auto & library : libraries) e->runtime_libraries += library + "\n";
#endif
        return e.release();
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    return nullptr;
}

// Preserve the placement ABI for native reference callers.
extern "C" engine * sd_open_placement(const char * path, uint32_t context, uint32_t batch,
        uint32_t ubatch, int32_t flash_attention, int32_t threads, bool cuda,
        int32_t gpu_layers, int32_t cpu_moe_layers,
        char * error, size_t error_cap) noexcept {
    return sd_open_loading(path, context, batch, ubatch, flash_attention, threads,
        cuda ? 1 : 0, gpu_layers, cpu_moe_layers, -1, error, error_cap);
}

// Keep the original native entry point for existing callers and reference tests.
extern "C" engine * sd_open(const char * path, uint32_t context, uint32_t batch,
        uint32_t ubatch, int32_t flash_attention, int32_t threads, bool cuda,
        char * error, size_t error_cap) noexcept {
    return sd_open_placement(path, context, batch, ubatch, flash_attention, threads,
        cuda, cuda ? -1 : 0, 0, error, error_cap);
}

extern "C" void sd_clear(engine * e) noexcept;
extern "C" void sd_close(engine * e) noexcept { delete e; }
extern "C" const char * sd_vision_marker() noexcept { return mtmd_default_marker(); }
extern "C" bool sd_load_vision_projector(engine * e, const char * path,
        char * error, size_t error_cap) noexcept {
    try {
        if (!e || !e->model || !path || !*path) throw std::runtime_error("invalid vision projector path");
        if (e->vision) throw std::runtime_error("vision projector already loaded");
        auto params = mtmd_context_params_default();
        params.print_timings = log_info_enabled();
        params.use_gpu = e->devices[0] != nullptr;
        params.device = e->devices[0];
        params.n_threads = e->context_params.n_threads;
        params.flash_attn_type = e->context_params.flash_attn_type;
        std::lock_guard<std::mutex> load_lock(model_load_mutex);
        clear_load_error();
        auto * vision = mtmd_init_from_file(path, e->model, params);
        if (!vision) throw std::runtime_error("vision projector load failed: " + load_error("llama.cpp did not report a cause"));
        if (!mtmd_support_vision(vision)) {
            mtmd_free(vision);
            throw std::runtime_error("projector does not support image input");
        }
        e->vision = vision;
        sd_clear(e);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    return false;
}
extern "C" const char * sd_runtime_libraries(const engine * e) noexcept { return e->runtime_libraries.c_str(); }
extern "C" const char * sd_description(const engine * e) noexcept { return e->description.c_str(); }
extern "C" const char * sd_architecture(const engine * e) noexcept { return e->architecture.c_str(); }
extern "C" const char * sd_chat_template(const engine * e) noexcept {
    return llama_model_chat_template(e->model, nullptr);
}
extern "C" int32_t sd_required_bos(const engine * e) noexcept {
    const auto * vocab = llama_model_get_vocab(e->model);
    return llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : -1;
}
extern "C" const char * sd_bos_text(const engine * e) noexcept {
    const auto * vocab = llama_model_get_vocab(e->model);
    const auto token = llama_vocab_bos(vocab);
    return token < 0 ? "" : llama_vocab_get_text(vocab, token);
}
extern "C" const char * sd_eos_text(const engine * e) noexcept {
    const auto * vocab = llama_model_get_vocab(e->model);
    const auto token = llama_vocab_eos(vocab);
    return token < 0 ? "" : llama_vocab_get_text(vocab, token);
}
extern "C" const char * sd_device(const engine * e) noexcept {
    return e->devices[0] ? ggml_backend_dev_description(e->devices[0]) : "CPU";
}
extern "C" int32_t sd_vocab_size(const engine * e) noexcept {
    return llama_vocab_n_tokens(llama_model_get_vocab(e->model));
}
extern "C" int32_t sd_tokenize(const engine * e, const char * text, int32_t length,
        bool special, int32_t * out, int32_t capacity) noexcept {
    try {
        return llama_tokenize(llama_model_get_vocab(e->model), text, length, out, capacity, false, special);
    } catch (...) { return INT32_MIN; }
}

extern "C" void sd_clear(engine * e) noexcept {
    if (!e) return;
    e->cached_tokens.clear();
    e->last_feature_row = -1;
    // Every operation that can write KV marks it dirty before entering llama.cpp.
    // Failed decodes/restores may have written partial state and still need zeroing.
    if (!e->memory_dirty && !e->force_kv_clear) {
        ++e->vision_metrics.kv_clear_skipped;
        return;
    }
    if (e->ctx) {
        if (auto memory = llama_get_memory(e->ctx)) {
            llama_memory_clear(memory, true);
            ++e->vision_metrics.kv_clear_calls;
        }
    }
    e->memory_dirty = false;
}

extern "C" void sd_set_vision_projector_reuse(engine * e, bool enabled) noexcept {
    if (e) e->vision_projector_reuse = enabled;
}

// Unmasked NextN exposes Gemma4's post-norm LM-head input without forcing
// every token through the vocabulary projection or pruning the last-layer graph.
// Only the final row is copied into Rust; this preserves the base compute shapes.
extern "C" bool sd_set_features(engine * e, bool enabled) noexcept {
    if (!e->ctx || (enabled && (e->architecture != "gemma4" || llama_pooling_type(e->ctx) != LLAMA_POOLING_TYPE_NONE))) return false;
    if (e->features_enabled != enabled) {
        sd_clear(e);
        e->features_enabled = enabled;
        llama_set_embeddings_nextn(e->ctx, enabled, false);
    }
    return true;
}
extern "C" int32_t sd_feature_size(engine * e) noexcept {
    return llama_model_n_embd_out(e->model);
}
extern "C" bool sd_copy_features(engine * e, float * output, size_t size) noexcept {
    if (!e->ctx || !e->features_enabled || e->last_feature_row < 0 || size != size_t(sd_feature_size(e))) return false;
    const float * features = llama_get_embeddings_nextn_ith(e->ctx, e->last_feature_row);
    if (!features) return false;
    std::copy_n(features, size, output);
    return true;
}

extern "C" bool sd_load_lora(engine * e, const char * path, char * error, size_t error_cap) noexcept {
    try {
        if (e->adapter) throw std::runtime_error("one LoRA adapter is supported per backend; load a new backend to change it");
        if (!e->ctx) throw std::runtime_error("context unavailable; create a new backend before loading LoRA");
        auto * adapter = llama_adapter_lora_init(e->model, path);
        if (!adapter) throw std::runtime_error("LoRA adapter load failed; see llama.cpp stderr");
        sd_clear(e);
        float scale = 1.0f;
        if (llama_set_adapters_lora(e->ctx, &adapter, 1, &scale) != 0) {
            llama_adapter_lora_free(adapter);
            throw std::runtime_error("LoRA adapter activation failed");
        }
        e->adapter = adapter;
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    return false;
}

// One model allocation; resize only its context when changing execution modes.
static void ensure_sequences(engine * e, uint32_t capacity,
        uint32_t requested_context = 0, bool dynamic_context = false,
        bool shared_kv = true) {
    if (capacity < 1 || capacity > 32 || e->context_size > uint32_t(INT_MAX) / capacity)
        throw std::runtime_error("parallel width requires 1..32 and context * width <= INT_MAX");
    const uint32_t maximum_context = e->context_size * capacity;
    if (requested_context == 0) requested_context = maximum_context;
    if (requested_context > maximum_context)
        throw std::runtime_error("parallel context reservation is outside the configured limits");
    if (e->ctx && e->sequence_capacity == capacity &&
            e->parallel_context_dynamic == dynamic_context &&
            e->parallel_shared_kv == shared_kv &&
            (dynamic_context ? e->allocated_context_size >= requested_context :
                e->allocated_context_size == requested_context)) return;
    sd_clear(e);
    if (e->ctx) llama_free(e->ctx);
    e->ctx = nullptr;
    auto cp = e->context_params;
    cp.n_seq_max = capacity;
    cp.n_ctx = requested_context;
    // Text prefix sharing copies KV cells across sequence IDs and needs one
    // unified stream. Independent images use separate streams so attention
    // batches avoid scanning the other images' masked KV entries.
    cp.kv_unified = capacity > 1 ? shared_kv : e->context_params.kv_unified;
    e->ctx = llama_init_from_model(e->model, cp);
    if (!e->ctx) throw std::runtime_error("parallel context allocation failed; reduce parallel width");
    if (e->adapter) {
        float scale = 1.0f;
        if (llama_set_adapters_lora(e->ctx, &e->adapter, 1, &scale) != 0) {
            llama_free(e->ctx);
            e->ctx = nullptr;
            throw std::runtime_error("LoRA activation failed after context resize");
        }
    }
    llama_set_embeddings_nextn(e->ctx, e->features_enabled, false);
    e->sequence_capacity = capacity;
    e->allocated_context_size = requested_context;
    e->parallel_context_dynamic = dynamic_context;
    e->parallel_shared_kv = shared_kv;
}

// Both output paths use identical decoding and prefix handling. The returned
// view belongs to llama.cpp and is consumed before another native call.
static const float * forward_logits(engine * e, const int32_t * tokens, int32_t count,
        bool reuse, int32_t * reused) {
    ensure_sequences(e, 1);
    if (count <= 0 || uint32_t(count) > llama_n_ctx(e->ctx))
        throw std::runtime_error("input exceeds context or is empty; truncation is disabled");
    auto memory = llama_get_memory(e->ctx);
    if (!memory) throw std::runtime_error("decoder memory unavailable");
    // Recurrent/hybrid state cannot generally be rolled back to an arbitrary
    // prefix. Use fresh evaluation there, and whenever partial removal fails.
    int32_t common = 0;
    if (reuse && !llama_model_is_recurrent(e->model) && !llama_model_is_hybrid(e->model)) {
        // Always evaluate at least the final token to obtain current logits.
        const auto limit = std::min(e->cached_tokens.size(), size_t(count - 1));
        while (size_t(common) < limit && e->cached_tokens[common] == tokens[common]) ++common;
        // Reuse only complete original prefill batches. An arbitrary split
        // changes CUDA kernel/batch shapes and can flip threshold decisions.
        common -= common % int32_t(e->batch_size);
    }
    if (common == 0 || !llama_memory_seq_rm(memory, 0, common, -1)) {
        sd_clear(e);
        common = 0;
    }
    e->cached_tokens.clear();
    struct batch_guard {
        llama_batch value;
        ~batch_guard() { llama_batch_free(value); }
    } b { llama_batch_init(int32_t(e->batch_size), 0, 1) };
    if (!b.value.token || !b.value.pos || !b.value.n_seq_id || !b.value.seq_id || !b.value.logits)
        throw std::runtime_error("batch allocation failed");
    for (int32_t start = common; start < count;) {
        b.value.n_tokens = std::min(int32_t(e->batch_size), count - start);
        for (int32_t j = 0; j < b.value.n_tokens; ++j) {
            b.value.token[j] = tokens[start + j];
            b.value.pos[j] = start + j;
            b.value.n_seq_id[j] = 1;
            b.value.seq_id[j][0] = 0;
            b.value.logits[j] = start + j == count - 1;
        }
        e->memory_dirty = true;
        if (llama_decode(e->ctx, b.value) != 0) throw std::runtime_error("llama_decode failed");
        e->last_feature_row = b.value.n_tokens - 1;
        start += b.value.n_tokens;
    }
    const float * output = llama_get_logits_ith(e->ctx, -1);
    if (!output) throw std::runtime_error("missing final logits");
    if (reuse) e->cached_tokens.assign(tokens, tokens + count);
    *reused = common;
    return output;
}

extern "C" bool sd_forward(engine * e, const int32_t * tokens, int32_t count,
        bool reuse, int32_t * reused, float * logits, size_t logits_count,
        char * error, size_t error_cap) noexcept {
    *reused = 0;
    try {
        if (logits_count != size_t(sd_vocab_size(e))) throw std::runtime_error("wrong logits buffer size");
        const float * output = forward_logits(e, tokens, count, reuse, reused);
        std::copy_n(output, logits_count, logits);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    sd_clear(e);
    return false;
}

// Match the text compact path's full-vocabulary f64 normalization exactly.
// This reduces bridge copies; llama.cpp still computes the full vocabulary.
static void validate_candidate_bank(const int32_t * ids, size_t count, int32_t vocab) {
    if (!ids || count < 2 || count > 26 || vocab <= 0)
        throw std::runtime_error("invalid vision compact candidate bank");
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= vocab) throw std::runtime_error("candidate token outside vocabulary");
        for (size_t j = 0; j < i; ++j)
            if (ids[i] == ids[j]) throw std::runtime_error("duplicate candidate token");
    }
}
static void copy_compact_logits(const float * output, int32_t vocab,
        const int32_t * ids, size_t count, float * logits, double * log_normalizer) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < vocab; ++i) {
        const double value = output[i];
        if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
            throw std::runtime_error("invalid vocabulary logit");
        maximum = std::max(maximum, value);
    }
    if (!std::isfinite(maximum)) throw std::runtime_error("no finite vocabulary logit");
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(output[ids[i]])) throw std::runtime_error("nonfinite candidate logit");
    double sum = 0;
    for (int32_t i = 0; i < vocab; ++i) sum += std::exp(double(output[i]) - maximum);
    const double normalizer = maximum + std::log(sum);
    if (!std::isfinite(normalizer)) throw std::runtime_error("invalid vocabulary normalizer");
    for (size_t i = 0; i < count; ++i) logits[i] = output[ids[i]];
    *log_normalizer = normalizer;
}

// Vision input is evaluated from fresh request-local state. The five prompt
// parts preserve the trust boundary between template control tokens and data.
static bool forward_vision(engine * e,
        const char * prefix, size_t prefix_len,
        const char * data_before, size_t before_len,
        const uint8_t * image, size_t image_len,
        const char * data_after, size_t after_len,
        const char * suffix, size_t suffix_len,
        const int32_t * continuation, size_t continuation_count,
        float * logits, size_t logits_count, size_t * input_tokens,
        const int32_t * candidate_ids, size_t candidate_count, double * log_normalizer,
        char * error, size_t error_cap) noexcept {
    try {
        if (!e || !e->vision || !prefix || !data_before || !image || !image_len ||
                !data_after || !suffix || (continuation_count && !continuation) ||
                !logits || !input_tokens ||
                logits_count != (candidate_ids ? candidate_count : size_t(sd_vocab_size(e))))
            throw std::runtime_error("invalid vision inference arguments");
        e->vision_metrics = {};
        if (candidate_ids) {
            if (!log_normalizer) throw std::runtime_error("missing vision compact normalizer");
            validate_candidate_bank(candidate_ids, candidate_count, sd_vocab_size(e));
        }
        ensure_sequences(e, 1);
        sd_clear(e);
        auto init_opt = mtmd_helper_init_opt_default();
        auto wrapped = mtmd_helper_bitmap_init_from_buf(e->vision, image, image_len, false, init_opt);
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(wrapped.bitmap, mtmd_bitmap_free);
        if (wrapped.video_ctx) {
            mtmd_helper_video_free(wrapped.video_ctx);
            throw std::runtime_error("video input is unsupported; provide one still image");
        }
        if (!bitmap || mtmd_bitmap_is_audio(bitmap.get()))
            throw std::runtime_error("image must be a supported still-image format");
        mtmd_input_text texts[] = {
            {prefix, prefix_len, false, true},
            {data_before, before_len, false, false},
            {data_after, after_len, false, false},
            {suffix, suffix_len, false, true},
        };
        mtmd_input_part parts[] = {
            {&texts[0], nullptr}, {&texts[1], nullptr}, {nullptr, bitmap.get()},
            {&texts[2], nullptr}, {&texts[3], nullptr},
        };
        const mtmd_input_part * part_ptrs[] = {&parts[0], &parts[1], &parts[2], &parts[3], &parts[4]};
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
            mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (!chunks || mtmd_tokenize_from_parts(e->vision, chunks.get(), part_ptrs, 5, true) != 0)
            throw std::runtime_error("vision prompt tokenization failed");
        const size_t tokens = mtmd_helper_get_n_tokens(chunks.get());
        const auto positions = mtmd_helper_get_n_pos(chunks.get());
        if (!tokens || tokens > e->context_size || positions <= 0 ||
                size_t(positions) > e->context_size ||
                continuation_count > e->context_size - size_t(positions))
            throw std::runtime_error("vision input exceeds context; truncation is disabled");
        llama_pos end = 0;
        e->memory_dirty = true;
        if (mtmd_helper_eval_chunks(e->vision, e->ctx, chunks.get(), 0, 0,
                    int32_t(e->batch_size), true, &end) != 0 || end != positions)
            throw std::runtime_error("vision decode failed");
        if (continuation_count) {
            struct batch_guard {
                llama_batch value;
                ~batch_guard() { llama_batch_free(value); }
            } b { llama_batch_init(int32_t(e->batch_size), 0, 1) };
            if (!b.value.token || !b.value.pos || !b.value.n_seq_id ||
                    !b.value.seq_id || !b.value.logits)
                throw std::runtime_error("vision continuation batch allocation failed");
            for (size_t start = 0; start < continuation_count;) {
                b.value.n_tokens = int32_t(std::min(size_t(e->batch_size), continuation_count - start));
                for (int32_t j = 0; j < b.value.n_tokens; ++j) {
                    const size_t index = start + size_t(j);
                    if (continuation[index] < 0 || continuation[index] >= sd_vocab_size(e))
                        throw std::runtime_error("vision continuation token outside vocabulary");
                    b.value.token[j] = continuation[index];
                    b.value.pos[j] = end + llama_pos(index);
                    b.value.n_seq_id[j] = 1;
                    b.value.seq_id[j][0] = 0;
                    b.value.logits[j] = index + 1 == continuation_count;
                }
                e->memory_dirty = true;
                if (llama_decode(e->ctx, b.value) != 0)
                    throw std::runtime_error("vision continuation decode failed");
                start += size_t(b.value.n_tokens);
            }
        }
        const float * output = llama_get_logits_ith(e->ctx, -1);
        if (!output) throw std::runtime_error("missing vision final logits");
        if (candidate_ids) copy_compact_logits(output, sd_vocab_size(e), candidate_ids, candidate_count, logits, log_normalizer);
        else std::copy_n(output, logits_count, logits);
        *input_tokens = tokens;
        sd_clear(e);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    if (e) sd_clear(e);
    return false;
}

extern "C" bool sd_forward_vision(engine * e,
        const char * prefix, size_t prefix_len, const char * data_before, size_t before_len,
        const uint8_t * image, size_t image_len, const char * data_after, size_t after_len,
        const char * suffix, size_t suffix_len, const int32_t * continuation, size_t continuation_count,
        float * logits, size_t logits_count, size_t * input_tokens, char * error, size_t error_cap) noexcept {
    return forward_vision(e, prefix, prefix_len, data_before, before_len, image, image_len,
        data_after, after_len, suffix, suffix_len, continuation, continuation_count,
        logits, logits_count, input_tokens, nullptr, 0, nullptr, error, error_cap);
}
extern "C" bool sd_forward_vision_compact(engine * e,
        const char * prefix, size_t prefix_len, const char * data_before, size_t before_len,
        const uint8_t * image, size_t image_len, const char * data_after, size_t after_len,
        const char * suffix, size_t suffix_len, const int32_t * continuation, size_t continuation_count,
        const int32_t * candidate_ids, size_t candidate_count,
        float * logits, size_t logits_count, double * log_normalizer,
        size_t * input_tokens, char * error, size_t error_cap) noexcept {
    if (!candidate_ids || !log_normalizer) { if (e) sd_clear(e); report(error, error_cap, "null vision compact argument"); return false; }
    return forward_vision(e, prefix, prefix_len, data_before, before_len, image, image_len,
        data_after, after_len, suffix, suffix_len, continuation, continuation_count,
        logits, logits_count, input_tokens, candidate_ids, candidate_count, log_normalizer, error, error_cap);
}

// Independent media inputs retain their own sequence IDs and trust-separated
// prompt parts. Compatible image chunks share a projector encode; decoder
// batches mix rows from several sequences without sharing image KV entries.
struct vision_input {
    const char * prefix; size_t prefix_len;
    const char * data_before; size_t before_len;
    const uint8_t * image; size_t image_len;
    const char * data_after; size_t after_len;
    const char * suffix; size_t suffix_len;
};

extern "C" bool sd_vision_batch_metrics(const engine * e, vision_batch_metrics * out) noexcept {
    if (!e || !out) return false;
    *out = e->vision_metrics;
    return true;
}

static bool forward_vision_parallel(engine * e, const vision_input * inputs,
        int32_t sequences, uint32_t capacity, bool dynamic_context,
        float * logits, size_t logits_count, size_t * input_tokens,
        const int32_t * const * candidate_ids, const size_t * candidate_counts, double * log_normalizers,
        char * error, size_t error_cap) noexcept {
    try {
        if (!e || !e->vision || !inputs || !logits || !input_tokens ||
                sequences < 1 || sequences > int32_t(capacity) || capacity > 32)
            throw std::runtime_error("invalid parallel vision arguments");
        e->vision_metrics = {};
        if (llama_model_is_recurrent(e->model) || llama_model_is_hybrid(e->model))
            throw std::runtime_error("parallel vision is unsupported for recurrent/hybrid models");
        const size_t vocab = size_t(sd_vocab_size(e));
        std::vector<size_t> candidate_offsets(size_t(sequences) + 1, 0);
        if (candidate_ids) {
            if (!candidate_counts || !log_normalizers) throw std::runtime_error("missing parallel compact arguments");
            for (int32_t i = 0; i < sequences; ++i) {
                validate_candidate_bank(candidate_ids[i], candidate_counts[i], int32_t(vocab));
                candidate_offsets[size_t(i) + 1] = candidate_offsets[size_t(i)] + candidate_counts[i];
            }
        }
        if (logits_count != (candidate_ids ? candidate_offsets.back() : vocab * size_t(sequences)))
            throw std::runtime_error("wrong parallel vision logits buffer size");
        using chunks_owner = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
        struct segment {
            const mtmd_input_chunk * chunk;
            std::shared_ptr<std::vector<float>> embeddings;
            size_t offset = 0;
            int32_t source = 0;
            size_t image_ordinal = 0;
        };
        struct sequence {
            chunks_owner chunks{nullptr, mtmd_input_chunks_free};
            std::vector<segment> segments;
            size_t current = 0;
            llama_pos position = 0;
        };
        std::vector<sequence> seqs(static_cast<size_t>(sequences));
        std::vector<segment *> media;
        size_t maximum_tokens = 0;
        const size_t n_embd = size_t(llama_model_n_embd_inp(e->model));
        const bool mrope = mtmd_decode_use_mrope(e->vision);
        for (int32_t s = 0; s < sequences; ++s) {
            const auto & in = inputs[s];
            if (!in.prefix || !in.data_before || !in.image || !in.image_len || !in.data_after || !in.suffix)
                throw std::runtime_error("invalid parallel vision input");
            auto opt = mtmd_helper_init_opt_default();
            auto wrapped = mtmd_helper_bitmap_init_from_buf(e->vision, in.image, in.image_len, false, opt);
            std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(wrapped.bitmap, mtmd_bitmap_free);
            if (wrapped.video_ctx) {
                mtmd_helper_video_free(wrapped.video_ctx);
                throw std::runtime_error("video input is unsupported; provide still images");
            }
            if (!bitmap || mtmd_bitmap_is_audio(bitmap.get()))
                throw std::runtime_error("image must be a supported still-image format");
            mtmd_input_text texts[] = {
                {in.prefix, in.prefix_len, false, true},
                {in.data_before, in.before_len, false, false},
                {in.data_after, in.after_len, false, false},
                {in.suffix, in.suffix_len, false, true},
            };
            mtmd_input_part parts[] = {
                {&texts[0], nullptr}, {&texts[1], nullptr}, {nullptr, bitmap.get()},
                {&texts[2], nullptr}, {&texts[3], nullptr},
            };
            const mtmd_input_part * ptrs[] = {&parts[0], &parts[1], &parts[2], &parts[3], &parts[4]};
            auto & seq = seqs[size_t(s)];
            seq.chunks.reset(mtmd_input_chunks_init());
            if (!seq.chunks || mtmd_tokenize_from_parts(e->vision, seq.chunks.get(), ptrs, 5, true) != 0)
                throw std::runtime_error("parallel vision prompt tokenization failed");
            const size_t tokens = mtmd_helper_get_n_tokens(seq.chunks.get());
            const auto positions = mtmd_helper_get_n_pos(seq.chunks.get());
            if (!tokens || tokens > e->context_size || positions <= 0 || uint32_t(positions) > e->context_size)
                throw std::runtime_error("parallel vision input exceeds per-question context; truncation is disabled");
            input_tokens[s] = tokens;
            maximum_tokens = std::max(maximum_tokens, tokens);
            const size_t size = mtmd_input_chunks_size(seq.chunks.get());
            seq.segments.reserve(size);
            size_t image_ordinal = 0;
            for (size_t c = 0; c < size; ++c) {
                auto * chunk = mtmd_input_chunks_get(seq.chunks.get(), c);
                if (!mtmd_input_chunk_get_n_tokens(chunk)) continue;
                const auto type = mtmd_input_chunk_get_type(chunk);
                if (type != MTMD_INPUT_CHUNK_TYPE_TEXT && type != MTMD_INPUT_CHUNK_TYPE_IMAGE)
                    throw std::runtime_error("parallel vision only supports text and image chunks");
                seq.segments.push_back({chunk, nullptr, 0, s,
                    type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? image_ordinal++ : 0});
            }
            if (seq.segments.empty() || mtmd_input_chunk_get_type(seq.segments.back().chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT)
                throw std::runtime_error("parallel vision requires a final text suffix for decision logits");
        }
        // Segment addresses are stable after every sequence has finished building.
        for (auto & seq : seqs) for (auto & seg : seq.segments)
            if (mtmd_input_chunk_get_type(seg.chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) media.push_back(&seg);
        // Reuse projector work only within this call. Exact encoded bytes plus
        // the same preprocessed image ordinal and decoder shape prevent mixing
        // resolution tiles or M-RoPE positions. Each sequence keeps its own chunk,
        // positions and independent KV stream; only immutable embeddings alias.
        std::vector<std::pair<segment *, segment *>> duplicates;
        if (e->vision_projector_reuse) {
            std::vector<segment *> unique;
            auto equal_image = [&](const segment & a, const segment & b) {
                const auto & ai = inputs[a.source];
                const auto & bi = inputs[b.source];
                if (a.image_ordinal != b.image_ordinal || ai.image_len != bi.image_len ||
                        std::memcmp(ai.image, bi.image, ai.image_len) != 0 ||
                        mtmd_input_chunk_get_n_tokens(a.chunk) != mtmd_input_chunk_get_n_tokens(b.chunk) ||
                        mtmd_input_chunk_get_n_pos(a.chunk) != mtmd_input_chunk_get_n_pos(b.chunk) ||
                        mtmd_decode_use_non_causal(e->vision, a.chunk) != mtmd_decode_use_non_causal(e->vision, b.chunk))
                    return false;
                const auto * aid = mtmd_input_chunk_get_id(a.chunk);
                const auto * bid = mtmd_input_chunk_get_id(b.chunk);
                if ((aid == nullptr) != (bid == nullptr) || (aid && std::strcmp(aid, bid) != 0)) return false;
                if (mrope) {
                    const auto * at = mtmd_input_chunk_get_tokens_image(a.chunk);
                    const auto * bt = mtmd_input_chunk_get_tokens_image(b.chunk);
                    if (!at || !bt) return false;
                    for (size_t t = 0; t < mtmd_input_chunk_get_n_tokens(a.chunk); ++t) {
                        const auto ap = mtmd_image_tokens_get_decoder_pos(at, 0, t);
                        const auto bp = mtmd_image_tokens_get_decoder_pos(bt, 0, t);
                        if (ap.t != bp.t || ap.y != bp.y || ap.x != bp.x || ap.z != bp.z) return false;
                    }
                }
                return true;
            };
            for (auto * seg : media) {
                const auto same = std::find_if(unique.begin(), unique.end(),
                    [&](const segment * other) { return equal_image(*seg, *other); });
                if (same == unique.end()) unique.push_back(seg);
                else duplicates.emplace_back(seg, *same);
            }
            media = std::move(unique);
        }
        const uint64_t maximum = uint64_t(e->context_size) * capacity;
        if (maximum > uint64_t(INT_MAX)) throw std::runtime_error("parallel vision context exceeds INT_MAX");
        // Separate streams divide the context evenly. Reserve the longest
        // image prompt in every stream, rather than the sum of unequal lengths.
        // Token batch headroom remains within the configured per-image limit.
        const uint64_t per_sequence = std::min(uint64_t(e->context_size),
            uint64_t(maximum_tokens) + e->batch_size);
        const uint32_t requested = dynamic_context && capacity > 1 ?
            uint32_t(per_sequence * capacity) : uint32_t(maximum);
        ensure_sequences(e, capacity, requested, dynamic_context, false);
        sd_clear(e);
        // Reuse encodes each unique image chunk independently: changing a
        // different image must not change this image's projector batch shape or
        // order. Embedding reuse and projector batching are alternative paths;
        // decoder rows still batch across all independent sequences below.
        // The default path retains upstream compatible projector batching.
        for (size_t start = 0; start < media.size();) {
            std::unique_ptr<mtmd_batch, decltype(&mtmd_batch_free)> batch(mtmd_batch_init(e->vision), mtmd_batch_free);
            if (!batch) throw std::runtime_error("vision projector batch allocation failed");
            size_t end = start;
            while (end < media.size()) {
                const int rc = mtmd_batch_add_chunk(batch.get(), media[end]->chunk);
                if (rc == 0) {
                    ++end;
                    if (e->vision_projector_reuse) break;
                    continue;
                }
                if (end > start && (rc == 2 || rc == 3)) break;
                throw std::runtime_error("vision projector batch preparation failed");
            }
            if (mtmd_batch_encode(batch.get()) != 0)
                throw std::runtime_error("vision projector batch encoding failed");
            ++e->vision_metrics.projector_encode_calls;
            e->vision_metrics.projector_batch_max = std::max(e->vision_metrics.projector_batch_max, end - start);
            for (size_t i = start; i < end; ++i) {
                const auto * embd = mtmd_batch_get_output_embd(batch.get(), media[i]->chunk);
                if (!embd) throw std::runtime_error("missing batched vision embeddings");
                const size_t count = mtmd_input_chunk_get_n_tokens(media[i]->chunk) * n_embd;
                media[i]->embeddings = std::make_shared<std::vector<float>>(embd, embd + count);
            }
            start = end;
        }
        for (const auto & duplicate : duplicates) duplicate.first->embeddings = duplicate.second->embeddings;
        e->vision_metrics.projector_reused_chunks = duplicates.size();
        auto advance = [&](sequence & seq) {
            while (seq.current < seq.segments.size()) {
                auto & seg = seq.segments[seq.current];
                if (seg.offset < mtmd_input_chunk_get_n_tokens(seg.chunk)) break;
                seq.position += mtmd_input_chunk_get_n_pos(seg.chunk);
                ++seq.current;
            }
        };
        struct row { int32_t seq; segment * seg; size_t offset; llama_pos base; bool output; };
        int32_t completed = 0;
        while (completed < sequences) {
            for (auto & seq : seqs) advance(seq);
            // Token and embedding inputs cannot coexist in llama_batch. Run all
            // ready text rows together, then all ready image rows together.
            bool text_mode = false;
            for (auto & seq : seqs) if (seq.current < seq.segments.size() &&
                    mtmd_input_chunk_get_type(seq.segments[seq.current].chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) text_mode = true;
            bool non_causal = false;
            if (!text_mode) for (auto & seq : seqs) if (seq.current < seq.segments.size()) {
                non_causal = mtmd_decode_use_non_causal(e->vision, seq.segments[seq.current].chunk);
                break;
            }
            std::vector<row> rows;
            const size_t limit = non_causal ? std::min(e->batch_size, llama_n_ubatch(e->ctx)) : e->batch_size;
            if (non_causal) {
                // Bidirectional image attention must see the complete image in
                // one physical microbatch. Several complete independent images
                // may share it; sequence IDs keep their masks isolated.
                for (int32_t s = 0; s < sequences; ++s) {
                    auto & seq = seqs[size_t(s)];
                    if (seq.current == seq.segments.size()) continue;
                    auto & seg = seq.segments[seq.current];
                    if (!mtmd_decode_use_non_causal(e->vision, seg.chunk)) continue;
                    const size_t count = mtmd_input_chunk_get_n_tokens(seg.chunk);
                    if (count > limit) throw std::runtime_error("non-causal vision image requires batch and ubatch >= image tokens");
                    if (rows.size() + count > limit) continue;
                    while (seg.offset < count) rows.push_back({s, &seg, seg.offset++, seq.position, false});
                }
            } else {
                bool progress = true;
                while (progress && rows.size() < limit) {
                    progress = false;
                    for (int32_t s = 0; s < sequences && rows.size() < limit; ++s) {
                        auto & seq = seqs[size_t(s)];
                        advance(seq);
                        if (seq.current == seq.segments.size()) continue;
                        auto & seg = seq.segments[seq.current];
                        const bool text = mtmd_input_chunk_get_type(seg.chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT;
                        if (text != text_mode || (!text && mtmd_decode_use_non_causal(e->vision, seg.chunk))) continue;
                        const bool output = seq.current + 1 == seq.segments.size() &&
                            seg.offset + 1 == mtmd_input_chunk_get_n_tokens(seg.chunk);
                        rows.push_back({s, &seg, seg.offset++, seq.position, output});
                        progress = true;
                    }
                }
            }
            if (rows.empty()) throw std::runtime_error("parallel vision scheduler made no progress");
            const size_t count = rows.size();
            std::vector<llama_token> tokens(text_mode ? count : 0);
            std::vector<float> embeddings(text_mode ? 0 : count * n_embd);
            const size_t axes = !text_mode && mrope ? 4 : 1;
            std::vector<llama_pos> positions(count * axes);
            std::vector<int32_t> n_seq(count, 1);
            std::vector<llama_seq_id> ids(count);
            std::vector<llama_seq_id *> id_ptrs(count);
            std::vector<int8_t> outputs(count, 0);
            std::vector<bool> participating(size_t(sequences), false);
            for (size_t i = 0; i < count; ++i) {
                const auto & row = rows[i];
                ids[i] = row.seq; id_ptrs[i] = &ids[i]; outputs[i] = row.output;
                participating[size_t(row.seq)] = true;
                if (text_mode) {
                    size_t size;
                    const auto * ts = mtmd_input_chunk_get_tokens_text(row.seg->chunk, &size);
                    tokens[i] = ts[row.offset];
                    positions[i] = row.base + llama_pos(row.offset);
                } else {
                    std::copy_n(row.seg->embeddings->data() + row.offset * n_embd, n_embd, embeddings.data() + i * n_embd);
                    if (mrope) {
                        const auto * image = mtmd_input_chunk_get_tokens_image(row.seg->chunk);
                        if (!image) throw std::runtime_error("missing image position metadata");
                        const auto pos = mtmd_image_tokens_get_decoder_pos(image, row.base, row.offset);
                        positions[i] = pos.t;
                        positions[i + count] = pos.y;
                        positions[i + count * 2] = pos.x;
                        positions[i + count * 3] = pos.z;
                    } else positions[i] = row.base + llama_pos(row.offset);
                }
            }
            llama_batch batch = {int32_t(count), text_mode ? tokens.data() : nullptr,
                text_mode ? nullptr : embeddings.data(), positions.data(), n_seq.data(), id_ptrs.data(), outputs.data()};
            struct attention_guard {
                llama_context * ctx; bool non_causal;
                ~attention_guard() { if (non_causal) llama_set_causal_attn(ctx, true); }
            } guard{e->ctx, non_causal};
            if (non_causal) llama_set_causal_attn(e->ctx, false);
            e->memory_dirty = true;
            const int rc = llama_decode(e->ctx, batch);
            if (rc != 0) throw std::runtime_error("parallel vision decode failed (llama_decode=" + std::to_string(rc) + ")");
            ++e->vision_metrics.decoder_calls;
            const size_t width = size_t(std::count(participating.begin(), participating.end(), true));
            e->vision_metrics.decoder_batch_max_sequences = std::max(e->vision_metrics.decoder_batch_max_sequences, width);
            for (size_t i = 0; i < count; ++i) if (rows[i].output) {
                const auto * output = llama_get_logits_ith(e->ctx, int32_t(i));
                if (!output) throw std::runtime_error("missing parallel vision final logits");
                const size_t seq = size_t(rows[i].seq);
                if (candidate_ids) copy_compact_logits(output, int32_t(vocab), candidate_ids[seq],
                    candidate_counts[seq], logits + candidate_offsets[seq], log_normalizers + seq);
                else std::copy_n(output, vocab, logits + seq * vocab);
                ++completed;
            }
        }
        sd_clear(e);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    if (e) sd_clear(e);
    return false;
}

extern "C" bool sd_forward_vision_parallel(engine * e, const vision_input * inputs,
        int32_t sequences, uint32_t capacity, bool dynamic_context,
        float * logits, size_t logits_count, size_t * input_tokens,
        char * error, size_t error_cap) noexcept {
    return forward_vision_parallel(e, inputs, sequences, capacity, dynamic_context,
        logits, logits_count, input_tokens, nullptr, nullptr, nullptr, error, error_cap);
}
extern "C" bool sd_forward_vision_parallel_compact(engine * e, const vision_input * inputs,
        int32_t sequences, uint32_t capacity, bool dynamic_context,
        const int32_t * const * candidate_ids, const size_t * candidate_counts,
        float * logits, size_t logits_count, double * log_normalizers, size_t * input_tokens,
        char * error, size_t error_cap) noexcept {
    if (!candidate_ids || !candidate_counts || !log_normalizers) { if (e) sd_clear(e); report(error, error_cap, "null parallel vision compact argument"); return false; }
    return forward_vision_parallel(e, inputs, sequences, capacity, dynamic_context,
        logits, logits_count, input_tokens, candidate_ids, candidate_counts, log_normalizers, error, error_cap);
}

// Compact transfer only: llama.cpp still computes and exposes the entire
// vocabulary on the host. Keep its exact normalizer so candidate mass retains
// the same meaning as the full-logit path. This is not a GPU reduction.
extern "C" bool sd_forward_compact(engine * e, const int32_t * tokens, int32_t count,
        bool reuse, int32_t * reused, const int32_t * candidate_ids, size_t candidate_count,
        float * candidate_logits, size_t candidate_logits_count, double * log_normalizer,
        int32_t * vocabulary_size, char * error, size_t error_cap) noexcept {
    if (reused) *reused = 0;
    try {
        if (!e || !tokens || !reused || !candidate_ids || !candidate_logits ||
                !log_normalizer || !vocabulary_size)
            throw std::runtime_error("null compact evidence argument");
        const int32_t vocab = sd_vocab_size(e);
        if (vocab <= 0 || candidate_count < 2 || candidate_count > 26 ||
                candidate_logits_count != candidate_count)
            throw std::runtime_error("invalid compact evidence buffer size");
        for (size_t i = 0; i < candidate_count; ++i) {
            if (candidate_ids[i] < 0 || candidate_ids[i] >= vocab)
                throw std::runtime_error("candidate token outside vocabulary");
            for (size_t j = 0; j < i; ++j) {
                if (candidate_ids[i] == candidate_ids[j])
                    throw std::runtime_error("duplicate candidate token");
            }
        }
        const float * output = forward_logits(e, tokens, count, reuse, reused);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int32_t i = 0; i < vocab; ++i) {
            const double value = output[i];
            if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
                throw std::runtime_error("invalid vocabulary logit");
            maximum = std::max(maximum, value);
        }
        if (!std::isfinite(maximum)) throw std::runtime_error("no finite vocabulary logit");
        for (size_t i = 0; i < candidate_count; ++i) {
            if (!std::isfinite(output[candidate_ids[i]]))
                throw std::runtime_error("nonfinite candidate logit");
        }
        double sum = 0;
        for (int32_t i = 0; i < vocab; ++i) sum += std::exp(double(output[i]) - maximum);
        const double normalizer = maximum + std::log(sum);
        if (!std::isfinite(normalizer)) throw std::runtime_error("invalid vocabulary normalizer");
        for (size_t i = 0; i < candidate_count; ++i) candidate_logits[i] = output[candidate_ids[i]];
        *log_normalizer = normalizer;
        *vocabulary_size = vocab;
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    if (reused) *reused = 0;
    if (e) sd_clear(e);
    return false;
}

// Independent questions share only their exact common token prefix. Sequence
// IDs isolate suffix attention; logits are copied by original batch token index.
extern "C" bool sd_forward_parallel(engine * e, const int32_t * const * tokens,
        const int32_t * counts, int32_t sequences, uint32_t capacity, bool dynamic_context, int32_t * reused,
        float * logits, size_t logits_count, char * error, size_t error_cap) noexcept {
    try {
        if (sequences < 1 || sequences > int32_t(capacity) || capacity > 32)
            throw std::runtime_error("invalid parallel sequence count");
        if (llama_model_is_recurrent(e->model) || llama_model_is_hybrid(e->model))
            throw std::runtime_error("parallel prefix sharing is unsupported for recurrent/hybrid models");
        const size_t vocab = size_t(sd_vocab_size(e));
        if (logits_count != vocab * size_t(sequences))
            throw std::runtime_error("wrong parallel logits buffer size");
        uint64_t total_tokens = 0;
        for (int32_t s = 0; s < sequences; ++s) {
            reused[s] = 0;
            if (counts[s] <= 0 || uint32_t(counts[s]) > e->context_size)
                throw std::runtime_error("parallel input exceeds per-question context; truncation is disabled");
            total_tokens += uint32_t(counts[s]);
        }
        const uint64_t maximum_context = uint64_t(e->context_size) * capacity;
        const uint32_t requested_context = dynamic_context && capacity > 1 ?
            uint32_t(std::min(maximum_context, total_tokens + e->batch_size)) :
            uint32_t(maximum_context);
        ensure_sequences(e, capacity, requested_context, dynamic_context);
        sd_clear(e);
        int32_t common = counts[0] - 1;
        for (int32_t s = 1; s < sequences; ++s) {
            common = std::min(common, counts[s] - 1);
            int32_t i = 0;
            while (i < common && tokens[0][i] == tokens[s][i]) ++i;
            common = i;
        }
        if (sequences == 1) common = 0;
        common -= common % int32_t(e->batch_size);
        struct batch_guard {
            llama_batch value;
            ~batch_guard() { llama_batch_free(value); }
        } b { llama_batch_init(int32_t(e->batch_size), 0, 1) };
        if (!b.value.token || !b.value.pos || !b.value.n_seq_id || !b.value.seq_id || !b.value.logits)
            throw std::runtime_error("parallel batch allocation failed");
        auto add = [&](int32_t s, int32_t position, bool output) {
            const int32_t j = b.value.n_tokens++;
            b.value.token[j] = tokens[s][position];
            b.value.pos[j] = position;
            b.value.n_seq_id[j] = 1;
            b.value.seq_id[j][0] = s;
            b.value.logits[j] = output;
            return j;
        };
        for (int32_t start = 0; start < common;) {
            b.value.n_tokens = 0;
            while (start < common && b.value.n_tokens < int32_t(e->batch_size)) add(0, start++, false);
            e->memory_dirty = true;
            const int32_t rc = llama_decode(e->ctx, b.value);
            if (rc != 0) {
                throw std::runtime_error("parallel shared prefill failed (llama_decode=" + std::to_string(rc) +
                    ", batch_tokens=" + std::to_string(b.value.n_tokens) +
                    ", context=" + std::to_string(llama_n_ctx(e->ctx)) +
                    "; code 1 means no KV slot for this batch; see llama.cpp stderr for other codes)");
            }
        }
        if (common > 0) {
            auto memory = llama_get_memory(e->ctx);
            for (int32_t s = 1; s < sequences; ++s) {
                llama_memory_seq_cp(memory, 0, s, 0, common);
                reused[s] = common;
            }
        }
        std::vector<int32_t> positions(size_t(sequences), common);
        int32_t completed = 0;
        while (completed < sequences) {
            b.value.n_tokens = 0;
            std::vector<std::pair<int32_t, int32_t>> outputs;
            bool added = true;
            while (added && b.value.n_tokens < int32_t(e->batch_size)) {
                added = false;
                for (int32_t s = 0; s < sequences && b.value.n_tokens < int32_t(e->batch_size); ++s) {
                    if (positions[s] == counts[s]) continue;
                    const bool last = positions[s] == counts[s] - 1;
                    const int32_t j = add(s, positions[s]++, last);
                    if (last) outputs.emplace_back(s, j);
                    added = true;
                }
            }
            if (b.value.n_tokens == 0) throw std::runtime_error("parallel scheduler made no progress");
            e->memory_dirty = true;
            const int32_t rc = llama_decode(e->ctx, b.value);
            if (rc != 0) {
                throw std::runtime_error("parallel suffix decode failed (llama_decode=" + std::to_string(rc) +
                    ", batch_tokens=" + std::to_string(b.value.n_tokens) +
                    ", completed=" + std::to_string(completed) + "/" + std::to_string(sequences) +
                    ", context=" + std::to_string(llama_n_ctx(e->ctx)) +
                    "; code 1 means no KV slot for this batch; see llama.cpp stderr for other codes)");
            }
            for (const auto & item : outputs) {
                const auto * output = llama_get_logits_ith(e->ctx, item.second);
                if (!output) throw std::runtime_error("missing parallel final logits");
                std::copy_n(output, vocab, logits + size_t(item.first) * vocab);
                ++completed;
            }
        }
        sd_clear(e);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    sd_clear(e);
    return false;
}

extern "C" bool sd_recurrent_or_hybrid(const engine * e) noexcept {
    return llama_model_is_recurrent(e->model) || llama_model_is_hybrid(e->model);
}
extern "C" uint32_t sd_context_tokens(const engine * e) noexcept {
    return e->ctx ? llama_n_ctx(e->ctx) : 0;
}
extern "C" uint32_t sd_training_context(const engine * e) noexcept { return llama_model_n_ctx_train(e->model); }
struct restore_metrics {
    size_t snapshot_bytes;
    double save_ms, restore_ms, prefill_ms, suffix_ms;
    size_t restores;
    int32_t fallback; // 1=no aligned prefix, 2=budget, 3=save unavailable, 4=restore unavailable
};
extern "C" bool sd_forward_restore(engine * e, const int32_t * const * tokens,
        const int32_t * counts, int32_t sequences, size_t limit, int32_t * reused,
        float * logits, size_t logits_count, restore_metrics * metrics,
        char * error, size_t error_cap) noexcept {
    *metrics = {};
    try {
        ensure_sequences(e, 1);
        sd_clear(e);
        const size_t vocab = size_t(sd_vocab_size(e));
        if (sequences < 1 || logits_count != vocab * size_t(sequences))
            throw std::runtime_error("invalid snapshot sequence or logits count");
        int32_t common = counts[0] - 1;
        for (int32_t s = 0; s < sequences; ++s) {
            reused[s] = 0;
            if (counts[s] <= 0 || uint32_t(counts[s]) > e->context_size)
                throw std::runtime_error("snapshot input exceeds context");
            common = std::min(common, counts[s] - 1);
            int32_t i = 0;
            while (i < common && tokens[0][i] == tokens[s][i]) ++i;
            common = i;
        }
        if (sequences == 1) common = 0;
        common -= common % int32_t(e->batch_size);
        using clock = std::chrono::steady_clock;
        auto elapsed = [](clock::time_point start) { return std::chrono::duration<double, std::milli>(clock::now()-start).count(); };
        auto fresh = [&]() {
            for (int32_t s = 0; s < sequences; ++s) {
                auto start = clock::now();
                if (!sd_forward(e, tokens[s], counts[s], false, &reused[s], logits+size_t(s)*vocab, vocab, error, error_cap)) return false;
                metrics->suffix_ms += elapsed(start);
            }
            sd_clear(e);
            return true;
        };
        if (common == 0) { metrics->fallback = 1; return fresh(); }
        struct batch_guard { llama_batch value; ~batch_guard() { llama_batch_free(value); } } b { llama_batch_init(int32_t(e->batch_size), 0, 1) };
        if (!b.value.token || !b.value.pos || !b.value.n_seq_id || !b.value.seq_id || !b.value.logits)
            throw std::runtime_error("snapshot batch allocation failed");
        auto decode = [&](int32_t s, int32_t start, int32_t end, bool final) {
            while (start < end) {
                b.value.n_tokens = std::min(int32_t(e->batch_size), end-start);
                for (int32_t j=0; j<b.value.n_tokens; ++j) {
                    b.value.token[j]=tokens[s][start+j]; b.value.pos[j]=start+j;
                    b.value.n_seq_id[j]=1; b.value.seq_id[j][0]=0;
                    b.value.logits[j]=final && start+j==end-1;
                }
                e->memory_dirty = true;
                if (llama_decode(e->ctx,b.value)!=0) throw std::runtime_error("snapshot decode failed");
                start+=b.value.n_tokens;
            }
        };
        auto start = clock::now();
        decode(0,0,common,false);
        metrics->prefill_ms = elapsed(start);
        start = clock::now();
        const size_t bytes = llama_state_seq_get_size(e->ctx,0);
        if (bytes == 0) { metrics->fallback=3; return fresh(); }
        if (bytes > limit) { metrics->fallback=2; return fresh(); }
        // The only snapshot allocation is checked before allocation and scoped to this call.
        std::vector<uint8_t> snapshot(bytes);
        metrics->snapshot_bytes=bytes;
        if (llama_state_seq_get_data(e->ctx,snapshot.data(),bytes,0)!=bytes) {
            metrics->fallback=3; return fresh();
        }
        metrics->save_ms=elapsed(start);
        for (int32_t s=0; s<sequences; ++s) {
            // The first suffix can use the state left by the common prefill.
            // Restore only for later, independent suffixes; the counter is the
            // number of actual state loads, not the number of decisions.
            if (s > 0) {
                start=clock::now();
                sd_clear(e);
                e->memory_dirty = true;
                if (llama_state_seq_set_data(e->ctx,snapshot.data(),bytes,0)!=bytes) {
                    metrics->fallback=4; metrics->restores=0; return fresh();
                }
                metrics->restore_ms+=elapsed(start);
                ++metrics->restores;
            }
            start=clock::now();
            decode(s,common,counts[s],true);
            const float * output=llama_get_logits_ith(e->ctx,-1);
            if (!output) throw std::runtime_error("missing snapshot final logits");
            std::copy_n(output,vocab,logits+size_t(s)*vocab);
            metrics->suffix_ms+=elapsed(start);
            reused[s]=s==0 ? 0 : common;
        }
        sd_clear(e);
        return true;
    } catch (const std::exception & ex) { report(error,error_cap,ex.what()); }
      catch (...) { report(error,error_cap,"unknown snapshot exception"); }
    sd_clear(e);
    return false;
}
