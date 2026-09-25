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
    llama_context_params context_params = {};
    std::string description;
    std::string architecture;
    std::string runtime_libraries;
    std::vector<int32_t> cached_tokens;
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
            ggml_backend_load_all();
            llama_backend_init();
        });
        auto e = std::make_unique<engine>();
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
    e->cached_tokens.clear();
    e->last_feature_row = -1;
    if (e->ctx) {
        if (auto memory = llama_get_memory(e->ctx)) llama_memory_clear(memory, true);
    }
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
        uint32_t requested_context = 0, bool dynamic_context = false) {
    if (capacity < 1 || capacity > 32 || e->context_size > uint32_t(INT_MAX) / capacity)
        throw std::runtime_error("parallel width requires 1..32 and context * width <= INT_MAX");
    const uint32_t maximum_context = e->context_size * capacity;
    if (requested_context == 0) requested_context = maximum_context;
    if (requested_context > maximum_context)
        throw std::runtime_error("parallel context reservation is outside the configured limits");
    if (e->ctx && e->sequence_capacity == capacity &&
            e->parallel_context_dynamic == dynamic_context &&
            (dynamic_context ? e->allocated_context_size >= requested_context :
                e->allocated_context_size == requested_context)) return;
    sd_clear(e);
    if (e->ctx) llama_free(e->ctx);
    e->ctx = nullptr;
    auto cp = e->context_params;
    cp.n_seq_max = capacity;
    cp.n_ctx = requested_context;
    cp.kv_unified = capacity > 1 ? true : e->context_params.kv_unified;
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

// Vision input is evaluated from fresh request-local state. The five prompt
// parts preserve the trust boundary between template control tokens and data.
extern "C" bool sd_forward_vision(engine * e,
        const char * prefix, size_t prefix_len,
        const char * data_before, size_t before_len,
        const uint8_t * image, size_t image_len,
        const char * data_after, size_t after_len,
        const char * suffix, size_t suffix_len,
        const int32_t * continuation, size_t continuation_count,
        float * logits, size_t logits_count, size_t * input_tokens,
        char * error, size_t error_cap) noexcept {
    try {
        if (!e || !e->vision || !prefix || !data_before || !image || !image_len ||
                !data_after || !suffix || (continuation_count && !continuation) ||
                !logits || !input_tokens ||
                logits_count != size_t(sd_vocab_size(e)))
            throw std::runtime_error("invalid vision inference arguments");
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
                if (llama_decode(e->ctx, b.value) != 0)
                    throw std::runtime_error("vision continuation decode failed");
                start += size_t(b.value.n_tokens);
            }
        }
        const float * output = llama_get_logits_ith(e->ctx, -1);
        if (!output) throw std::runtime_error("missing vision final logits");
        std::copy_n(output, logits_count, logits);
        *input_tokens = tokens;
        sd_clear(e);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    if (e) sd_clear(e);
    return false;
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
