#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#if defined(__linux__)
#include <link.h>
#endif
#include <climits>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

struct engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_adapter_lora * adapter = nullptr; // owned by model
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    bool features_enabled = false;
    int32_t last_feature_row = -1;
    uint32_t batch_size = 0;
    uint32_t context_size = 0;
    uint32_t sequence_capacity = 1;
    llama_context_params context_params = {};
    std::string description;
    std::string architecture;
    std::string runtime_libraries;
    std::vector<int32_t> cached_tokens;
    ~engine() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

static void report(char * out, size_t cap, const char * message) {
    if (cap) std::snprintf(out, cap, "%s", message);
}

extern "C" engine * sd_open(const char * path, uint32_t context, uint32_t batch,
        uint32_t ubatch, int32_t flash_attention, int32_t threads, bool cuda,
        char * error, size_t error_cap) noexcept {
    try {
        if (ubatch == 0 || ubatch > batch || flash_attention < -1 || flash_attention > 1)
            throw std::runtime_error("invalid microbatch or FlashAttention option");
        static std::once_flag init;
        std::call_once(init, [] { ggml_backend_load_all(); llama_backend_init(); });
        auto e = std::make_unique<engine>();
        auto mp = llama_model_default_params();
        if (cuda) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto dev = ggml_backend_dev_get(i);
                auto reg = ggml_backend_dev_backend_reg(dev);
                if (std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0 &&
                    ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    e->devices[0] = dev;
                    break;
                }
            }
            if (!e->devices[0]) throw std::runtime_error("CUDA device unavailable; select CPU explicitly to use CPU");
        }
        mp.devices = e->devices;
        mp.n_gpu_layers = cuda ? -1 : 0;
        e->model = llama_model_load_from_file(path, mp);
        if (!e->model) throw std::runtime_error("model load failed; see llama.cpp stderr");
        char arch[64] = {};
        llama_model_meta_val_str(e->model, "general.architecture", arch, sizeof(arch));
        e->architecture = arch;
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
        cp.offload_kqv = cuda;
        cp.op_offload = cuda;
        cp.flash_attn_type = static_cast<llama_flash_attn_type>(flash_attention);
        e->context_params = cp;
        e->context_size = context;
        e->ctx = llama_init_from_model(e->model, cp);
        if (!e->ctx) throw std::runtime_error("context allocation failed");
        e->batch_size = llama_n_batch(e->ctx);
        char desc[512] = {};
        llama_model_desc(e->model, desc, sizeof(desc));
        e->description = desc;
#if defined(__linux__)
        std::vector<std::string> libraries;
        dl_iterate_phdr([](dl_phdr_info * info, size_t, void * data) {
            const std::string path = info->dlpi_name;
            const auto name = path.substr(path.find_last_of('/') + 1);
            if (name.rfind("libllama.so", 0) == 0 || name.rfind("libggml", 0) == 0)
                static_cast<std::vector<std::string> *>(data)->push_back(path);
            return 0;
        }, &libraries);
        std::sort(libraries.begin(), libraries.end());
        for (const auto & library : libraries) e->runtime_libraries += library + "\n";
#endif
        return e.release();
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    return nullptr;
}

extern "C" void sd_close(engine * e) noexcept { delete e; }
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
static void ensure_sequences(engine * e, uint32_t capacity) {
    if (capacity < 1 || capacity > 32 || e->context_size > uint32_t(INT_MAX) / capacity)
        throw std::runtime_error("parallel width requires 1..32 and context * width <= INT_MAX");
    if (e->ctx && e->sequence_capacity == capacity) return;
    sd_clear(e);
    if (e->ctx) llama_free(e->ctx);
    e->ctx = nullptr;
    auto cp = e->context_params;
    cp.n_seq_max = capacity;
    cp.n_ctx = e->context_size * capacity;
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
}

extern "C" bool sd_forward(engine * e, const int32_t * tokens, int32_t count,
        bool reuse, int32_t * reused, float * logits, size_t logits_count,
        char * error, size_t error_cap) noexcept {
    *reused = 0;
    try {
        ensure_sequences(e, 1);
        if (count <= 0 || uint32_t(count) > llama_n_ctx(e->ctx))
            throw std::runtime_error("input exceeds context or is empty; truncation is disabled");
        if (logits_count != size_t(sd_vocab_size(e))) throw std::runtime_error("wrong logits buffer size");
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
        std::copy_n(output, logits_count, logits);
        if (reuse) e->cached_tokens.assign(tokens, tokens + count);
        *reused = common;
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    sd_clear(e);
    return false;
}

// Independent questions share only their exact common token prefix. Sequence
// IDs isolate suffix attention; logits are copied by original batch token index.
extern "C" bool sd_forward_parallel(engine * e, const int32_t * const * tokens,
        const int32_t * counts, int32_t sequences, uint32_t capacity, int32_t * reused,
        float * logits, size_t logits_count, char * error, size_t error_cap) noexcept {
    try {
        if (sequences < 1 || sequences > int32_t(capacity) || capacity > 32)
            throw std::runtime_error("invalid parallel sequence count");
        if (llama_model_is_recurrent(e->model) || llama_model_is_hybrid(e->model))
            throw std::runtime_error("parallel prefix sharing is unsupported for recurrent/hybrid models");
        const size_t vocab = size_t(sd_vocab_size(e));
        if (logits_count != vocab * size_t(sequences))
            throw std::runtime_error("wrong parallel logits buffer size");
        for (int32_t s = 0; s < sequences; ++s) {
            reused[s] = 0;
            if (counts[s] <= 0 || uint32_t(counts[s]) > e->context_size)
                throw std::runtime_error("parallel input exceeds per-question context; truncation is disabled");
        }
        ensure_sequences(e, capacity);
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
            if (llama_decode(e->ctx, b.value) != 0) throw std::runtime_error("shared prefill failed");
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
            if (llama_decode(e->ctx, b.value) != 0) throw std::runtime_error("parallel suffix decode failed");
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
            start=clock::now();
            sd_clear(e);
            if (llama_state_seq_set_data(e->ctx,snapshot.data(),bytes,0)!=bytes) {
                metrics->fallback=4; metrics->restores=0; return fresh();
            }
            metrics->restore_ms+=elapsed(start);
            ++metrics->restores;
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
