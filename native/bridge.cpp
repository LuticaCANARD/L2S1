#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
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
    ggml_backend_dev_t devices[2] = {nullptr, nullptr};
    uint32_t batch_size = 0;
    std::string description;
    std::string architecture;
    ~engine() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

static void report(char * out, size_t cap, const char * message) {
    if (cap) std::snprintf(out, cap, "%s", message);
}

extern "C" engine * sd_open(const char * path, uint32_t context, uint32_t batch,
        int32_t threads, bool cuda, char * error, size_t error_cap) noexcept {
    try {
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
        cp.n_ubatch = batch;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.embeddings = false;
        cp.offload_kqv = cuda;
        cp.op_offload = cuda;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        e->ctx = llama_init_from_model(e->model, cp);
        if (!e->ctx) throw std::runtime_error("context allocation failed");
        e->batch_size = llama_n_batch(e->ctx);
        char desc[512] = {};
        llama_model_desc(e->model, desc, sizeof(desc));
        e->description = desc;
        return e.release();
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    return nullptr;
}

extern "C" void sd_close(engine * e) noexcept { delete e; }
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

extern "C" bool sd_forward(engine * e, const int32_t * tokens, int32_t count,
        float * logits, size_t logits_count, char * error, size_t error_cap) noexcept {
    try {
        if (count <= 0 || uint32_t(count) > llama_n_ctx(e->ctx))
            throw std::runtime_error("input exceeds context or is empty; truncation is disabled");
        if (logits_count != size_t(sd_vocab_size(e))) throw std::runtime_error("wrong logits buffer size");
        auto memory = llama_get_memory(e->ctx);
        if (!memory) throw std::runtime_error("decoder memory unavailable");
        llama_memory_clear(memory, true);
        struct batch_guard {
            llama_batch value;
            ~batch_guard() { llama_batch_free(value); }
        } b { llama_batch_init(int32_t(e->batch_size), 0, 1) };
        if (!b.value.token || !b.value.pos || !b.value.n_seq_id || !b.value.seq_id || !b.value.logits)
            throw std::runtime_error("batch allocation failed");
        for (int32_t start = 0; start < count;) {
            b.value.n_tokens = std::min(int32_t(e->batch_size), count - start);
            for (int32_t j = 0; j < b.value.n_tokens; ++j) {
                b.value.token[j] = tokens[start + j];
                b.value.pos[j] = start + j;
                b.value.n_seq_id[j] = 1;
                b.value.seq_id[j][0] = 0;
                b.value.logits[j] = start + j == count - 1;
            }
            if (llama_decode(e->ctx, b.value) != 0) throw std::runtime_error("llama_decode failed");
            start += b.value.n_tokens;
        }
        const float * output = llama_get_logits_ith(e->ctx, -1);
        if (!output) throw std::runtime_error("missing final logits");
        std::copy_n(output, logits_count, logits);
        return true;
    } catch (const std::exception & ex) { report(error, error_cap, ex.what()); }
      catch (...) { report(error, error_cap, "unknown native exception"); }
    return false;
}
