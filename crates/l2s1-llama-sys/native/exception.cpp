#include "exception.h"
#include <cstdio>
#include <exception>
#ifdef L2S1_ARM64_DISPATCH
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#endif
namespace {
thread_local char error[1024] = {};
void record(const char * message) noexcept {
    if (!error[0]) std::snprintf(error, sizeof(error), "%s", message);
}
}
extern "C" void sd_native_reset_error() noexcept { error[0] = 0; }
extern "C" const char * sd_native_error() noexcept { return error; }
#define L2S1_CATCH(fallback) \
    catch (const std::exception & ex) { record(ex.what()); return fallback; } \
    catch (...) { record("unknown upstream exception"); return fallback; }
extern "C" llama_model * sd_native_llama_model_load_from_file(const char * path, llama_model_params params) noexcept {
    try { return llama_model_load_from_file(path, params); } L2S1_CATCH(nullptr)
}
extern "C" llama_context * sd_native_llama_init_from_model(llama_model * model, llama_context_params params) noexcept {
    try { return llama_init_from_model(model, params); } L2S1_CATCH(nullptr)
}
extern "C" llama_adapter_lora * sd_native_llama_adapter_lora_init(llama_model * model, const char * path) noexcept {
    try { return llama_adapter_lora_init(model, path); } L2S1_CATCH(nullptr)
}
extern "C" int32_t sd_native_llama_set_adapters_lora(llama_context * ctx, llama_adapter_lora ** adapters, size_t count, float * scales) noexcept {
    try { return llama_set_adapters_lora(ctx, adapters, count, scales); } L2S1_CATCH(-1)
}
extern "C" llama_batch sd_native_llama_batch_init(int32_t tokens, int32_t embd, int32_t sequences) noexcept {
    try { return llama_batch_init(tokens, embd, sequences); } L2S1_CATCH({})
}
extern "C" int32_t sd_native_llama_decode(llama_context * ctx, llama_batch batch) noexcept {
    try { return llama_decode(ctx, batch); } L2S1_CATCH(-1)
}
extern "C" int32_t sd_native_llama_tokenize(const llama_vocab * vocab, const char * text, int32_t length, llama_token * tokens, int32_t capacity, bool add_special, bool parse_special) noexcept {
    try { return llama_tokenize(vocab, text, length, tokens, capacity, add_special, parse_special); } L2S1_CATCH(INT32_MIN)
}
extern "C" size_t sd_native_llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq) noexcept {
    try { return llama_state_seq_get_size(ctx, seq); } L2S1_CATCH(0)
}
extern "C" size_t sd_native_llama_state_seq_get_data(llama_context * ctx, uint8_t * data, size_t size, llama_seq_id seq) noexcept {
    try { return llama_state_seq_get_data(ctx, data, size, seq); } L2S1_CATCH(0)
}
extern "C" size_t sd_native_llama_state_seq_set_data(llama_context * ctx, const uint8_t * data, size_t size, llama_seq_id seq) noexcept {
    try { return llama_state_seq_set_data(ctx, data, size, seq); } L2S1_CATCH(0)
}
extern "C" mtmd_context * sd_native_mtmd_init_from_file(const char * path, const llama_model * model, mtmd_context_params params) noexcept {
    try { return mtmd_init_from_file(path, model, params); } L2S1_CATCH(nullptr)
}
extern "C" mtmd_input_chunks * sd_native_mtmd_input_chunks_init() noexcept {
    try { return mtmd_input_chunks_init(); } L2S1_CATCH(nullptr)
}
extern "C" mtmd_helper_bitmap_wrapper sd_native_mtmd_helper_bitmap_init_from_buf(const mtmd_context * ctx, const unsigned char * data, size_t size, bool placeholder, mtmd_helper_init_opt opt) noexcept {
    try { return mtmd_helper_bitmap_init_from_buf(ctx, data, size, placeholder, opt); } L2S1_CATCH({})
}
extern "C" int32_t sd_native_mtmd_tokenize_from_parts(const mtmd_context * ctx, mtmd_input_chunks * output, const mtmd_input_part * const * parts, size_t count, bool special) noexcept {
    try { return mtmd_tokenize_from_parts(ctx, output, parts, count, special); } L2S1_CATCH(-1)
}
extern "C" int32_t sd_native_mtmd_helper_eval_chunks(mtmd_context * ctx, llama_context * lctx, const mtmd_input_chunks * chunks, llama_pos past, llama_seq_id seq, int32_t batch, bool logits, llama_pos * end) noexcept {
    try { return mtmd_helper_eval_chunks(ctx, lctx, chunks, past, seq, batch, logits, end); } L2S1_CATCH(-1)
}
extern "C" mtmd_batch * sd_native_mtmd_batch_init(mtmd_context * ctx) noexcept {
    try { return mtmd_batch_init(ctx); } L2S1_CATCH(nullptr)
}
extern "C" int32_t sd_native_mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) noexcept {
    try { return mtmd_batch_add_chunk(batch, chunk); } L2S1_CATCH(-1)
}
extern "C" int32_t sd_native_mtmd_batch_encode(mtmd_batch * batch) noexcept {
    try { return mtmd_batch_encode(batch); } L2S1_CATCH(-1)
}
extern "C" void sd_native_ggml_backend_load_all() noexcept {
    try {
#ifdef L2S1_ARM64_DISPATCH
        // Resolve modules beside the linked GGML, including during cargo test.
        // Do not search the cwd or another build's absolute install directory.
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void *>(&ggml_backend_load_all_from_path), &info) || !info.dli_fname) {
            throw std::runtime_error("cannot locate matching GGML backend directory");
        }
        const auto directory = std::filesystem::canonical(info.dli_fname).parent_path().string();
        ggml_backend_load_all_from_path(directory.c_str());
#else
        ggml_backend_load_all();
#endif
    } L2S1_CATCH()
}
extern "C" void sd_native_llama_backend_init() noexcept {
    try { return llama_backend_init(); } L2S1_CATCH()
}
extern "C" void sd_native_llama_memory_clear(llama_memory_t memory, bool data) noexcept {
    try { return llama_memory_clear(memory, data); } L2S1_CATCH()
}
extern "C" bool sd_native_llama_memory_seq_rm(llama_memory_t memory, llama_seq_id seq, llama_pos start, llama_pos end) noexcept {
    try { return llama_memory_seq_rm(memory, seq, start, end); } L2S1_CATCH(false)
}
extern "C" void sd_native_llama_memory_seq_cp(llama_memory_t memory, llama_seq_id src, llama_seq_id dst, llama_pos start, llama_pos end) noexcept {
    try { return llama_memory_seq_cp(memory, src, dst, start, end); } L2S1_CATCH()
}
extern "C" int32_t sd_native_llama_model_meta_val_str(const llama_model * model, const char * key, char * out, size_t size) noexcept {
    try { return llama_model_meta_val_str(model, key, out, size); } L2S1_CATCH(-1)
}
extern "C" int32_t sd_native_llama_model_desc(const llama_model * model, char * out, size_t size) noexcept {
    try { return llama_model_desc(model, out, size); } L2S1_CATCH(-1)
}
extern "C" const char * sd_native_llama_model_chat_template(const llama_model * model, const char * name) noexcept {
    try { return llama_model_chat_template(model, name); } L2S1_CATCH(nullptr)
}
extern "C" int32_t sd_native_llama_token_to_piece(const llama_vocab * vocab, llama_token token, char * out, int32_t size, int32_t lstrip, bool special) noexcept {
    try { return llama_token_to_piece(vocab, token, out, size, lstrip, special); } L2S1_CATCH(INT32_MIN)
}
