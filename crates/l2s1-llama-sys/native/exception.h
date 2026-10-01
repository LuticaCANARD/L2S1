#pragma once
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
// C++ exceptions must never unwind into Rust. No scheduling/scoring policy lives here.
extern "C" {
void sd_native_reset_error() noexcept;
const char * sd_native_error() noexcept;
llama_model * sd_native_llama_model_load_from_file(const char * path, llama_model_params params) noexcept;
llama_context * sd_native_llama_init_from_model(llama_model * model, llama_context_params params) noexcept;
llama_adapter_lora * sd_native_llama_adapter_lora_init(llama_model * model, const char * path) noexcept;
int32_t sd_native_llama_set_adapters_lora(llama_context * ctx, llama_adapter_lora ** adapters, size_t count, float * scales) noexcept;
llama_batch sd_native_llama_batch_init(int32_t tokens, int32_t embd, int32_t sequences) noexcept;
int32_t sd_native_llama_decode(llama_context * ctx, llama_batch batch) noexcept;
int32_t sd_native_llama_tokenize(const llama_vocab * vocab, const char * text, int32_t length, llama_token * tokens, int32_t capacity, bool add_special, bool parse_special) noexcept;
size_t sd_native_llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq) noexcept;
size_t sd_native_llama_state_seq_get_data(llama_context * ctx, uint8_t * data, size_t size, llama_seq_id seq) noexcept;
size_t sd_native_llama_state_seq_set_data(llama_context * ctx, const uint8_t * data, size_t size, llama_seq_id seq) noexcept;
mtmd_context * sd_native_mtmd_init_from_file(const char * path, const llama_model * model, mtmd_context_params params) noexcept;
mtmd_input_chunks * sd_native_mtmd_input_chunks_init() noexcept;
mtmd_helper_bitmap_wrapper sd_native_mtmd_helper_bitmap_init_from_buf(const mtmd_context * ctx, const unsigned char * data, size_t size, bool placeholder, mtmd_helper_init_opt opt) noexcept;
int32_t sd_native_mtmd_tokenize_from_parts(const mtmd_context * ctx, mtmd_input_chunks * output, const mtmd_input_part * const * parts, size_t count, bool special) noexcept;
int32_t sd_native_mtmd_helper_eval_chunks(mtmd_context * ctx, llama_context * lctx, const mtmd_input_chunks * chunks, llama_pos past, llama_seq_id seq, int32_t batch, bool logits, llama_pos * end) noexcept;
mtmd_batch * sd_native_mtmd_batch_init(mtmd_context * ctx) noexcept;
int32_t sd_native_mtmd_batch_add_chunk(mtmd_batch * batch, const mtmd_input_chunk * chunk) noexcept;
int32_t sd_native_mtmd_batch_encode(mtmd_batch * batch) noexcept;
void sd_native_ggml_backend_load_all() noexcept;
void sd_native_llama_backend_init() noexcept;
void sd_native_llama_memory_clear(llama_memory_t memory, bool data) noexcept;
bool sd_native_llama_memory_seq_rm(llama_memory_t memory, llama_seq_id seq, llama_pos start, llama_pos end) noexcept;
void sd_native_llama_memory_seq_cp(llama_memory_t memory, llama_seq_id src, llama_seq_id dst, llama_pos start, llama_pos end) noexcept;
int32_t sd_native_llama_model_meta_val_str(const llama_model * model, const char * key, char * out, size_t size) noexcept;
int32_t sd_native_llama_model_desc(const llama_model * model, char * out, size_t size) noexcept;
const char * sd_native_llama_model_chat_template(const llama_model * model, const char * name) noexcept;
int32_t sd_native_llama_token_to_piece(const llama_vocab * vocab, llama_token token, char * out, int32_t size, int32_t lstrip, bool special) noexcept;
}
