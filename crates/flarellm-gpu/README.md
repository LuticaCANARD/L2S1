This is the `flarellm-gpu` crate from [FlareLLM](https://github.com/sauravpanda/flarellm) at commit `a8cb07cf54fa43c4e41664316498ed195fa7b018`, with two optional shader features disabled in `src/backend.rs` for wgpu 24 on Apple Metal. The original source is licensed MIT OR Apache-2.0; see `LICENSE`.

The baseline compute shaders still execute model prefill and decode on the GPU. Keep this compatibility fork pinned to the matching `flarellm-core` revision when updating.
