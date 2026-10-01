"""Optional CPU architecture checks, using random tiny models, never pretrained weights."""
import json
import unittest

try:
    import torch
    import transformers as t
    import peft
except ImportError:
    torch = t = None

from jev_model_profiles import DEFAULT_PROFILES, attach_adapter, read_profile, target_names


@unittest.skipIf(t is None, 'Optional torch/transformers/peft training environment not installed')
class ModelArchitectureTests(unittest.TestCase):
    def test_every_registered_family_backpropagates_only_into_adapter(self):
        torch.set_num_threads(2)
        common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
        for key in json.loads(DEFAULT_PROFILES.read_text())['models']:
            with self.subTest(model=key):
                torch.manual_seed(20260927)
                profile = read_profile(key)
                kind = profile['model_type']
                if kind == 'llama':
                    config = t.LlamaConfig(**common)
                elif kind == 'qwen3':
                    config = t.Qwen3Config(**common)
                elif kind == 'gemma3_text':
                    config = t.Gemma3TextConfig(**common, sliding_window=16)
                elif kind == 'gemma4':
                    config = t.Gemma4Config(text_config=t.Gemma4TextConfig(**common,
                        vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
                        global_head_dim=8, sliding_window=16))
                else:
                    self.fail('Add an architecture fixture for the new profile')
                model = getattr(t, profile['loader']).from_config(config, attn_implementation='eager')
                self.assertTrue(target_names(model, profile))
                model = attach_adapter(model, profile, dict(rank=2, alpha=4, dropout=0))
                base = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=.001)
                logits = model(input_ids=torch.tensor([[1,4,5,6]]), use_cache=False,
                               logits_to_keep=1).logits[0,-1].float()
                candidate = logits[torch.tensor([10,11,12])]
                loss = -(torch.tensor([.2,.6,.2])*torch.log_softmax(candidate,0)).sum()
                loss += .1*(torch.logsumexp(logits,0)-torch.logsumexp(candidate,0))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
                self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in params))
                self.assertTrue(any(torch.count_nonzero(p) for n,p in model.named_parameters() if 'lora_B' in n))
                for n,p in model.named_parameters():
                    if n in base:
                        self.assertTrue(torch.equal(base[n],p), n)


if __name__ == '__main__':
    unittest.main()
