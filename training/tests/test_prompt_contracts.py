"""Training must consume the same evidence and prompt as deployment."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from l2s1_training.common import digest, read_jsonl, save_status
from l2s1_training.prepare_jev_data import prepare_custom, validate_dataset, decision
from l2s1_training.train_jev_lora import training_rows
from l2s1_training.train_jev_mlx import checkpoint_modules, hf_module


class PromptContracts(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('L2S1_EXPORTER') and os.environ.get('SKID_MODEL'),
                         'requires native exporter and local GGUF')
    def test_real_exported_tokens_pass_training_contract_for_every_prompt(self):
        from l2s1_training.run_jev_lora import seal_tokens
        model, exporter = Path(os.environ['SKID_MODEL']), Path(os.environ['L2S1_EXPORTER'])
        for layout in ('legacy', 'state-first'):
            for detail in ('minimal', 'typed', 'typed-examples'):
                with self.subTest(layout=layout, detail=detail), tempfile.TemporaryDirectory() as tmp:
                    data = self.dataset(Path(tmp), layout, detail)
                    tokens = data/'tokens.jsonl'
                    subprocess.run([str(exporter), '--model', str(model), '--input', str(data/'train-token-requests.jsonl'),
                                    '--output', str(tokens), '--all-rotations', '--context', '8192',
                                    '--prompt-layout', layout, '--prompt-detail', detail], check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    seal_tokens(data, tokens, model, exporter, validate_dataset(data)['protocol'])
                    self.assertEqual(len(training_rows(data, tokens)[0]), 1)

    def dataset(self, root, layout='legacy', detail='minimal'):
        paths = []
        for split in ('train', 'development', 'test'):
            path = root / (split + '.jsonl')
            row = dict(id=split, workflow='shared-policy', state={'case': split},
                       shared={'policy': 'Approve only a verified account.'},
                       questions={'q': dict(type='noul', instructions='Approve?')},
                       gold={'q': dict(type='noul', label='false', probabilities={'false': 1., 'true': 0.})})
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            paths.append(path)
        data = root / 'data'
        prepare_custom(*paths, data, layout=layout, detail=detail)
        return data

    def tokens(self, data, layout='legacy', detail='minimal', context=8192):
        tokens = data / 'tokens.jsonl'
        base = 'gguf-jinja-state-first-v2' if layout == 'state-first' else 'gguf-jinja-decision-v1'
        variant = {'minimal': 'Minimal', 'typed': 'Typed', 'typed-examples': 'TypedExamples'}[detail]
        rows = [dict(id='train/q', decision_id='q', option_ids=['false', 'true'], code_rotation=r,
                     candidate_ids=[65 + (i + 2-r) % 2 for i in range(2)],
                     candidate_codes=[chr(65 + (i + 2-r) % 2) for i in range(2)], input_ids=[1, 2, 3],
                     model_identity=dict(weights_sha256='a'*64,
                         prompt_version=base if detail == 'minimal' and r == 0 else
                             f'{base}/detail-{variant}-v1/rotation-{r}')) for r in range(2)]
        tokens.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        seal = dict(schema_version=1, tokens_sha256=digest(tokens), manifest_sha256=digest(data/'manifest.json'),
                    requests_sha256=digest(data/'train-token-requests.jsonl'), model_sha256='a'*64,
                    prompt_layout=layout, prompt_detail=detail)
        if context is not None:
            seal['context'] = context
        save_status(tokens.with_suffix('.seal.json'), seal)
        return tokens

    def test_shared_survives_prepare_and_each_export_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.dataset(Path(tmp))
            validate_dataset(data)
            for split in ('train', 'development', 'test'):
                request = read_jsonl(data/f'{split}-requests.jsonl')[0]['request']
                self.assertEqual(request['shared'], {'policy': 'Approve only a verified account.'})
                self.assertNotIn('gold', request)
            flat = read_jsonl(data/'train-token-requests.jsonl')
            self.assertEqual(flat[0]['request']['shared'], request['shared'])
            flat[0]['request'].pop('shared')
            path = data/'train-token-requests.jsonl'
            path.write_text(json.dumps(flat[0])+'\n', encoding='utf-8')
            manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
            manifest['token_requests_sha256'] = digest(path)
            save_status(data/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Training request binding'):
                validate_dataset(data)

    def test_all_prompt_settings_and_old_seals(self):
        for layout in ('legacy', 'state-first'):
            for detail in ('minimal', 'typed', 'typed-examples'):
                with self.subTest(layout=layout, detail=detail), tempfile.TemporaryDirectory() as tmp:
                    data = self.dataset(Path(tmp), layout, detail)
                    tokens = self.tokens(data, layout, detail)
                    self.assertEqual(len(training_rows(data, tokens)[0]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            data = self.dataset(Path(tmp))
            self.assertEqual(len(training_rows(data, self.tokens(data, context=None))[0]), 1)

    def test_wrong_native_layout_rejected_even_when_seal_matches_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.dataset(Path(tmp), 'state-first', 'typed')
            tokens = self.tokens(data, 'legacy', 'typed')
            seal = json.loads(tokens.with_suffix('.seal.json').read_text(encoding='utf-8'))
            seal['prompt_layout'] = 'state-first'
            save_status(tokens.with_suffix('.seal.json'), seal)
            with self.assertRaisesRegex(ValueError, 'layout identity'):
                training_rows(data, tokens)

    def test_context_and_empty_tokens_are_rejected(self):
        for context in (0, -1, True, '8192', 2**31):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as tmp:
                data = self.dataset(Path(tmp))
                with self.assertRaisesRegex(ValueError, 'context'):
                    training_rows(data, self.tokens(data, context=context))
        for ids in ([], [1]*9):
            with tempfile.TemporaryDirectory() as tmp:
                data = self.dataset(Path(tmp))
                tokens = self.tokens(data, context=8)
                rows = read_jsonl(tokens)
                rows[0]['input_ids'] = ids
                tokens.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
                seal = json.loads(tokens.with_suffix('.seal.json').read_text(encoding='utf-8'))
                seal['tokens_sha256'] = digest(tokens)
                save_status(tokens.with_suffix('.seal.json'), seal)
                with self.assertRaisesRegex(ValueError, 'token count'):
                    training_rows(data, tokens)

    def test_mlx_checkpoint_mapping_without_mlx_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = json.dumps({name: {} for name in (
                'model.language_model.layers.0.self_attn.q_proj.weight',
                'model.vision_tower.layers.0.self_attn.q_proj.weight')}).encode()
            (root/'model.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header)
            modules = checkpoint_modules(root)
            module = 'language_model.model.layers.0.self_attn.q_proj'
            expected = 'model.language_model.layers.0.self_attn.q_proj'
            self.assertEqual(hf_module(module, modules), expected)
            for names in (set(), modules | {'other.layers.0.self_attn.q_proj'}):
                with self.assertRaisesRegex(ValueError, 'unique'):
                    hf_module(module, names)

    def test_score_invalid_criteria_raise_validation_error(self):
        for criteria in (42, None, 'abc', [], {'one': 'Only'}):
            with self.subTest(criteria=criteria), self.assertRaisesRegex(ValueError, 'Score criteria'):
                decision('q', dict(type='score', instructions='Level?', criteria=criteria))


if __name__ == '__main__':
    unittest.main()
