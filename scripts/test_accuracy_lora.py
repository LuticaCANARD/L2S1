import copy
import json
import hashlib
from unittest import mock
from pathlib import Path
import tempfile
import unittest

import train_accuracy_lora as trainer


class AccuracyDistillationTests(unittest.TestCase):
    def fixture(self, root):
        manifest = dict(schema_version=1, splits={})
        labels = {}
        for split in ('train', 'dev', 'calibration', 'test'):
            cid = split+'-case'
            labels[cid] = {'zone': 'cold'}
            row = dict(id=cid, group=split+'-group', request=dict(state={'temperature': 3}, decisions=[dict(
                id='zone', instruction='Choose storage zone.', kind=dict(type='choice', options=[
                    dict(id='warm', criterion='Temperature above 10.'),
                    dict(id='cold', criterion='Temperature at most 10.')]))]), expected=labels[cid])
            variants = {}
            for variant in ('natural', 'symbolic'):
                name = f'{split}-{variant}.jsonl'
                requests = f'{split}-{variant}-requests.jsonl'
                (root/name).write_text(json.dumps(row)+'\n')
                (root/requests).write_text(json.dumps({k: row[k] for k in ('id', 'group', 'request')})+'\n')
                variants[variant] = dict(path=name, sha256=trainer.digest(root/name),
                                         requests_path=requests, requests_sha256=trainer.digest(root/requests))
            manifest['splits'][split] = dict(ids=[cid], groups=[row['group']], count=1, variants=variants)
        (root/'labels.json').write_text(json.dumps(labels))
        manifest['labels'] = dict(path='labels.json', sha256=trainer.digest(root/'labels.json'))
        (root/'manifest.json').write_text(json.dumps(manifest))
        identity = dict(weights_sha256='a'*64, template_sha256='b'*64, runtime_build_sha256='c'*64)
        tokens = [dict(id='train-case', decision_id='zone', input_ids=[10, 11, 12+i],
                       candidate_ids=[31, 30] if i == 0 else [30, 31],
                       option_ids=['warm', 'cold'], candidate_codes=['B', 'A'] if i == 0 else ['A', 'B'],
                       model_identity=dict(identity, prompt_version=('base-v1' if i == 0 else
                                           f'base-v1/detail-Minimal-v1/rotation-{i}')),
                       code_rotation=i) for i in range(2)]
        (root/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in tokens))
        seal = dict(schema_version=1, split='train', variant='natural', detail='minimal',
                    manifest_sha256=trainer.digest(root/'manifest.json'),
                    token_sha256=trainer.digest(root/'tokens.jsonl'),
                    hf_model=trainer.MODEL, hf_revision=trainer.REVISION)
        (root/'seal.json').write_text(json.dumps(seal))
        return manifest, tokens, seal

    def load(self, root):
        return trainer.load_data(root, 'natural', root/'tokens.jsonl', root/'seal.json', 'minimal')

    def test_teacher_tokenization_accepts_mapping_and_preserves_attention_mask(self):
        class Tensor:
            shape = (1, 3)
            device = None
            def to(self, device):
                self.device = device
                return self
        class Tokenizer:
            def __init__(self, output):
                self.output = output
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return self.output
        ids, mask = Tensor(), Tensor()
        tokenizer = Tokenizer({'input_ids': ids, 'attention_mask': mask})
        actual = trainer.teacher_inputs(tokenizer, [{'role': 'user', 'content': 'Test'}], 'cpu')
        self.assertEqual(actual, {'input_ids': ids, 'attention_mask': mask})
        self.assertEqual(ids.device, 'cpu')
        self.assertEqual(mask.device, 'cpu')
        self.assertTrue(tokenizer.kwargs['return_dict'])
        self.assertTrue(tokenizer.kwargs['enable_thinking'])
        self.assertEqual(trainer.teacher_inputs(Tokenizer(ids), [], 'cpu'), {'input_ids': ids})
        with self.assertRaisesRegex(ValueError, 'no input IDs'):
            trainer.teacher_inputs(Tokenizer({'attention_mask': mask}), [], 'cpu')

    def test_streaming_digest_does_not_read_entire_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.safetensors'
            content = b'0123456789' * 350000
            path.write_bytes(content)
            with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file read')):
                self.assertEqual(trainer.digest(path), hashlib.sha256(content).hexdigest())

    def test_identity_normalization_only_removes_verified_rotation(self):
        for detail, variant in [('minimal', 'Minimal'), ('typed', 'Typed'), ('typed_examples', 'TypedExamples')]:
            normalized = []
            for rotation in range(3):
                version = ('base-v1' if detail == 'minimal' and rotation == 0 else
                           f'base-v1/detail-{variant}-v1/rotation-{rotation}')
                identity = dict(prompt_version=version, weights_sha256='a'*64)
                normalized.append(trainer.normalized_token_identity(identity, rotation, detail))
            self.assertEqual(normalized[0], normalized[1])
            self.assertEqual(normalized[1], normalized[2])
            self.assertIn(f'detail-{variant}', normalized[0]['prompt_version'])
        invalid = [
            ('base-v1/detail-Typed-v1/rotation-1', 1, 'minimal'),
            ('base-v1/detail-Minimal-v1/rotation-1', 2, 'minimal'),
            ('base-v1/detail-Minimal-v1/rotation-01', 1, 'minimal'),
            ('base-v1/detail-Minimal-v1/rotation-1/extra', 0, 'minimal'),
            ('base-v1/detail-Unknown-v1/rotation-1', 0, 'minimal'),
            ('base-v1', 0, 'typed'),
        ]
        for version, rotation, detail in invalid:
            with self.assertRaises(ValueError):
                trainer.normalized_token_identity(dict(prompt_version=version), rotation, detail)

    def test_rotation_identity_accepts_pair_but_rejects_mixed_weights_or_details(self):
        for mutation in ('weights', 'detail', 'base'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, rows, seal = self.fixture(root)
                self.load(root)  # Minimal zero legacy identity and explicit rotation one coexist.
                if mutation == 'weights':
                    rows[1]['model_identity']['weights_sha256'] = 'd'*64
                elif mutation == 'base':
                    rows[1]['model_identity']['prompt_version'] = 'other-base/detail-Minimal-v1/rotation-1'
                else:
                    rows[1]['model_identity']['prompt_version'] = 'base-v1/detail-Typed-v1/rotation-1'
                (root/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                seal['token_sha256'] = trainer.digest(root/'tokens.jsonl')
                (root/'seal.json').write_text(json.dumps(seal))
                with self.assertRaisesRegex(ValueError, 'Mixed production|detail/rotation identity'):
                    self.load(root)

    def test_only_channel_final_is_accepted_and_budget_exhaustion_is_explicit(self):
        self.assertEqual(trainer.parse_teacher('<|channel>thought\nA looks good', ['A', 'B'], True),
                         ('thought_budget_exhausted', None))
        self.assertEqual(trainer.parse_teacher('A', ['A', 'B'], False), ('missing_final', None))
        self.assertEqual(trainer.parse_teacher('<channel|>A', ['A', 'B'], False),
                         ('missing_thought_channel', None))
        self.assertEqual(trainer.parse_teacher('<|channel>thought\nB first, but A.\n<channel|>A<turn|>',
                                               ['A', 'B'], False), ('parsed', 'A'))
        for final in ('A or B', 'Answer: A', 'A\nmore explanation', 'Z', ''):
            self.assertEqual(trainer.parse_teacher('<|channel>thought\nB\n<channel|>'+final,
                                                   ['A', 'B'], False)[0], 'invalid_final')

    def test_sealed_rotations_preserve_semantics_and_teacher_has_no_gold_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            cases, rows, provenance = self.load(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(provenance['variant'], 'natural')
            messages = trainer.teacher_messages(cases['train-case'], rows[0])
            self.assertNotIn('expected', messages[0]['content'])
            self.assertIn('"code": "A", "criterion": "Temperature at most 10.", "id": "cold"',
                          messages[0]['content'])
            class Tokenizer:
                def __len__(self):
                    return 100
                def decode(self, ids, **kwargs):
                    return {30: 'A', 31: 'B'}[ids[0]]
            trainer.validate_tokenizer(Tokenizer(), rows)
            wrong = copy.deepcopy(rows)
            wrong[0]['candidate_ids'].reverse()
            with self.assertRaisesRegex(ValueError, 'answer-token mapping'):
                trainer.validate_tokenizer(Tokenizer(), wrong)

    def test_hash_mismatch_overlap_and_heldout_tokens_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, rows, seal = self.fixture(root)
            (root/'tokens.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'token hash mismatch'):
                self.load(root)
            (root/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest['splits']['dev']['groups'] = ['train-group']
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                self.load(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, rows, seal = self.fixture(root)
            rows[0]['id'] = 'test-case'
            (root/'tokens.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            seal['token_sha256'] = trainer.digest(root/'tokens.jsonl')
            (root/'seal.json').write_text(json.dumps(seal))
            with self.assertRaisesRegex(ValueError, 'Non-training ID'):
                self.load(root)

    def test_teacher_verified_semantic_target_applies_to_every_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            cases, rows, provenance = self.load(root)
            provenance['checkpoint_sha256'] = 'checkpoint'
            teacher_dir = root/'teacher'
            teacher_dir.mkdir()
            record = dict(id='train-case', decision_id='zone', group='train-group', status='correct',
                          code='A', predicted='cold', candidate_codes=['B', 'A'], option_ids=['warm', 'cold'],
                          generated_text='<|channel>thought\n3 is at most 10.\n<channel|>A<turn|>',
                          reached_budget=False)
            (teacher_dir/'teacher.jsonl').write_text(json.dumps(record)+'\n')
            report = dict(provenance, complete=True, enable_thinking=True, statuses={'correct': 1},
                          teacher_sha256=trainer.digest(teacher_dir/'teacher.jsonl'))
            (teacher_dir/'teacher-report.json').write_text(json.dumps(report))
            accepted, summary = trainer.accepted_training(cases, rows, provenance, teacher_dir)
            self.assertEqual(len(accepted), 2)
            self.assertEqual([r['teacher_target'] for r in accepted], ['cold', 'cold'])
            self.assertEqual(summary['accepted_logical_cases'], 1)
            self.assertEqual([r['candidate_codes'][r['option_ids'].index(r['teacher_target'])]
                              for r in accepted], ['A', 'B'])
            record['status'] = 'wrong'
            (teacher_dir/'teacher.jsonl').write_text(json.dumps(record)+'\n')
            report['teacher_sha256'] = trainer.digest(teacher_dir/'teacher.jsonl')
            report['statuses'] = {'wrong': 1}
            (teacher_dir/'teacher-report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'No verified teacher'):
                trainer.accepted_training(cases, rows, provenance, teacher_dir)

    def test_relabeling_wrong_teacher_as_correct_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            cases, rows, provenance = self.load(root)
            teacher_dir = root/'teacher'
            teacher_dir.mkdir()
            record = dict(id='train-case', decision_id='zone', group='train-group', status='correct',
                          code='B', predicted='warm', candidate_codes=['B', 'A'], option_ids=['warm', 'cold'],
                          generated_text='<|channel>thought\nWarm.\n<channel|>B<turn|>', reached_budget=False)
            (teacher_dir/'teacher.jsonl').write_text(json.dumps(record)+'\n')
            report = dict(provenance, complete=True, enable_thinking=True, statuses={'correct': 1},
                          teacher_sha256=trainer.digest(teacher_dir/'teacher.jsonl'))
            (teacher_dir/'teacher-report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'independent train label'):
                trainer.accepted_training(cases, rows, provenance, teacher_dir)

    def test_option_contract_supports_binary_and_arbitrary_ordinal_width(self):
        binary = {'kind': dict(type='binary', false_label='No', true_label='Yes')}
        self.assertEqual([o['id'] for o in trainer.option_specs(binary)], ['false', 'true'])
        ordinal = {'kind': dict(type='ordinal', levels=[dict(id=f'l{i}', criterion=str(i), value=i)
                                                       for i in range(7)])}
        self.assertEqual(len(trainer.option_specs(ordinal)), 7)


if __name__ == '__main__':
    unittest.main()
