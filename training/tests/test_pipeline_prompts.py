"""Check complete runner command wiring without substituting for native/model tests."""
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

from l2s1_training import run_jev_lora as runner
from l2s1_training.jev_model_profiles import read_profile
from l2s1_training.prepare_jev_data import prepare_custom


class PipelinePromptTests(unittest.TestCase):
    def test_export_base_and_adapter_use_the_same_deployment_prompt(self):
        for layout in ('legacy', 'state-first'):
            for detail in ('minimal', 'typed', 'typed-examples'):
                for execution in ('fresh', 'parallel'):
                    with self.subTest(layout=layout, detail=detail, execution=execution), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        paths = []
                        for split in ('train', 'development', 'test'):
                            path = root/f'{split}.jsonl'
                            path.write_text(json.dumps(dict(id=split, workflow='fixture', state={'id':split}, shared={'rule':'yes'},
                                questions={'q':dict(type='noul',instructions='Yes?')},
                                gold={'q':dict(type='noul',label='true',probabilities={'false':0,'true':1})}))+'\n')
                            paths.append(path)
                        data = root/'data'
                        prepare_custom(*paths, data, layout=layout, detail=detail)
                        profile = read_profile('gemma4')
                        checkpoint = root/profile['revision']
                        checkpoint.mkdir()
                        (checkpoint/'config.json').write_text('{}')
                        (root/profile['gguf']).write_bytes(b'fixture')
                        converter = root/'converter.py'
                        converter.write_text('')
                        commands = {}
                        def stage(name, command, output, *unused):
                            commands[name] = list(map(str,command))
                            if name == 'evaluator-check':
                                (output/'evaluator-check.log').write_text('--parallel-prefix-alignment')
                            if name == 'export':
                                (output/'train-tokens.jsonl').write_text('')
                            if name == 'convert':
                                (output/'adapter.gguf').write_bytes(b'fixture')
                            if name in ('base-report','adapter-report'):
                                report = output/name
                                report.mkdir()
                                overall = {k:None for k in ('raw_accuracy','coverage','accepted_accuracy','correct_all','soft_kl','soft_brier','score_mae')}
                                (report/'summary.json').write_text(json.dumps(dict(overall=overall, latency_ms={}, by_type={}, failures=[])))
                        argv = ['run', '--models','gemma4','--data',str(data),'--checkpoint-root',str(root),
                                '--gguf-root',str(root),'--converter',str(converter),'--exporter',sys.executable,
                                '--evaluator',sys.executable,'--output',str(root/'run'),'--eval-execution',execution]
                        previous = signal.getsignal(signal.SIGTERM)
                        try:
                            with mock.patch.object(sys,'argv',argv), mock.patch.object(runner,'execute_stage',stage):
                                runner.main('run')
                        finally:
                            signal.signal(signal.SIGTERM,previous)
                        for name in ('export','base','adapter'):
                            cmd=commands[name]
                            for flag,expected in (('--prompt-layout',layout),('--prompt-detail',detail),('--context','8192')):
                                self.assertEqual(cmd[cmd.index(flag)+1],expected)
                        for name in ('base','adapter'):
                            cmd=commands[name]
                            self.assertEqual(cmd[cmd.index('--execution-mode')+1],execution)
                            if execution == 'parallel':
                                self.assertEqual(cmd[cmd.index('--parallel-prefix-alignment')+1],
                                                 'token' if layout == 'state-first' else 'batch')
                            else:
                                self.assertNotIn('--parallel-prefix-alignment',cmd)
                        self.assertIn('--lora',commands['adapter'])
                        self.assertNotIn('--lora',commands['base'])
