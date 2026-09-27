"""Exercise the installed candidate through the already installed Python SDK."""
import asyncio
import json
from pathlib import Path
import sys
from l2s1 import L2S1, LoadOptions, DecisionRequest, __version__

async def main():
    root = Path(sys.argv[1])
    model = '/home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf'
    package = root / 'installed/kernels/node_modules/@l2s1/runtime-linux-arm64'
    fixture = json.loads((root/'source/tests/fixtures/decision_benchmark.json').read_text())['cases'][0]
    request = DecisionRequest.model_validate(fixture['request'])
    output = {'sdk_version': __version__, 'runtime_package': str(package), 'runs': []}
    for mode in ['fresh', None]:
        engine = await L2S1.load(LoadOptions(model=model, runtime_dir=package, context=2048,
                                            batch=256, ubatch=256, threads=4, execution_mode=mode))
        async with engine:
            cold, warm = await engine.decide(request), await engine.decide(request)
            assert [r.evidence for r in cold.results] == [r.evidence for r in warm.results]
            reused = sum(r.usage.reused_prefix_tokens or 0 for r in warm.results)
            if mode == 'fresh':
                assert reused == 0 and warm.backend.details['execution_mode'] == 'fresh'
            else:
                assert reused > 0 and warm.backend.details['prefix_plan'] == 'fixed-schema-split-v1'
            output['runs'].append({'execution_override': mode, 'warm_reused_tokens': reused,
                                   'cold': cold.model_dump(mode='json'), 'warm': warm.model_dump(mode='json')})
    (root/'sdk-smoke.json').write_text(json.dumps(output, indent=2)+'\n')
    print('Installed SDK fresh override and automatic resident reuse passed')

asyncio.run(main())
