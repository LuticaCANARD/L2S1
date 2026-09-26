#!/usr/bin/env python3
"""Verify frozen RTX 3060 JevBench records and export aggregate public evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'SmolLM2-135M-Instruct-Q8_0', 'tinyllama-1.1b-chat-v1.0.Q4_K_M'}
MEASURED_DATE = '2026-09-23'
EXPORT_DATE = '2026-09-26'


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def load(path):
    return json.loads(path.read_text())


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10)


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def model_name(filename):
    stem = Path(filename).stem.replace('.Q8_0', '-Q8_0')
    stem = re.sub(r'^gemma-(\d+)-', r'Gemma \1 ', stem)
    stem = re.sub(r'(\d)b(?=-| )', r'\1B', stem)
    stem = re.sub(r'-(Q8_0|Q4_K_M|UD-IQ2_XXS)$', r' \1', stem)
    stem = re.sub(r'-(\d+(?:\.\d+)?B)(?=-| )', r' \1', stem)
    return stem.replace('-it', ' it').replace('-Instruct', ' Instruct').replace('-instruct', ' Instruct').replace('-2512', ' 2512').replace('-E2B', ' E2B').replace('-E4B', ' E4B')


def file_size(source, manifest):
    filename = manifest['model_name']
    planned = next(item for item in load(source / 'plan.json') if Path(item['path']).name == filename)
    if 'size' in planned:
        require(planned['sha256'] == manifest['model_sha256'], f'{filename}: file-size checkpoint identity')
        size = planned['size']
        evidence = {'method': 'Frozen download metadata', 'sourceFile': 'plan.json', 'repository': planned['repo'], 'revision': planned['revision'], 'url': f"https://huggingface.co/{planned['repo']}/resolve/{planned['revision']}/{planned['file']}"}
    else:
        path = ROOT / 'models' / filename
        before = path.stat()
        require(sha256(path) == manifest['model_sha256'], f'{filename}: local checkpoint identity')
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'{filename}: file changed during verification')
        size = after.st_size
        evidence = {'method': 'Local file stat and SHA-256 verification', 'verifiedDate': EXPORT_DATE}
    require(type(size) is int and size > 0, f'{filename}: file size')
    return size, evidence


def audit_row(source, entry):
    identifier = entry['configuration_id']
    directory = source / identifier
    summary, manifest = load(directory / 'summary.json'), load(directory / 'manifest.json')
    require(sha256(directory / 'requests.jsonl') == manifest['requests_sha256'], f'{identifier}: request hash')
    predictions = json_lines(directory / 'predictions.jsonl')
    tasks = {item['id']: item for item in json_lines(directory / 'tasks-with-gold.jsonl')}
    records = json_lines(directory / 'jevbench-records.jsonl')
    selective = json_lines(directory / 'selective-policy.jsonl')
    require(len(tasks) == len(predictions) == len(records) == len(selective) == 231, f'{identifier}: record count')
    require(len({item['id'] for item in predictions}) == 231 and {item['id'] for item in predictions} == set(tasks), f'{identifier}: prediction ids')
    correct = errors = accepted = accepted_correct = abstained = 0
    latency = []
    for prediction in predictions:
        if prediction.get('error'):
            errors += 1
            continue
        task = tasks[prediction['id']]
        require(prediction['response']['policy'] == {'min_candidate_mass': 0.05, 'min_top_probability': 0.8}, f'{identifier}: policy')
        result = prediction['response']['results'][0]
        require(not result['truncated'], f'{identifier}: truncated prediction')
        mapping = {'false': 'no', 'true': 'yes'} if task['question']['type'] == 'noul' else {}
        labels = [mapping.get(score['id'], score['id']) for score in result['scores']]
        probabilities = dict(zip(labels, [score['option_probability'] for score in result['scores']]))
        require(labels == task['labels'], f'{identifier}: labels')
        require(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values()) and abs(sum(probabilities.values()) - 1) <= 0.001, f'{identifier}: probabilities')
        top = min(probabilities, key=lambda label: (-probabilities[label], label))
        correct += top == str(task['expected'])
        value = result['value']
        selected = {False: 'no', True: 'yes', None: None}[value['value']] if value['type'] == 'binary' else value['selected']
        accepted += selected is not None
        accepted_correct += selected is not None and str(selected) == str(task['expected'])
        abstained += selected is None
        latency.append(prediction['elapsed_ms'])
    valid = sum(item['valid'] is True for item in records)
    require(correct == summary['n_correct'] == entry['correct'] == sum(item['correct'] is True for item in records), f'{identifier}: raw correctness')
    require(valid == summary['n_valid'] == summary['n_scorable'] == 231, f'{identifier}: validity')
    selective_counts = {key: sum(item[key] is True for item in selective) for key in ['accepted', 'correct', 'abstained', 'error']}
    expected_selective = {'accepted': accepted, 'correct': accepted_correct, 'abstained': abstained, 'error': errors}
    require(selective_counts == expected_selective, f'{identifier}: selected values vs selective ledger')
    for key, value in expected_selective.items():
        require(summary['selective_policy'][key] == entry['selective'][key] == value, f'{identifier}: selective summary {key}')
    require(accepted + abstained + errors == 231, f'{identifier}: acceptance denominator')
    require(summary['n_attempted'] == summary['n_planned'] == manifest['planned'] == 231, f'{identifier}: planned denominator')
    require(close(summary['accuracy'], correct / 231) and close(entry['accuracy'], correct / 231), f'{identifier}: accuracy denominator')
    require(close(summary['selective_policy']['coverage'], accepted / 231), f'{identifier}: coverage denominator')
    require(close(summary['selective_policy']['accepted_accuracy'], accepted_correct / accepted), f'{identifier}: accepted accuracy denominator')
    p50, p95 = quantile(latency, 0.5), quantile(latency, 0.95)
    require(close(p50, summary['latency']['p50_s'] * 1000) and close(p95, summary['latency']['p95_s'] * 1000), f'{identifier}: latency')
    with (directory / 'gpu-memory.csv').open() as handle:
        memory = [float(row[1].strip()) for row in csv.reader(handle) if row]
    peak = max(memory)
    require(peak == summary['gpu_memory']['peak_board_used_mib'] == entry['peak_gpu_mib'], f'{identifier}: sampled memory maximum')
    backend = predictions[0]['response']['backend']
    require(backend['offload_device'] == 'NVIDIA GeForce RTX 3060' and backend['offload_requested'] is True, f'{identifier}: GPU identity')
    compute = backend['compute']
    require(compute['context'] == 8192 and backend['execution_mode'] == 'fresh', f'{identifier}: runtime configuration')
    overrides = manifest.get('runtime_environment_overrides', {})
    require(set(overrides) <= {'GGML_CUDA_DISABLE_GRAPHS'}, f'{identifier}: unexpected environment override')
    require(identifier != 'gpt-oss-20b-Q4_K_M-no-cuda-graphs' or overrides == {'GGML_CUDA_DISABLE_GRAPHS': '1'}, f'{identifier}: CUDA graph setting')
    counts = {'planned': 231, 'attempted': 231, 'scorable': valid, 'valid': valid, 'rawCorrect': correct, 'accepted': accepted, 'acceptedCorrect': accepted_correct, 'acceptedWrong': accepted - accepted_correct, 'abstained': abstained, 'errors': errors}
    size, size_evidence = file_size(source, manifest)
    row = {'fileSizeBytes': size, 'id': identifier, 'model': model_name(manifest['model_name']), 'modelFile': manifest['model_name'], 'rawAccuracy': correct / 231 * 100, 'coverage': accepted / 231 * 100, 'acceptedAccuracy': accepted_correct / accepted * 100, 'correctAll': accepted_correct / 231 * 100, 'p50Ms': p50, 'p95Ms': p95, 'peakGpuMiB': peak, 'valid': valid, 'errors': errors, 'counts': counts, 'config': {'context': compute['context'], 'batch': compute['batch'], 'microbatch': compute['ubatch'], 'threads': compute['threads'], 'flashAttention': compute['flash_attention'], 'executionMode': backend['execution_mode'], 'promptLayout': backend['prompt_layout'], 'runtimeEnvironmentOverrides': overrides}}
    filenames = ['summary.json', 'manifest.json', 'predictions.jsonl', 'gpu-memory.csv', 'selective-policy.jsonl', 'jevbench-records.jsonl', 'tasks-with-gold.jsonl', 'requests.jsonl']
    provenance = {'fileSizeBytes': size, 'fileSizeEvidence': size_evidence, 'id': identifier, 'modelFile': manifest['model_name'], 'modelSha256': manifest['model_sha256'], 'evaluatorSha256': manifest['evaluator_sha256'], 'requestsSha256': manifest['requests_sha256'], 'adapterSha256': manifest['adapter_sha256'], 'jevbenchRevision': manifest['jevbench_revision'], 'datasetHash': manifest['dataset_hash'], 'datasetFilesSha256': manifest['source_sha256'], 'sourceFilesSha256': {name: sha256(directory / name) for name in filenames}, 'command': [Path(arg).name if arg.startswith('/') else arg for arg in manifest['command']], 'runtimeEnvironmentOverrides': overrides, 'gpuMemorySamples': len(memory), 'gpuMemorySamplingMs': summary['gpu_memory']['sampling_ms']}
    return row, provenance


def document(summary, language, localized=True):
    text = {
        'ko': {'title': 'RTX 3060: JevBench 공개 231문항', 'intro': '2026-09-23 측정 · NVIDIA GeForce RTX 3060 12 GiB · CUDA · 20개 모델 설정 × 231문항 = **4,620개 판단**.', 'counts': '전체 집계: raw 정답 {rawCorrect}/{planned} · 수락 {accepted} · 수락 정답 {acceptedCorrect} · 수락 오답 {acceptedWrong} · 보류 {abstained} · 유효 {valid} · 오류 {errors}.', 'header': '| 모델 | Raw 정답 / 전체 | 수락 / 전체 | 정답 / 수락 | 수락 정답 / 전체 | p50 / p95 ms | GPU MiB | GGUF GiB | 유효 / 오류 |', 'method': '방법과 출처', 'definitions': 'Raw는 정책 적용 전 최고 후보의 정답 수입니다. 수락률은 수락/231, 수락 정확도는 수락 정답/수락, correctAll은 수락 정답/231입니다. p50/p95는 모델 로드와 한 번의 워밍업을 제외한 Rust 직렬 판단 시간입니다. GPU MiB는 200 ms 간격의 전체 장치 사용량 표본 중 최대값입니다. GGUF GiB는 모델 파일 다운로드·디스크 용량입니다. 동결 다운로드 메타데이터 또는 SHA-256 검증한 로컬 파일의 바이트 수가 기준이며, 1 GiB = 1,024 MiB입니다.', 'config': '설정: context 8192, batch/microbatch 256, CPU threads 4, Flash Attention off, fresh, legacy prompt. GPT-OSS 설정은 `GGML_CUDA_DISABLE_GRAPHS=1`입니다.', 'source': 'JevBench revision `{revision}` · easy 48 / original 72 / hard 111. 고정 evaluator SHA-256: `{evaluator}`.', 'links': '집계 JSON · 출처 manifest', 'reproduce': '집계 재생성'},
        'en': {'title': 'RTX 3060: JevBench public 231 items', 'intro': 'Measured 2026-09-23 · NVIDIA GeForce RTX 3060 12 GiB · CUDA · 20 model configurations × 231 items = **4,620 decisions**.', 'counts': 'Totals: raw correct {rawCorrect}/{planned} · accepted {accepted} · accepted correct {acceptedCorrect} · accepted wrong {acceptedWrong} · abstained {abstained} · valid {valid} · errors {errors}.', 'header': '| Model | Raw correct / all | Accepted / all | Correct / accepted | Accepted correct / all | p50 / p95 ms | GPU MiB | GGUF GiB | Valid / errors |', 'method': 'Method and provenance', 'definitions': 'Raw counts the top candidate before policy. Coverage is accepted/231; accepted accuracy is accepted correct/accepted; correctAll is accepted correct/231. p50/p95 measure Rust serial decisions with model loading and one warmup excluded. GPU MiB is the maximum whole-device usage sampled every 200 ms. GGUF GiB measures model download and disk space from frozen download metadata or SHA-256-verified local file bytes; 1 GiB = 1,024 MiB.', 'config': 'Configuration: context 8192, batch/microbatch 256, CPU threads 4, Flash Attention off, fresh, legacy prompt. GPT-OSS uses `GGML_CUDA_DISABLE_GRAPHS=1`.', 'source': 'JevBench revision `{revision}` · easy 48 / original 72 / hard 111. Frozen evaluator SHA-256: `{evaluator}`.', 'links': 'Summary JSON · provenance manifest', 'reproduce': 'Regenerate aggregates'},
        'ja': {'title': 'RTX 3060：JevBench公開231項目', 'intro': '測定日2026-09-23 · NVIDIA GeForce RTX 3060 12 GiB · CUDA · 20モデル設定 × 231項目 = **4,620判断**。', 'counts': '合計：raw正解 {rawCorrect}/{planned} · 受理 {accepted} · 受理正解 {acceptedCorrect} · 受理不正解 {acceptedWrong} · 保留 {abstained} · 有効 {valid} · エラー {errors}。', 'header': '| モデル | Raw正解 / 全体 | 受理 / 全体 | 正解 / 受理 | 受理正解 / 全体 | p50 / p95 ms | GPU MiB | GGUF GiB | 有効 / エラー |', 'method': '方法と出典', 'definitions': 'Rawは方針適用前の最高候補の正解数です。受理率は受理/231、受理精度は受理正解/受理、correctAllは受理正解/231です。p50/p95はモデルロードと1回のウォームアップを除いたRust直列判断時間です。GPU MiBは200 ms間隔で採取した装置全体の使用量の最大値です。GGUF GiBはダウンロード・ディスク容量で、固定ダウンロードメタデータまたはSHA-256検証済みローカルファイルのバイト数が基準です。1 GiB = 1,024 MiB。', 'config': '設定：context 8192、batch/microbatch 256、CPU threads 4、Flash Attention off、fresh、legacy prompt。GPT-OSSは `GGML_CUDA_DISABLE_GRAPHS=1` を使用しています。', 'source': 'JevBench revision `{revision}` · easy 48 / original 72 / hard 111。固定evaluator SHA-256：`{evaluator}`。', 'links': '集計JSON · 出典manifest', 'reproduce': '集計の再生成'}
    }[language]
    prefix = '../' if localized else ''
    nav = f'[English]({prefix}en/RTX3060_BENCHMARK.md) · [한국어]({prefix}ko/RTX3060_BENCHMARK.md) · [日本語]({prefix}ja/RTX3060_BENCHMARK.md)'
    artifact = '../../benchmarks/rtx3060-20260926/' if localized else '../benchmarks/rtx3060-20260926/'
    lines = [f"# {text['title']}", '', nav, '', text['intro'], '', text['counts'].format(**summary['totals']), '', text['header'], '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['rows']:
        c = row['counts']
        lines.append(f"| {row['model']} | {c['rawCorrect']}/231 ({row['rawAccuracy']:.2f}%) | {c['accepted']}/231 ({row['coverage']:.2f}%) | {c['acceptedCorrect']}/{c['accepted']} ({row['acceptedAccuracy']:.2f}%) | {c['acceptedCorrect']}/231 ({row['correctAll']:.2f}%) | {row['p50Ms']:.2f} / {row['p95Ms']:.2f} | {row['peakGpuMiB']:.0f} | {row['fileSizeBytes'] / 1024 ** 3:.2f} | {row['valid']} / {row['errors']} |")
    lines += ['', f"## {text['method']}", '', text['definitions'], '', text['config'], '', text['source'].format(revision=summary['jevbenchRevision'], evaluator=summary['evaluatorSha256']), '', f"[{text['links'].split(' · ')[0]}]({artifact}summary.json) · [{text['links'].split(' · ')[1]}]({artifact}manifest.json)", '', f"## {text['reproduce']}", '', '```sh', 'python3 scripts/export_rtx3060_web.py', 'python3 scripts/export_rtx3060_web.py --check', '```', '']
    return '\n'.join(lines).encode()


def export(source):
    report, environment = load(source / 'REPORT.json'), load(source / 'environment.json')
    selected = [item for item in report['completed'] if item['configuration_id'] not in EXCLUDED]
    require(len(selected) == 20, 'Expected 20 public model configurations')
    rows, provenance = [], []
    for item in selected:
        row, record = audit_row(source, item)
        rows.append(row)
        provenance.append(record)
    require(len({row['id'] for row in rows}) == 20, 'Duplicate model configuration')
    require(len({row['modelFile'] for row in rows}) == 20, 'Duplicate checkpoint')
    counts = {key: sum(row['counts'][key] for row in rows) for key in rows[0]['counts']}
    require(counts['planned'] == counts['valid'] == 4620 and counts['errors'] == 0, 'Aggregate denominator')
    summary = {'schema': 'l2s1.rtx3060.public-aggregates.v1', 'date': MEASURED_DATE, 'exportDate': EXPORT_DATE, 'dataset': 'JevBench v1.1 public 231', 'gpu': 'NVIDIA GeForce RTX 3060', 'vramMiB': 12288, 'driver': '595.71.05', 'context': 8192, 'models': 20, 'decisions': 4620, 'itemsPerModel': 231, 'tierCounts': {'easy': 48, 'original': 72, 'hard': 111}, 'policy': {'minTopProbability': 0.8, 'minCandidateMass': 0.05}, 'latencyScope': 'Rust serial decide_batch, batch size 1; model load and one warmup excluded', 'fileSizeScope': 'GGUF model file bytes; frozen download metadata or SHA-256-verified local file stat', 'gpuMemoryScope': 'Maximum sampled whole-device memory used, 200 ms sampling', 'jevbenchRevision': provenance[0]['jevbenchRevision'], 'evaluatorSha256': provenance[0]['evaluatorSha256'], 'totals': counts, 'rows': rows}
    require(all(item['jevbenchRevision'] == summary['jevbenchRevision'] and item['evaluatorSha256'] == summary['evaluatorSha256'] for item in provenance), 'Frozen source mismatch')
    data = json_bytes(summary)
    manifest = {'schema': 'l2s1.rtx3060.public-provenance.v1', 'measuredDate': MEASURED_DATE, 'exportDate': EXPORT_DATE, 'publicModels': 20, 'publicDecisions': 4620, 'sourceDirectory': source.name, 'sourceFilesSha256': {'REPORT.json': sha256(source / 'REPORT.json'), 'environment.json': sha256(source / 'environment.json'), 'plan.json': sha256(source / 'plan.json')}, 'summarySha256': hashlib.sha256(data).hexdigest(), 'hardware': {'gpu': summary['gpu'], 'vramMiB': summary['vramMiB'], 'driver': summary['driver']}, 'llamaCppRevision': environment['llama_cpp_revision'], 'sourceAndBinarySha256': environment['source_and_binary_sha256'], 'runtimeLibrarySha256': environment['runtime_library_sha256'], 'publicExport': {'contents': 'Aggregated counts, metrics, command flags, and source hashes', 'absolutePaths': 'Command path arguments replaced with basenames', 'sourceStates': 'Omitted', 'reproduce': 'python3 scripts/export_rtx3060_web.py --check'}, 'models': provenance}
    readme = ('# RTX 3060 public benchmark artifacts\n\nMeasured 2026-09-23 · exported 2026-09-26 · JevBench public 231 items · 20 model configurations · 4,620 decisions.\n\n- [summary.json](summary.json): verified aggregate counts, score percentages, latency, sampled whole-device GPU memory, GGUF file bytes, and runtime configuration.\n- [manifest.json](manifest.json): original artifact SHA-256 values, checkpoint hashes, file-size provenance, frozen evaluator identity, and portable command arguments.\n- [English report](../../docs/en/RTX3060_BENCHMARK.md) · [한국어](../../docs/ko/RTX3060_BENCHMARK.md) · [日本語](../../docs/ja/RTX3060_BENCHMARK.md).\n\n```sh\npython3 scripts/export_rtx3060_web.py\npython3 scripts/export_rtx3060_web.py --check\n```\n').encode()
    outputs = {ROOT / 'benchmarks/rtx3060-20260926/summary.json': data, ROOT / 'benchmarks/rtx3060-20260926/manifest.json': json_bytes(manifest), ROOT / 'benchmarks/rtx3060-20260926/README.md': readme, ROOT / 'web/src/lib/rtx3060.json': data, ROOT / 'docs/RTX3060_BENCHMARK.md': document(summary, 'ko', False)}
    for language in ['en', 'ko', 'ja']:
        outputs[ROOT / f'docs/{language}/RTX3060_BENCHMARK.md'] = document(summary, language)
    registry_path = ROOT / 'docs/translations.json'
    registry = load(registry_path)
    entry = {'source': 'docs/RTX3060_BENCHMARK.md', 'path': 'RTX3060_BENCHMARK.md', 'source_sha256': hashlib.sha256(outputs[ROOT / 'docs/RTX3060_BENCHMARK.md']).hexdigest(), 'languages': ['en', 'ko', 'ja']}
    entries = registry['documents']
    existing = next((index for index, item in enumerate(entries) if item['source'] == entry['source']), None)
    if existing is None:
        entries.append(entry)
    else:
        entries[existing] = entry
    outputs[registry_path] = json_bytes(registry)
    return outputs, {'sourceCompletedModels': len(report['completed']), 'sourceCompletedDecisions': sum(item['total'] for item in report['completed']), 'publicModels': 20, 'publicDecisions': 4620, 'counts': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/jevbench-matrix-20260923')
    parser.add_argument('--check', action='store_true', help='Verify counts and compare generated bytes without writing')
    args = parser.parse_args()
    outputs, audit = export(args.source)
    for path, data in outputs.items():
        if args.check:
            require(path.is_file() and path.read_bytes() == data, f'Export differs: {path.relative_to(ROOT)}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(json.dumps({'checked' if args.check else 'exported': len(outputs), **audit}, ensure_ascii=False))


if __name__ == '__main__':
    main()
