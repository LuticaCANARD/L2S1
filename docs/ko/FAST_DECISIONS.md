# Prefix 재사용, ONNX 판정, 검증된 모델 전환

[English](../en/FAST_DECISIONS.md) · [한국어](FAST_DECISIONS.md) · [日本語](../ja/FAST_DECISIONS.md)

호환되는 텍스트 상주 서버(`--listen` / `--stdio`, SDK `load` 포함)는 실행 모드를 생략하면 크기가 제한된 고정 스키마 prefix KV 재사용을 기본으로 사용합니다. 단발 CLI와 저수준 Rust 백엔드는 `fresh`를 유지합니다. 명시한 실행 모드, 비전/projector, calibration/output head, compact evidence는 기존 경로를 유지하며 recurrent/hybrid 모델은 시작 메시지와 함께 fresh로 전환합니다. `--execution-mode fresh`로 끄거나 `--fixed-schema`로 지원을 필수 조건으로 지정할 수 있습니다. 네이티브 병렬 배치는 여전히 `--execution-mode parallel`이 필요합니다. 분할 계획에 따라 점수가 달라질 수 있으므로 capabilities와 `usage.reused_prefix_tokens`를 확인하세요.

이 기본값은 v0.1.3 이후 소스 변경입니다. 배포된 v0.1.3에서는 `fixedSchema: true` / `fixed_schema=True`를 명시해야 하며, 다음 릴리스 전에는 새로 빌드한 바이너리를 사용하세요.

## 고정 스키마 세션

`llama`, `llama-cuda`, `llama-metal` 중 환경에 맞는 기능으로 빌드합니다.

```sh
l2s1 --model model.gguf --listen 127.0.0.1:8080
```

`FixedSchemaBackend`는 cold와 warm 호출을 같은 prefix/suffix 경계로 나눕니다. prefill batch 크기를 줄일 필요가 없습니다. 전체 프롬프트를 토큰화한 후 실제 토큰을 대조하므로 BPE 경계를 보존합니다. 활성 KV context는 하나이며, prefix snapshot은 최대 8개·256MiB입니다. 스키마 토큰 준비 캐시는 별도로 64개·4MiB로 제한합니다. 캐시가 차면 제거한 prefix를 다시 계산하며 답 자체는 저장하지 않습니다. `clear()`로 KV와 snapshot을 비웁니다. 신뢰 영역마다 전용 서버/백엔드를 사용하세요.

분할 방식 자체가 기존 fresh 점수를 바꿀 수 있습니다. **기존 fresh 대 split cold**, **split cold 대 split warm**을 각각 비교해야 합니다. 후자가 캐시 동등성 검증입니다. 명시적 고정 스키마 요청은 recurrent/hybrid 모델, 기존 calibration, output head를 거부하며 full evidence가 필요합니다. 26개 초과 후보와 이미지는 KV를 비운 기존 경로를 사용합니다. `shared_decision()`의 기존 batch 정렬 방식은 유지됩니다.

TypeScript는 `L2S1.load({ model, fixedSchema: true, binaryPath: "./target/release/l2s1" })`, Python은 `LoadOptions(model=..., fixed_schema=True, binary_path="./target/release/l2s1")`를 사용합니다. 실행 모드를 생략하면 prefix-reuse를 선택하며 stdio와 HTTP를 지원합니다. `prepare()`만으로 KV가 생기지는 않습니다. capabilities의 `prefix_reuse`, 결과의 `usage.reused_prefix_tokens`를 확인하세요.

`fixedSchema: true` / `fixed_schema=True`는 이 계획을 필수로 요구하며 호환되지 않으면 오류를 냅니다. 옵션 생략은 자동 선택이고, `false` / `False`는 실행 모드가 따로 없을 때 fresh를 선택합니다. 기존 cascade 정책은 실행 계획까지 일치해야 하므로 이전 명시 설정을 유지하거나 정책을 다시 적합·검증하세요.

## Laya ONNX

```sh
cargo build --release --features llama,onnx --bin l2s1 --bin l2s1-onnx
python3 scripts/fetch_laya_onnx.py --output models/laya-en
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so
l2s1-onnx --model-dir models/laya-en --min-top-probability 0 --listen 127.0.0.1:8081
```

ONNX Runtime 1.24 이상을 별도 설치합니다. CUDA는 `onnx-cuda` 빌드 기능, 호환 CUDA 라이브러리와 `--cuda`가 필요하며 CPU로 조용히 전환하지 않습니다. CUDA용 FP16 그래프는 별도 디렉터리에 `--precision fp16`으로 받습니다. GGUF CUDA는 별도로 `llama-cuda`가 필요합니다.

다운로더는 그래프·원본 가중치·토크나이저·보정 파일의 해시를 검증합니다. 모델은 Apache-2.0이며 출처는 `third_party/ollaya`에 기록합니다. Laya marker head를 지원하며 임의 ONNX 모델을 받지는 않습니다. state 토큰화만 공유하고 질문별 encoder 입력은 독립적입니다. 배치는 padding 포함 32,768토큰·128개 판정으로 제한합니다. 구조화된 JSON은 키를 정렬하므로 Ollaya와 비교할 때 키 순서도 맞춰야 합니다. 긴 state·규칙·후보를 조용히 자르지 않고 거부합니다.

`discriminative` 증거에는 후보 점수·확률, 스키마 식별자, 적용한 temperature, 추정값이 있습니다. vocabulary token ID와 candidate mass는 없습니다. GGUF용 요청 정책과 `target_error_rate`는 거부합니다. 별도 최대 확률 문턱도 오류율 보증은 아닙니다. 모델과 ONNX Runtime 라이브러리가 serving identity에 포함됩니다. 배포할 하드웨어와 정밀도에서 검증하세요.

## 빠른 모델과 느린 모델 연결

빠른 서버와 사용할 느린 서버를 먼저 실행합니다. 서로 분리한 calibration·validation JSONL은 `{id, request, expected: {decision_id: option_id}}` 형식입니다. binary 정답은 문자열 `false`/`true`, ordinal 정답은 level ID입니다.

```sh
python3 scripts/calibrate_cascade.py --fast-url http://127.0.0.1:8081 \
  --slow-url http://127.0.0.1:8080 --calibration calibration.jsonl \
  --validation heldout.jsonl --max-error 0.05 --output cascade.json
```

ID와 동일 state 중복을 거부하고 calibration에서만 문턱을 선택합니다. 고정된 문턱이 validation에서 실패하면 해당 규칙을 제외합니다. 유사 중복·학습 오염·운영 분포 변화까지 증명하는 검사는 아닙니다. 오류 한도는 held-out 표본의 경험적 수치이며 통계적 보증이 아닙니다. 표본 수와 coverage를 함께 확인하세요.

느린 서버 설정을 유지하고 `--fast-model-dir models/laya-en --cascade-policy cascade.json`을 추가해 재시작합니다. CUDA 빠른 모델은 `--fast-cuda`도 추가합니다. 양쪽 백엔드 기능으로 빌드해야 합니다. 정책이 없으면 느린 모델만 사용하고, 모델/runtime 식별자가 다르면 시작을 거부합니다. 검증된 동일 스키마에서만 빠른 답을 수락합니다. 낮은 점수·거부·지원하지 않는 입력·빠른 모델 오류는 느린 모델로 넘기고, 이미지는 vision을 지원하는 느린 모델로 바로 보냅니다. 느린 모델 오류는 호출자에게 전달합니다. 결과마다 선택한 모델·전환 이유가 기록되며 판정 순서를 보존합니다. cascade에는 요청별 수락 정책을 덮어쓸 수 없습니다.

HTTP는 native batch가 활성화된 백엔드에서만 text/direct 요청을 묶습니다. 최대 8개 요청·128개 판정·44MiB·1ms 수집입니다. prefix 세션과 cascade는 현재 순차 실행합니다. 대기 요청은 180초 후 만료되지만 이미 실행 중인 native 호출은 중단할 수 없습니다. 처리량·단일 요청 지연·p95는 별도로 평가해야 합니다.

## 검증

`benchmark_fixed_schema`는 원래 여러 판정이 들어 있는 규칙 요청에서 기존 fresh, split cold, split reuse를 비교합니다. `benchmark_onnx`는 준비·추론·점수화 지연과 입력/출력 근거를 기록합니다. 지역적 진단이며 일반 정확도나 다른 하드웨어의 Ollaya 대표 속도를 보장하지 않습니다.


## 로컬 측정 (2026-09-27)

WSL2 CPU의 Qwen3 0.6B Q8_0에서 decision-rules 36개 판단 전체가 fresh 29.132초, split cold 31.062초, split reuse 14.962초였습니다. fresh 대비 1.95배 개선입니다. batch 256을 유지하며 4,983개 중 2,799개 토큰을 재사용했고 선택·확률 변화는 없었습니다. 워밍업 후 한 번 측정한 전체 묶음 시간이며 요청 p50이나 M5 Max 결과가 아닙니다.

RTX 3080의 영어 Laya fp16 경로는 요청 p50 19.179ms, 같은 모델의 Ollaya는 19.567ms였고 p95는 각각 22.165/23.338ms였습니다. 질문 5개씩 포함한 합성 티켓 입력 20개를 워밍업 20회 후 두 번 반복했습니다. CPU와 CUDA 모두 토큰 ID·marker 위치·logit이 정확히 일치했고 확률 차이는 2.23e-16 미만이었습니다. L2S1 시간에는 확률 계산이 포함되지만 비교 runner에는 포함되지 않습니다. 로컬 동등성 검증이며 일반 정확도나 보편적인 속도 우위를 뜻하지 않습니다.

실제 HTTP에서 fast 채택, 스키마·입력 fallback, Python SDK 수신을 검증했고 동시 요청 16개 중 최대 6개가 한 배치로 실행됐습니다. calibration 검증에는 시험용 데이터를 사용했으므로 운영 품질 근거로 삼을 수 없습니다. [측정 요약](../../benchmarks/fast-decisions-20260927/summary.json)과 [출처](../../benchmarks/fast-decisions-20260927/provenance.json)를 참고하십시오. 원시 기록과 재현 명령은 저장소의 `benchmarks/fast-decisions-20260927`에 있습니다.


## SIMD 점수 후처리

전체 vocabulary의 logit 검증·최댓값 탐색에 AVX2/NEON 런타임 선택과 scalar fallback을 적용했습니다. 지수함수와 순차 f64 합산 순서는 유지합니다. 로컬 AVX2에서 3.2만~26.2만 vocabulary의 탐색은 약 10배, 전체 정규화는 1.17~1.26배 빨라졌습니다. **모델 추론 전체의 가속 배율은 아닙니다.** 실제 SmolLM2 규칙 판단 36개의 결과는 적용 전 바이너리와 정확히 같았습니다. [SIMD 측정](../../benchmarks/evidence-simd-20260927/summary.json)과 [native 동등성](../../benchmarks/evidence-simd-20260927/native-parity.json)을 참고하세요. NEON 속도는 아직 측정하지 않았습니다.
