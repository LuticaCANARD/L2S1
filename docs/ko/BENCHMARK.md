<a id="local-model-comparison-benchmark"></a>
# 국내 모델 비교 벤치마크

[English](../en/BENCHMARK.md) · [한국어](BENCHMARK.md) · [日本語](../ja/BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

`decision-rules-v1` 벤치마크는 **12 요청의 기존 GGUF 체크포인트를 36 라벨이 붙은 판단**와 비교합니다. 규칙 준수 정확성, 판단 보류, 반복 출력 일관성 및 추론 대기 시간을 측정합니다. 모델을 다운로드하거나 배포하지 않습니다.

기본값은 레거시 v1 프롬프트입니다. 선택적 v2 상태 우선 프롬프트 및 선택적 접두사 재사용은 [SEMIF_ALGORITHM.md](SEMIF_ALGORITHM.md)에 문서화되어 있습니다. 비교를 위해 별도의 출력 디렉터리를 사용하여 v2의 경우 `--prompt-layout state-first`를 전달한 다음 `--execution-mode fresh`(기본값) 또는 `--execution-mode prefix-reuse`를 Rust 실행기에 전달합니다. 보고서는 요청된 모드, 논리적 입력 토큰, 재사용된 접두사 토큰 및 실제 평가된 토큰을 기록합니다. 동일한 프롬프트 버전, 모델, 장치 및 배치 설정을 비교합니다. 프롬프트를 변경하면 캐시 재사용과 관계없이 정답률이 변경될 수 있습니다.

`tests/fixtures/decision_benchmark.json`의 영어 픽스처는 두 가지 도메인을 다룹니다.

- 창고: 0, 6, 7, 24, 25 및 48 시간에 보관 선택, 저온 유통 요건 및 파견 우선순위.
- 액세스 제어: 역할 간 매핑, 부울 편집 권한, 0, 1, 2, 3 및 4 시도 시 시도 실패 검토 우선순위.

모든 요청에는 하나의 선택 사항, 하나의 바이너리, 하나의 서열 판단이 있습니다. 선택 옵션 순서는 다양합니다. 서열 값은 API의 요구에 따라 정렬된 상태로 유지됩니다. 일부 경우에는 관련 없는 메모가 포함되어 있습니다. 예상되는 답변은 A/B/C 토큰 위치와 무관한 의미론적 옵션 ID입니다. 일반 Rust 테스트는 시나리오 규칙의 모든 레이블을 독립적으로 다시 계산합니다.

이는 일반적인 추론, 코딩, 다국어, 안전 또는 운영 환경-정답률 평가가 아닌 소규모 합성 규칙 준수 벤치마크입니다. 이러한 레이블을 조정하지 말고 결과 숫자를 보류된 정답률로 표시하지 마십시오. `tests/native.rs`의 실제 모델 일관성 테스트는 별도로 유지됩니다.

<a id="recorded-windows-rtx-5090-results"></a>
## Windows / RTX 5090 측정 결과

사용자가 제공한 과거 요약 자료입니다. 기록 시각은 **2026-09-26 16:21:17 UTC**(한국 시각 2026-09-27 01:21:17)이며, 사용자가 GPU를 **NVIDIA GeForce RTX 5090**으로 확인했습니다. Windows x86_64 환경이고 CPU 기종·메모리 측정값은 없습니다. GPU 기종은 사용자 확인 정보이며 장치 로그는 제공되지 않았습니다.

| 모델 | 장치 | 상태 | 수락 비율 | 수락 정답률 | 전체 대비 수락 정답 | 원시 top-1 | 요청당 p50 ms | 요청당 p95 ms | 판단/초 | 수락 정답/초 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smollm2 | cpu | ok | 13.9% | 20.0% | 2.8% | 38.9% | 7156.9 | 7805.6 | 0.41 | 0.01 |
| smollm2 | cuda | ok | 11.1% | 25.0% | 2.8% | 38.9% | 41.6 | 44.9 | 71.61 | 1.99 |
| qwen3 | cpu | ok | 77.8% | 35.7% | 27.8% | 33.3% | 22681.9 | 25154.9 | 0.13 | 0.04 |
| qwen3 | cuda | ok | 83.3% | 33.3% | 27.8% | 41.7% | 44.7 | 47.0 | 66.42 | 18.45 |
| gemma3 | cpu | ok | 91.7% | 51.5% | 47.2% | 50.0% | 33004.2 | 36213.8 | 0.09 | 0.04 |
| gemma3 | cuda | ok | 91.7% | 57.6% | 52.8% | 61.1% | 67.7 | 70.7 | 44.37 | 23.42 |
| tinyllama | cpu | ok | 0.0% | n/a | 0.0% | 38.9% | 9513.6 | 10583.5 | 0.31 | 0.00 |
| tinyllama | cuda | ok | 0.0% | n/a | 0.0% | 38.9% | 34.2 | 36.3 | 87.59 | 0.00 |
| gemma4 | cpu | timeout | — | — | — | — | — | — | — | — |
| gemma4 | cuda | ok | 97.2% | 94.3% | 91.7% | 94.4% | 67.6 | 80.7 | 43.90 | 40.24 |

완료 실행마다 독립 요청 12개·정답 판단 36개를 3회 반복했습니다. 요청 시간 표본은 36개, 측정 판단은 108개지만 반복을 독립 정답률 표본으로 세지 않습니다. 로딩과 워밍업은 제외했습니다. 설정: `legacy`, `fresh`, 컨텍스트 2048, 배치 256, 스레드 4개, 임계값 0.8 / 0.05, 제한 시간 1800초.

Gemma4 CUDA는 이 픽스처에서 가장 좋은 후보입니다. 동일한 매 회차에 수락 정답 33개·수락 오답 2개·보류 1개이며, 수락 정답률 33/35(94.3%), 전체 대비 수락 정답 33/36(91.7%), 원시 top-1 정답률 34/36(94.4%), 수락 비율 35/36(97.2%)입니다. p50 67.6 ms는 판단 3개가 포함된 요청의 시간입니다. 판단/초는 보류도 포함합니다. TinyLlama는 전부 보류해 수락 정답률이 `n/a`이며, Gemma4 CPU는 시간 초과로 완료 지표가 없습니다.

표의 10개 행이 제공된 요약과 일치합니다. 집계 검산은 요청 시간 324개, 개수·비율, 최근접 순위 백분위, 처리량을 확인합니다. 픽스처 SHA256은 LF를 Windows CRLF로 바꾸면 일치합니다. 완료 실행마다 반복 변경은 0/72지만 Qwen3·Gemma3의 원시 top-1과 SmolLM2의 수락 비율은 CPU/CUDA 사이에 다릅니다. 참조된 개별 JSON·로그가 없어 사례별 예측·점수, 실제 장치 배치, 체크포인트·실행 파일 해시, 네이티브 일관성은 독립 검증하지 못했습니다. 추론을 다시 실행하지 않았습니다. 이 결과는 합성 픽스처에 한정되며 모델의 일반적인 품질을 입증하지 않습니다.

[요약 JSON](../../benchmarks/decision-rules-windows-20260926/summary.json) · [집계 검산](../../benchmarks/decision-rules-windows-20260926/audit.json) · [출처와 하드웨어](../../benchmarks/decision-rules-windows-20260926/provenance.json)

```sh
node web/scripts/verify-decision-rules.mjs
```


<a id="recorded-apple-m5-max-results"></a>
## macOS / Apple M5 Max 측정 결과

**2026-09-27 04:33:25 UTC**(13:33:25 KST)에 **Apple M5 Max**(통합 메모리 128 GB, macOS 27.0)에서 측정했습니다. AC 전원에 연결한 고성능 모드였습니다. 설정은 위의 Windows 측정과 같습니다: `legacy`, `fresh`, 컨텍스트 2048, 배치 256, 스레드 4, 임계값 0.8 / 0.05, 측정 3회, 워밍업 1회, 타임아웃 1800초. Metal 실행에는 이 저장소 리비전의 `--device cpu metal`을 사용했습니다. Gemma 4 26B-A4B Q4_K_M은 사용자 지정 manifest로 여섯 번째 체크포인트로 실행했습니다.

| Model | Device | Status | Coverage | Accepted accuracy | Correct / all | Raw top-1 | p50 ms/request | p95 ms/request | Decisions/s | Correct accepted/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smollm2 | cpu | ok | 8.3% | 33.3% | 2.8% | 38.9% | 153.3 | 167.7 | 19.58 | 0.54 |
| smollm2 | metal | ok | 13.9% | 40.0% | 5.6% | 38.9% | 21.8 | 23.2 | 137.29 | 7.63 |
| qwen3 | cpu | ok | 83.3% | 33.3% | 27.8% | 41.7% | 474.3 | 522.1 | 6.26 | 1.74 |
| qwen3 | metal | ok | 83.3% | 33.3% | 27.8% | 36.1% | 42.9 | 43.7 | 69.99 | 19.44 |
| gemma3 | cpu | ok | 91.7% | 54.5% | 50.0% | 50.0% | 610.4 | 679.0 | 4.83 | 2.42 |
| gemma3 | metal | ok | 88.9% | 53.1% | 47.2% | 50.0% | 55.6 | 60.7 | 52.82 | 24.94 |
| tinyllama | cpu | ok | 0.0% | n/a | 0.0% | 38.9% | 1399.0 | 1556.7 | 2.13 | 0.00 |
| tinyllama | metal | ok | 0.0% | n/a | 0.0% | 38.9% | 59.6 | 60.7 | 50.23 | 0.00 |
| gemma4 | cpu | ok | 94.4% | 100.0% | 94.4% | 97.2% | 1784.4 | 2377.2 | 1.65 | 1.56 |
| gemma4 | metal | ok | 94.4% | 97.1% | 91.7% | 94.4% | 203.8 | 246.4 | 14.68 | 13.46 |
| gemma4-26b-a4b | cpu | ok | 100.0% | 100.0% | 100.0% | 100.0% | 5837.7 | 6489.0 | 0.52 | 0.52 |
| gemma4-26b-a4b | metal | ok | 100.0% | 100.0% | 100.0% | 100.0% | 506.0 | 647.7 | 5.75 | 5.75 |

Gemma 4 26B-A4B는 두 장치 모두 모든 회차에서 36개 판단을 전부 맞혔습니다(Metal p50 506.0 ms). Gemma 4 E2B는 두 장치 모두 회차마다 36개 중 34개를 수락했습니다. CPU는 34개를 모두 맞혔고, Metal은 33개를 맞히고 1개를 틀렸습니다. 모든 실행에서 반복 비교 72회 중 변경은 0회였지만, 일부 체크포인트는 CPU와 Metal 사이에 약간의 차이가 있습니다. 모든 Metal 리포트에는 `backend.offload_device: "Apple M5 Max"`가 기록되어 있습니다. 실행별 리포트와 로그는 로컬에만 보관하고 공개하지 않았습니다. 이 결과는 합성 fixture에 대한 근거이며, 일반적인 모델 품질을 나타내지 않습니다.

[결과와 한계](../../benchmarks/decision-rules-macos-m5max-20260927/README.md) · [요약 JSON](../../benchmarks/decision-rules-macos-m5max-20260927/summary.json) · [출처와 하드웨어](../../benchmarks/decision-rules-macos-m5max-20260927/provenance.json)


<a id="run-the-five-model-matrix"></a>
## 5개 모델 매트릭스 실행

전제 조건: 이미 캐시된 Rust 종속성, CMake, C++17 컴파일러 및 로컬에서 획득한 체크포인트. 번들 sys 종속성은 고정된 llama.cpp 소스를 빌드합니다. CUDA 실행에는 추가로 CUDA 툴킷이 필요합니다. 모델 경로는 `tests/fixtures/benchmark_models.json`에 나열되어 있습니다. 러너는 Hugging Face 클라이언트, 계정 또는 네트워크 요청을 사용하지 않습니다. 모델을 구입하기 전에 별도의 라이선스를 검토하세요.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools benchmark-models \
  --device cpu cuda \
  --iterations 3 \
  --warmup 1
```

Runner는 요청 시 `--locked --offline` 및 CUDA 기능을 사용하여 릴리스 테스트 실행 파일을 한 번 빌드하고 각 모델을 해시하며 한 번에 하나의 모델/장치를 실행합니다. CPU가 기본 장치입니다. CUDA는 명시적이며 자동으로 CPU로 대체될 수 없습니다. macOS에서는 `--device cpu metal`을 지정하면 `llama-metal` 기능으로 한 번 빌드합니다. CUDA와 Metal은 한 번의 실행에서 함께 지정할 수 없습니다. GPU 배치 여부는 각 Metal 리포트의 `backend.offload_device`에서 확인합니다. 네이티브 백엔드는 전체 실행에 대해 하나의 로드된 모델을 재사용합니다. Rust 프로세스 시작, 화물 편집, 적재 및 워밍업은 정상 상태 타이밍에서 제외됩니다.

하위 집합을 선택하거나 설정을 변경합니다.

```sh
target/release/l2s1-tools benchmark-models \
  --model gemma4 --model qwen3 \
  --device cuda \
  --iterations 10 --warmup 2 \
  --context 2048 --batch 256 --threads 4 \
  --min-top-probability 0.8 --min-candidate-mass 0.05 \
  --timeout 1800 \
  --output results/benchmark/my-comparison
```

`--iterations`는 모든 12 요청에 대해 전체 측정 통과를 의미합니다. `--warmup`는 별도의 시간이 설정된 첫 번째 요청에 추가로 완전히 제외된 패스를 의미합니다. 케이스 순서는 측정된 패스 간에 결정론적으로 회전합니다. 반복은 타이밍과 일관성을 특성화하는 데 도움이 됩니다. 독립적인 정답률 예제의 수는 증가하지 않습니다.

오래된 보고서를 현재 결과로 착각할 수 없도록 출력 디렉터리는 새로운 것이어야 합니다. 기본값은 무시된 `results/benchmark/` 아래의 타임스탬프 디렉터리입니다. `--timeout`는 워밍업을 포함하여 각 모델/장치에 별도로 적용됩니다. 누락된 파일, 네이티브 오류 및 시간 초과는 실패한 행으로 계속 표시됩니다. 실행기는 다른 모델을 계속 사용하고 실행이 실패하면 0이 아닌 종료 코드를 반환합니다. 낮음 정답률은 테스트 실패가 아닌 측정으로 보고됩니다.

사용자 정의 체크포인트의 경우 `--manifest /path/to/models.json`를 전달합니다.

```json
{
  "models": [
    {"id": "my-model", "path": "/absolute/path/to/chat-model.gguf"}
  ]
}
```

상대 체크포인트 경로는 사용자 정의 매니페스트를 포함하여 저장소 루트에 대해 확인됩니다. ID는 문자, 숫자, 하이픈, 밑줄만 포함하는 고유한 소문자 이름이어야 합니다. 벤치마크는 백엔드의 자동 프롬프트 프로필을 사용하므로 모델은 해당 템플릿을 통해 동일한 의미 작업을 받습니다. 템플릿/토큰 수와 모델 크기를 비교하세요.

<a id="reports-and-metric-definitions"></a>
## 보고서 및 측정항목 정의

각 디렉터리에는 `summary.md`, `summary.json`, 빌드 로그, 모델/장치별 JSON 및 네이티브 로그가 포함됩니다. 체크포인트 SHA256, 픽스처 SHA256, 실행 가능한 SHA256, 사용 가능한 저장소/llama.cpp 개정, 장치, 양자화 설명, 설정, 정책, 개별 레이블, 예측, 확률 및 대기 시간 샘플이 유지됩니다. 원시 로컬 보고서에는 로컬 모델 경로가 포함될 수 있습니다. 게시하기 전에 검토하세요.

| 지표 | 정의 |
| --- | --- |
| 수락률 | 받아들인 판단 / 측정된 모든 판단 |
| 판단 보류 환율 | 판단 보류된 판단 / 신중한 모든 판단 |
| 수락된 판단의 정답률 | 승인된 판단/승인된 판단을 수정합니다. `null` / `n/a`(아무 것도 허용되지 않는 경우) |
| 수락된 정답 / 전체 | 승인된 판단/모든 판단을 수정합니다. 판단 보류은 올바른 것으로 간주되지 않습니다 |
| 원시 상단-1 정답률 | 정책 판단 보류 이전에 레이블 대비 가장 높은 점수를 받은 의미론적 옵션; 동점이 잘못되었습니다 |
| 올바른 허용됨 | 올바른 승인된 판단/측정된 추론 시간(초) |
| 판단/초 | 판단 보류을 포함한 모든 완료된 판단/측정된 추론 시간(초) |
| p50/p95 요청 | 전체 `decide` 호출의 가장 가까운 순위 백분위수(각 호출에는 세 가지 순차적 판단이 포함됨) |
| 입력 토큰/초 | 처리된 입력 토큰 / 추론 시간(초). 생성 토큰/초와 구분함 |
| 반복 일관성 | 허용된 출력 및 원시 상위 1 변경 사항과 동일한 사례/판단에 대한 첫 번째 패스 비교 |

품질 개수 및 분수도 도메인 및 판단 종류별로 표시됩니다. 전체 샘플은 사례별 검사를 허용합니다. 반복 출력은 점수가 변경되는 동안 동일한 선택된 옵션을 유지할 수 있습니다. 이 보고서는 네이티브 제품군의 확률/일괄 검사를 대체하지 않습니다.

로드 및 첫 번째 요청은 별도로 보고됩니다. 타이밍에는 신속한 준비, 사전 작성, logits 전송 및 채점이 포함되지만 JSON 보고서 직렬화는 제외됩니다. OS 페이지 캐시는 플러시되지 않습니다. 로드 시간은 콜드 디스크 측정이 아닙니다. 전체 컨텍스트 용량, 프로세스 RSS, 최대 VRAM 및 동시 제공 처리량은 측정되지 않습니다. 하드웨어, 양자화, 컨텍스트, 배치, 스레드 수 및 정책은 의미 있는 비교를 위해 일관성을 유지해야 합니다. 임의의 하드웨어 속도 임계값은 없습니다.

<a id="test-the-benchmark-without-models"></a>
## 모델 없이 벤치마크 테스트

```sh
cargo test --locked --offline --test benchmark
cargo test --locked --offline -p l2s1-tools
```

이는 동점 및 완전 판단 보류 출력을 포함하여 픽스처 레이블 및 측정 단위 분모를 검증하고 매니페스트 검증 및 오류 보존 보고서를 제공합니다. 그들은 실제 추론 정확성을 확립하지 않습니다.

실제 모델 벤치마크는 `tests/benchmark.rs`에서 무시된 Rust 테스트입니다. 직접 실행할 수도 있습니다.

```sh
SKID_MODEL=/absolute/path/to/model.gguf \
SKID_CUDA=1 \
SKID_BENCH_ITERATIONS=3 \
SKID_BENCH_WARMUP=1 \
SKID_BENCH_OUTPUT=results/benchmark/single-model.json \
  cargo test --release --locked --offline --features llama --test benchmark \
  native::model_decision_benchmark -- --exact --ignored --nocapture
```

직접 실행에서는 `SKID_DEVICE`(`cpu`, `cuda`, `metal`. 해당 기능으로 빌드해야 합니다), `SKID_CONTEXT`, `SKID_BATCH`, `SKID_THREADS`, `SKID_MIN_TOP_PROBABILITY` 및 `SKID_MIN_CANDIDATE_MASS`를 허용합니다. Rust 실행기는 체크포인트 해시와 비교 테이블을 추가합니다. 이전 `tests/performance.rs`는 단일 창고 예의 반복 측정에 계속 사용할 수 있습니다.

## 실행·캐시 설정 비교

`benchmark-models`는 macOS에서 `--device metal` (`llama-metal` 빌드)과 `--execution-mode fresh|prefix-reuse|parallel|state-restore`를 지원한다. CUDA와 Metal은 별도 빌드가 필요하다. `--parallel-width`는 1–32이며 기본값은 3이다. 늘리면 KV 용량과 메모리 부담이 커진다. Parallel과 state-restore는 명시적으로 비교할 실험 설정이며 fresh보다 느릴 수도 있다.

`--evidence-transfer full|compact`의 기본값은 full이며, 텍스트 compact는 fresh 또는 prefix-reuse에서 사용한다. 전체 어휘 확률 정규화는 유지하면서 후보 증거를 작게 반환한다. `--preparation-cache-bytes 8388608`은 크기가 제한된 준비 캐시를 켠다(기본 0으로 비활성, 캐시별 최대 128개 항목). 보고서는 측정 전후 카운터를 기록하며 워밍업에서 캐시가 채워질 수 있다. 준비 캐시 적중은 렌더링·토큰화를 줄이지 transformer 추론을 줄이지 않으며, 동일 fixture 반복은 새로운 state에서의 이득을 과장할 수 있다. 실제 `reused_prefix_tokens`를 별도로 확인한다.

직접 실행하는 ignored test의 대응 환경변수는 `SKID_DEVICE`, `SKID_EXECUTION_MODE`, `SKID_PARALLEL_WIDTH`, `SKID_EVIDENCE_TRANSFER`, `SKID_PREPARATION_CACHE_BYTES`다. `SKID_DEVICE`가 없으면 기존 `SKID_CUDA`를 사용한다. 일반 요청은 prefix-reuse에서도 격리된다. state가 바뀌는 요청 간 재사용은 별도 [고정 스키마 세션 실험](SEMIF_ALGORITHM.md#fixed-schema-sessions)으로 측정한다. 그룹 전체 시간은 기존 판단 3개 요청의 p50과 다른 지표다.
