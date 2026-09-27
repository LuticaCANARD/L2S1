<a id="state-first-decisions-and-request-local-prefix-reuse"></a>
# 상태 우선 판단 및 요청-로컬 접두사 재사용

[English](../en/SEMIF_ALGORITHM.md) · [한국어](SEMIF_ALGORITHM.md) · [日本語](../ja/SEMIF_ALGORITHM.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

접두사 재사용 모드는 SemIf의 증거 우선 프롬프트 및 직렬 접두사 캐시 아이디어를 기존 Rust/libllama 판단 백엔드에 적용합니다. 채점 계약은 전체 어휘 후보 확률 질량, 입력된 결과 및 명시적 판단 보류를 포함하는 단일 토큰 조건부 소프트맥스로 유지됩니다.

<a id="reference-and-scope"></a>
## 참조 및 범위

2026-09-21에서 [SemIf](https://github.com/TheoLeeCJ/SemIf) 소스를 검토했습니다.

- [`core.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/core.py): 판단 기준 및 옵션 이전의 증거. Git Blob `6e93b16dcd4ab56c7a859c9046c48c6731d7180c`를 검토했습니다.
- [`serial.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/serial.py): 정확한 접두사 검증, 독립적인 접미사 평가 및 후보 대량 진단. Git Blob `015023b7e9d616e6ace0a600548e6fe33ad7f98f`를 검토했습니다.
- [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py): 병렬 공유 상태 분기 레이아웃. Git Blob `6e6870305d1b7693cead70b30637c0fec2684354`를 검토했습니다.

SemIf는 MIT 라이센스, 저작권 2026 TheoLeeCJ입니다. 이 업데이트는 이러한 알고리즘 아이디어를 독립적으로 구현한 Rust/C++입니다. SemIf 코드, 모델 가중치 또는 데이터 세트를 공급하지 않습니다. 게시된 성능과 정답률은 이 프로젝트의 측정값이 아닙니다. 이 구현은 접미사를 순차적으로 평가합니다. 병렬 실행은 [PARALLEL_EXECUTION.md](PARALLEL_EXECUTION.md)에 별도로 설명되어 있습니다.

<a id="algorithm"></a>
## 알고리즘

1. 옵트인 `state-first`의 경우 입력된 페이로드를 고정 순서 `state`, `instruction`, `options`로 직렬화합니다. Rust 구조체는 serde_json 맵 기능이 다른 경우에도 순서를 보장합니다. 구조화된 JSON를 보존하고 특수 토큰 구문 분석을 비활성화한 상태에서 모든 요청 데이터를 토큰화합니다.
2. 기존 모델별 채팅 템플릿과 어시스턴트 경계를 적용합니다. Qwen3 non-thinking 및 GPT-OSS Harmony 최종 프리필은 계속 지원됩니다. 각 A~Z 응답 코드를 안정적인 단일 토큰 연속으로 검증합니다.
3. `fresh`의 경우 메모리를 지우고 전체 프롬프트를 평가합니다. 옵트인 `prefix-reuse`의 경우 전체 토큰화된 프롬프트를 이전 판단 토큰과 비교하세요. 상태 해시 또는 원시 텍스트 길이에서 동등성을 추론하지 마십시오.
4. 정확한 공통 접두사를 완전한 원래 사전 채우기 배치로 반올림합니다. 동일한 프롬프트라도 평가를 위해 하나 이상의 최종 토큰을 보관하세요. 이는 새로운 추론과 동일한 접미사 배치 경계를 유지합니다. `--batch`보다 짧은 공유 접두사는 재사용되지 않습니다.
5. `llama_memory_seq_rm`로 이전 접미사를 제거하고 원래 절대 위치에서 나머지 토큰을 평가합니다. 반복/하이브리드 모델 또는 접미사 제거 실패의 경우 메모리를 지우고 새로 평가합니다. 모든 요청 경계에서 그리고 실패 후에 네이티브 캐시 메타데이터와 KV 메모리를 모두 지웁니다.
6. 최종 전체 어휘 logits를 읽고 변경되지 않은 채점 및 판단 보류 정책을 적용합니다. `input_tokens`는 논리적 토큰을 계산합니다. `reused_prefix_tokens`는 실제 재사용된 토큰을 계산합니다. 그 차이는 평가된 숫자입니다. `backend.execution_mode`는 재사용이 새로운 상태로 되돌아가는 경우를 포함하여 요청된 모드를 기록합니다.

기본값은 `--prompt-layout legacy --execution-mode fresh`로 유지되어 원래 프롬프트 동작을 유지합니다. 버전이 지정된 v2 프롬프트를 사용하려면 `--prompt-layout state-first`를 독립적으로 선택하세요. 두 레이아웃 모두 두 실행 모드 중 하나를 지원하지만 레거시 프롬프트는 재사용 가능성이 낮은 증거를 제공합니다. 이전 v1 벤치마크 결과는 v2 결과로 다시 라벨링되어서는 안 됩니다. 증거를 이동하면 캐시 재사용과 관계없이 모델 예측이 변경됩니다. 점수는 보정되지 않은 상태로 유지되며 속도 향상은 더 나은 의미 체계 정답률을 설정하지 않습니다.

<a id="reproduce"></a>
## 재현

README에 설명된 고정된 sys 종속성 빌드를 사용하세요.

```sh
cargo test --locked --offline
cargo test --release --locked --offline --features llama

SKID_MODEL=models/Qwen3-0.6B-Q8_0.gguf SKID_CUDA=0 \
  SKID_REUSE_OUTPUT=/tmp/qwen3-reuse.json \
  cargo test --release --locked --offline --features llama \
  --test prefix_reuse -- --ignored --nocapture
```

CUDA에는 `--features llama-cuda`와 `SKID_CUDA=1`을 사용하며 GPU 접근이 필요합니다. 작은 prefill 배치를 확인하려면 `SKID_BATCH=32`를 설정합니다(기본값 256). 선택 실행하는 네이티브 테스트는 라벨이 있는 합성 요청 12개(판단 36개), 긴 상태의 판단 6개 요청 하나, 긴 상태에서 프롬프트가 같은 판단 3개 요청 하나를 사용합니다. 확률·후보 질량·원시 top-1·정책이 채택한 선택을 비교하며 오류 및 요청 격리를 확인합니다. 기존 확률·질량 허용 오차 0.02는 회귀 검사이며 정답률 보장이 아닙니다. top-1 또는 채택한 선택이 바뀌어도 테스트는 실패합니다. 모드별 준비 요청 하나를 실행하고 측정 순서는 모드 간에 번갈아 배치합니다. 각 요청은 모드별로 한 번만 측정하므로 시간은 안정적인 백분위수가 아닌 실행 확인용 측정입니다.

레이블이 지정된 정답률, 수락률, 대기 시간 분포 및 반복 실행의 경우:

```sh
target/release/l2s1-tools benchmark-models --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode fresh --output results/v2-fresh
target/release/l2s1-tools benchmark-models --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode prefix-reuse --output results/v2-reuse
```

별도의 출력 디렉터리를 사용하고 모델 해시, 프롬프트 버전, 장치, 배치, 컨텍스트 및 정책을 보존합니다. 실행기는 재사용 및 평가된 토큰 수를 기록합니다. 논리적 입력 토큰 처리량만으로는 저장된 실제 컴퓨팅을 측정할 수 없습니다.

<a id="fixed-schema-sessions"></a>
## 가변 상태 사이의 고정 스키마 세션

`LlamaBackend::shared_decision(decision)`은 `shared_state(state)`의 반대 용도다. 질문·선택지 스키마를 고정하고 바뀌는 상태들을 평가한다. 실제 디코더 KV를 호출 사이에서 유지하지만 일반 `decide()`의 요청 경계에서는 계속 KV를 지운다. Python/TypeScript SDK의 `PreparedDecision`은 정의 스냅샷이며 이 네이티브 세션을 활성화하지 않는다.

```rust
backend.set_execution_mode(l2s1::ExecutionMode::PrefixReuse);
backend.set_prompt_layout(l2s1::PromptLayout::Legacy);
let mut session = backend.shared_decision(decision)?;
let first = session.decide(first_state)?;
let next = session.decide(next_state)?;
println!("reused: {}", next.results[0].reused_prefix_tokens);
```

기본 JSON 맵 정렬에서 Legacy/minimal은 질문과 선택지를 가변 상태 앞에 둔다. 의존성에서 `serde_json/preserve_order`를 켜면 legacy 순서가 달라질 수 있으므로 실제 재사용 토큰을 확인한다. State-first는 고정 상태에 질문이 바뀌는 반대 작업에 맞는다. 세션은 백엔드를 독점 차용하므로 모델·정책·스키마가 고정된다. 생성·오류·종료 시 KV를 비우고 이전 상태의 뒷부분을 교체하며 답을 캐시하지 않는다. Recurrent/hybrid 모델과 출력 헤드는 거부한다. 하나의 시퀀스만 유지하며 자동 다중 스키마 HTTP 캐시는 아니다.

배치 정렬은 유지한다. 공통 프리픽스가 180토큰이면 배치 256에서 0토큰, 128에서 128토큰, 64에서도 128토큰을 재사용한다. 작은 배치는 prefill 처리량을 낮출 수 있다. 배치별 fresh/reuse 비교와 가장 빠른 fresh 기준을 모두 비교해야 하며 재사용 카운터만으로 순속도 향상을 판단하지 않는다.

`examples/benchmark_schema_reuse.rs`는 `decision-rules-v1`의 라벨 있는 36개 결정을 정확히 같은 질문·선택지 순서·스키마에 따라 10개 그룹으로 재생한다. 두 경로 모두 호출당 결정 하나를 처리하며 순서를 교대해 fresh와 세션을 비교하고 확률·후보 질량·argmax·채택 답·보류 사유를 확인한다. 이 그룹별 처리량은 원래 결정 세 개짜리 요청의 지연과 별도 측정이다. 정답·프롬프트·선택지 순서는 변경하지 않는다.

```sh
mkdir -p results
cargo build --release --locked --features llama --example benchmark_schema_reuse
cargo run --release --locked --features llama --example benchmark_schema_reuse -- \
  --model models/Qwen3-0.6B-Q8_0.gguf --device cpu --batch 64 --rounds 3 \
  --output results/schema-reuse-b64.json
```

배치 256에서도 새 출력 경로로 반복한다. macOS는 `llama-metal`과 `--device metal`, CUDA는 `llama-cuda`와 `--device cuda`를 사용한다. 세션 생성·종료, 토큰 준비, 채점을 시간에 포함한다. 로딩·워밍업은 제외하고 준비 캐시는 끈다. 결과 일치 검증은 독립 데이터 정확도의 증명이 아니다.

### CPU 실측 검증 (2026-09-27)

Linux/WSL2, i9-9900K로 표시되는 논리 CPU 4개, 스레드 4개에서 Qwen3-0.6B-Q8_0을 실행했다. Context 2048, full evidence, 준비 캐시 비활성이다. 각 셀은 36개 판단 전체를 그룹으로 처리한 세 번의 중앙값이다. 각 경로에 별도 워밍업을 수행하고 쌍 내 fresh/reuse 순서를 교대했다. **전체 패스의 초 단위 시간**이며 기존 요청 p50이 아니다.

| Batch | Fresh | Shared decision | Reused / input tokens |
| --- | ---: | ---: | ---: |
| 64 | 33.311 s | 22.068 s | 1,984 / 4,983 |
| 256 | 31.691 s | 31.589 s | 0 / 4,983 |

배치 64 재사용은 같은 배치의 fresh보다 1.51배, **측정한 가장 빠른 fresh 설정(배치 256)보다 1.44배 빠르다**. 논리 입력 길이는 120–165토큰이므로 전체 배치 단위 규칙상 배치 256에서는 재사용이 없다. 네이티브 호출이 시간을 지배한다. 배치 64 첫 fresh 패스의 전체 33.311초 중 네이티브는 32.806초, 준비는 0.039초다.

동일 배치의 모든 쌍, 반복 패스, 배치 간 비교에서 확률·후보 질량·raw top-1·수락 답·기권 차이는 0이었다. Raw top-1은 13/36, 수락은 30개, 그중 정답은 10개로 그대로다. 빨라졌다고 정답률이 개선된 것은 아니다. 실제 네이티브 추론으로 세션 오류 후 복구와 요청·세션 격리도 통과했다. M5 Max/Metal, jv.py, 동시 HTTP, 메모리 사용량이나 일반 정답률을 측정한 결과는 아니다. [축약 증거](../../benchmarks/schema-reuse-20260927/summary.json)와 [출처](../../benchmarks/schema-reuse-20260927/provenance.json)에 실행 시간, 첫 패스 점수, 해시와 범위를 남겼다. 전체 응답·로그는 로컬 ignored 파일이다.

다른 작업은 `--input requests.jsonl --context 16384`로 실행한다. 입력은 `evaluate_jsonl`과 같은 정답 없는 `{ "id": "unique-case", "request": { "state": ..., "decisions": [...] } }` 행이다. 그룹화에서도 질문·선택지 순서를 그대로 유지한다. 외부 입력 보고서는 fresh/세션의 결과 일치를 검사하며, 정답률은 분리된 정답 데이터로 별도 채점한다.

[RTX 3080 JevBench 231문항 검증](JEVBENCH.md#rtx3080-rerun-20260927)에서는 batch 64에서 토큰 6.48–7.20%만 재사용했습니다. 같은 batch의 출력은 보존했지만 Qwen3 0.6B·Gemma 4 E2B 모두 batch 256 fresh보다 느렸고, batch 변경 자체는 일부 top-1을 바꿨습니다. CPU 규칙 픽스처의 이득은 워크로드에 한정되며 최적화할 때 배치 처리량과 정답 변화도 함께 확인해야 합니다.

큰 MoE 모델을 CUDA에서 비교할 때는 `--cpu-moe-layers N`, `--gpu-layers N`도 지정할 수 있습니다. 두 경로에 동일한 배치를 적용하고 모델 식별 정보에 기록합니다. 속도를 비교할 때 컨텍스트·batch·배치를 맞춰야 하며 CPU expert offload는 전체 GPU 추론과 다릅니다. [26B 추가 검증](JEVBENCH.md#gemma26-rtx3080-20260927)에서 이 경로를 사용합니다.
