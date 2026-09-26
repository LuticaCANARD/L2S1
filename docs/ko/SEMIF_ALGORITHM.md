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
