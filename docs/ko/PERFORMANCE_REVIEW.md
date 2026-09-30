# 성능 개선 검토 — 2026-09-30

[English](../en/PERFORMANCE_REVIEW.md) · [한국어](PERFORMANCE_REVIEW.md) · [日本語](../ja/PERFORMANCE_REVIEW.md)

main `6b2fcfa` / v0.2.0 기준 코드 검토입니다. 아래 표는 최초 검토 당시의 개선 후보이며, 하단 구현 후속 작업에서 이후 측정 결과를 확인할 수 있습니다. 기존 측정의 리비전과 조건은 별도로 표시합니다. [측정 증거와 재집계](../../benchmarks/decision-performance-20260930/README.md).

## 기준 성능과 실패 유형

| 모델 | 정답 / 231 | 정확도 | p50 / p95 ms | 채택 정답 / 채택 |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 159 | 68.83% | 40.41 / 485.74 | 153 / 213 |
| Gemma 4 12B QAT Q4_0 | 194 | 83.98% | 93.48 / 1359.39 | 181 / 203 |
| Laya 영어 421M | 133 | 57.58% | 45.18 / 71.63 | 133 / 231 (보류 정책 없음) |

JevBench `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`의 공개 231문항(Easy 48, Original 72, Hard 111)을 각각 1회 측정했습니다. 공식 전체 534문항 순위가 아닙니다. Gemma는 L2S1 `74334ec`, RTX 3080 10 GiB, fresh/legacy, context 4096, batch/ubatch 256, threads 4, FlashAttention off, Rust in-process입니다. Laya는 공식 SDK 0.3.21, 체크포인트 `55cf4c4e`, PyTorch 2.8.0 CUDA/BF16 autocast, Python in-process, 기본 512토큰입니다. 로딩과 워밍업 1회는 제외했습니다. Laya는 57문항 입력이 잘렸고 Gemma는 잘림이 없었습니다. 런타임·양자화·문맥 한도·실행 시점이 달라 엔진만의 통제 비교는 아닙니다.

12B도 시간·수치 문제 15개 중 11개, multi-hop 18개 중 6개, long-policy 19개 중 6개를 틀렸습니다. 채택 오답은 22개입니다. 점수 0.99 이상인 오답도 E2B 37개, 12B 10개입니다. 후보 점수를 정답 확률로 해석하면 안 됩니다.

## 실험 우선순위

| 순서 | 실험 | 현재 지원과 작업 범위 | 채택에 필요한 증거 |
| --- | --- | --- | --- |
| 1 | 12B 메모리·prefill 설정 | 기존 FlashAttention off/on, batch/ubatch 128/256, width 1/2/3, `set_parallel_context_dynamic` | 단계별 시간, 실제 GPU 적재, 메모리, p50/p95, 원시·채택 정답 변화 |
| 2 | 고정 질문·공통 입력 재사용 | 기존 상주 fixed-schema, `shared`, `parallel_prefix_session()` | fresh/split-cold/split-warm, 실제 재사용 토큰, 상태 변경 시 수치 일치와 오류 복구 |
| 3 | 어려운 문항의 정확도 | 간결한 판단 기준·수치/시간 전처리, 별도 데이터의 LoRA 또는 output head | 개발 데이터에서 선택 고정 후 별도 그룹 holdout 평가; 12B와 작은 모델 포함 |
| 4 | 채택 정책과 상위 모델 이관 | 기존 family/task calibration과 애플리케이션 라우팅 | 오류율-채택률, 오답 채택, 채택 정답/전체, 실제 전체 지연·비용 |

별도 합성 규칙 실험(고유 판단 36개, 3회 반복)에서 12B의 parallel/state-first p50은 313.23 ms로 fresh/state-first 243.59 ms보다 느렸습니다. E2B는 105.52→72.01 ms였습니다. 모든 설정의 재사용 토큰은 0이었습니다. 12B 병렬 실행 시 데스크톱을 포함한 GPU 전체 메모리 관측치가 10013 MiB에 도달했으므로 메모리 설정을 우선 조사하되, 이것만으로 원인을 확정하지 않습니다. 병렬 실행을 보편적인 고속 기본값으로 선택할 근거는 부족합니다.

검토 코드: `src/llama.rs`(compute·동적 context), `src/llama/shared_decision.rs`(고정 질문 세션), `src/llama/parallel_session.rs`(호출 간 공통 prefix), `src/llama/interchange.rs`(설정·보정 식별자). batch 경계보다 짧은 prefix는 재사용이 0일 수 있습니다. token 정렬은 점수를 바꿀 수 있어 별도 검증해야 합니다.

## 단순 2단계 모델 사용의 한계

기존 E2B 정책(top probability >=0.8, candidate mass >=0.05, 동률 없음)이 보류한 경우에만 12B로 넘기고 12B 정책을 유지해 저장된 응답을 재집계했습니다.

| 정책 | 12B 이관 | 원시 정답 | 채택 정답 / 채택 | 보류 |
| --- | ---: | ---: | ---: | ---: |
| E2B → 12B | 18 / 231 | 166 / 231 (71.86%) | 159 / 222 (71.62%) | 9 |
| 12B 단독 | 231 / 231 | 194 / 231 (83.98%) | 181 / 203 (89.16%) | 28 |

실제 2단계 실행이 아닌 과거 예측의 산술 재집계입니다. 모델 동시 적재, 로딩·라우팅 비용과 속도 개선을 측정하지 않았습니다. 확신도가 높은 E2B 오답이 그대로 남습니다. 새 라우터를 만들려면 별도 데이터로 학습·선택하고 실제 메모리와 통신 비용을 검증해야 합니다.

## 실험 조건과 경계

모델·데이터·소스·런타임·어댑터 해시를 고정합니다. 설정 하나씩 비교한 후 조합하고, baseline/candidate 순서를 교차하며 동일 워밍업으로 최소 3회 시간을 측정합니다. 반복 문항을 독립 정확도 표본으로 세지 않습니다. 입력 길이, 오류·잘림, 원시 정확도, 채택률, 채택 정확도, 채택 정답/전체, 오답 채택, p50/p95, 처리량, 메모리를 함께 기록합니다. 같은 모델의 확률·점수 변화도 확인합니다. 실험 전에 지연 목표와 허용할 정확도·채택률 변화를 정하고 두 조건을 통과해야 채택합니다. 이미 살펴본 공개 JevBench는 회귀 데이터이며 새 선택의 미사용 holdout이 아닙니다.

첫 실험은 기존 설정으로 가능한 1·2번입니다. KV 양자화는 현재 L2S1 노출 기능이 아니어서 bridge·설정 식별자·수치 검증 작업이 필요합니다. speculative decoding은 다중 토큰 생성용이므로 직접 한 단계 채점에서는 우선순위가 낮습니다. 임계값 보정은 채택 정책을 바꾸며 원시 오답을 자동으로 고치지 않습니다.

Laya 후속 비교에서는 긴 문맥 체크포인트나 windowing을 명시적으로 선택하고 모델·문맥·집계·속도 변화를 기록해야 합니다. 잘린 토큰을 복원하면 정확도도 복원된다고 가정하지 않습니다. [Laya 공식 문서](https://huggingface.co/convaiinnovations/laya)는 체크포인트별 한도를 설명합니다. [llama.cpp 성능 가이드](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md)는 실제 GPU 적재와 스레드 수 확인을 권하며, 그 생성 속도를 L2S1 지연 예측으로 사용하지 않습니다.

## 구현 후속 작업

1~3번을 실행할 수 있는 [튜닝 도구·상주 세션·구조화된 사실 전처리](../DECISION_PERFORMANCE.md)를 추가했습니다. 실제 측정 결과와 적용 범위는 연결된 실험 보고서를 참고하세요.
