<a id="typed-decisions-recorded-full-test-measurements"></a>
# Typed-decisions: 전체 테스트 측정 기록

[English](../en/TYPED_DECISIONS_BENCHMARK.md) · [한국어](TYPED_DECISIONS_BENCHMARK.md) · [日本語](../ja/TYPED_DECISIONS_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

2026년 9월 26일, 두 체크포인트가 각각 [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90/all)의 **테스트 400케이스 / 판단 2,000개** 전체를 완료했습니다. 이는 교사 모델이 라벨을 붙인 합성 워크플로 판단이며 사람이 라벨을 붙인 실제 배포 결과가 아닙니다. 두 실행 모두 추론 오류나 잘린 출력이 없었습니다.

점수는 판단 보류 전 hard argmax 일치율입니다. JevBench 공개 부분집합 점수와 별개입니다. 테스트 분할은 학습, 보정 학습, 프롬프트 선택에 사용하지 않았습니다.

<a id="results"></a>
## 결과

| 체크포인트 | 정책 적용 전 top-1 | 수락률 | 정답 / 수락 | 수락된 정답 / 전체 | 수락된 오답 | 케이스 p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 1086/2000 (54.30%) | 92.75% | 55.69% | 1033/2000 (51.65%) | 822 | 322.64 / 370.98 |
| Qwen3 0.6B Q8_0 | 625/2000 (31.25%) | 55.60% | 34.35% | 382/2000 (19.10%) | 730 | 173.46 / 204.22 |

한 케이스에는 판단 다섯 개가 있습니다. 지연 시간은 전체 케이스의 완료 시간이며 **질문당 지연 시간이나 분할 평균 비용이 아닙니다.** 로딩과 측정하지 않은 워밍업 케이스 하나는 제외합니다. 기본 정책은 최상위 후보 확률 >= 0.8, 후보 확률 질량 >= 0.05, 동점 없음입니다. 점수는 보정되지 않았습니다.

측정은 상당한 과신을 보여줍니다. 모델 점수 기준을 높이는 것만으로 요청한 실제 오류율을 보장할 수 없습니다. 두 모델 모두 수락된 판단의 정답률이 정책 적용 전 정답률보다 약간 높지만 수락한 부분집합에도 Gemma 822개, Qwen 730개의 오답이 있습니다.

<a id="by-decision-kind"></a>
## 판단 종류별

| 체크포인트 | 타입 | 정책 적용 전 정답 / 계획 | 수락률 | 정답 / 수락 |
| --- | --- | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | binary | 304/600 (50.67%) | 99.83% | 50.75% |
| Gemma 4 E2B Q8_0 | choice | 337/600 (56.17%) | 94.83% | 57.64% |
| Gemma 4 E2B Q8_0 | ordinal | 445/800 (55.62%) | 85.88% | 58.37% |
| Qwen3 0.6B Q8_0 | binary | 264/600 (44.00%) | 62.33% | 42.25% |
| Qwen3 0.6B Q8_0 | choice | 161/600 (26.83%) | 61.50% | 33.88% |
| Qwen3 0.6B Q8_0 | ordinal | 200/800 (25.00%) | 46.12% | 26.83% |

판단 2,000개는 binary 600개, choice 600개, ordinal 800개입니다. 이진 원시 라벨은 p_true >= 0.5를 사용하고 choice·ordinal 동점은 원래 후보 순서를 따릅니다. 네이티브 수락 정답은 해당 원시 동점 규칙과 독립적으로 실제 선택값을 채점합니다.

<a id="by-workflow"></a>
## 워크플로우별

| 체크포인트 | 워크플로 | 정책 적용 전 정답 / 계획 |
| --- | --- | ---: |
| Gemma 4 E2B Q8_0 | agent_trace_observability | 239/500 (47.80%) |
| Gemma 4 E2B Q8_0 | customer_service | 328/500 (65.60%) |
| Gemma 4 E2B Q8_0 | invoice_processing | 222/500 (44.40%) |
| Gemma 4 E2B Q8_0 | security_incidents | 297/500 (59.40%) |
| Qwen3 0.6B Q8_0 | agent_trace_observability | 170/500 (34.00%) |
| Qwen3 0.6B Q8_0 | customer_service | 155/500 (31.00%) |
| Qwen3 0.6B Q8_0 | invoice_processing | 186/500 (37.20%) |
| Qwen3 0.6B Q8_0 | security_incidents | 114/500 (22.80%) |

<a id="environment-and-provenance"></a>
## 환경과 출처

두 실행은 같은 고정 네이티브 평가기를 사용했습니다. NVIDIA GeForce RTX 3080 10 GiB, 드라이버 596.21, WSL2의 Intel Core i9-9900K, CPU 스레드 네 개입니다. 설정은 CUDA 전체 오프로드, 컨텍스트 8192, 배치·마이크로배치 256, Flash Attention 비활성화, legacy/minimal 프롬프트, fresh 직렬 실행, read 로딩입니다. LoRA, 출력 헤드, 학습된 보정, 준비 캐시는 사용하지 않았습니다. 모델은 차례로 실행했습니다. 기존 디스플레이·호스트 GPU 사용이 있었으며 로컬 단일 실행의 시간입니다.

평가기 소스는 선택적 thinking 구현 전 `fb3ad4e`로 고정했습니다. 전체 테스트 수치는 **direct 모드** 측정입니다. Thinking은 별도의 제한된 생성·데모 검증이 있으며 전체 테스트에서 정답률 향상을 주장하지 않습니다. 최종 소스 변경이 측정된 실행 파일을 소급 변경하지 않습니다.

- 평가자 SHA-256: `7fb6e5237188de8161a3025f6b0ce7e16e7b2e3274520a19879e95a115d221ba`.
- 데이터 세트 Parquet SHA-256: `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`.
- JSONL SHA-256 요청: `a677142b77f72f4445d79d10213d79e7de564f223553bb3a4c88105a181f7e6d`.
- 출처: [manifest](../../benchmarks/typed-decisions-20260926/manifest.json). 체크인된 `gemma4-e2b-run.json` 및 `qwen3-06b-run.json`에는 원래 명령과 모델 해시가 포함되어 있습니다. 공개 웹사이트 매니페스트에는 이식 가능한 명령 경로와 함께 이 두 가지 실행 기록도 포함되어 있습니다.
- 공개 다운로드: [Gemma 요약 JSON](../../benchmarks/typed-decisions-20260926/gemma4-e2b-summary.json), [Qwen 요약 JSON](../../benchmarks/typed-decisions-20260926/qwen3-06b-summary.json), [Gemma 판단 2,000개 채점 기록](../../benchmarks/typed-decisions-20260926/gemma4-e2b-scored.jsonl), [Qwen 판단 2,000개 채점 기록](../../benchmarks/typed-decisions-20260926/qwen3-06b-scored.jsonl) 및 위의 매니페스트. 점수가 매겨진 레코드에는 확률, 원시/네이티브 선택, 정확성, 질량, 작업 흐름 및 대기 시간이 포함됩니다. 여기에는 원래 상태나 모델에서 생성된 사고 텍스트가 포함되어 있지 않습니다. 공개 요약 및 점수가 매겨진 JSONL는 체크인된 바이트를 보존합니다. 공개 매니페스트는 실행 메타데이터를 포함하고 절대 네이티브 라이브러리 및 명령 경로를 기본 이름으로 바꿉니다. `public_export` 필드는 이러한 변경 사항을 기록합니다. 명령 플래그와 해시는 보존됩니다.
- 전체 네이티브 예측 JSONL 및 로그는 무시된 `results/typed-decisions-20260926/`에 남아 있습니다. 이러한 원시 파일과 고정된 실행 파일은 저장소에 번들로 포함되지 않습니다.

요약은 hard Brier, hard NLL(자연 로그, 확률 하한 1e-12), 최대 확률의 동일 폭 구간 10개·15개를 사용하는 ECE, binary·choice의 soft-target Brier/TVD/KL, ordinal의 기대 수준 MAE·within-one도 보고합니다. ECE 정의는 명시되어 있으며 다른 프로젝트 구현과 같다고 가정하지 않습니다.

<a id="ollaya-reference-figures"></a>
## Ollaya 참고 수치

[Ollaya의 공개 GGUF 연구](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/families/llm-logits.md#measured-quality-typed-decisions-test-400-rows--2000-questions)는 자체 400케이스 / 2,000질문 typed 테스트에서 Gemma 4 E2B Q8_0 56.6%, Gemma 4 12B Q4_0 71.7%, decider-2B 59.1%를 보고합니다. 이는 **원본에 보고된 참조 수치**이며 여기서 수행한 Ollaya 실행이 아닙니다. 프롬프트·런타임·보정·평가 절차가 다르며 원본은 L2S1과 바이트가 같은 요청이나 통제된 지연 시간 비교를 입증하지 않습니다.

우리 Gemma E2B 결과는 54.3%입니다. 이 근거는 L2S1의 정답률 우위를 입증하지 않습니다. 다른 크기의 모델, 작업별 미세 조정, 독립 측정한 지연 시간을 런타임만의 비교처럼 제시하면 안 됩니다.

<a id="reproduce"></a>
## 재현

[작업 어댑터 가이드](LAYA_BENCHMARK.md#prepare-and-run)의 fetch·prepare·run·score 명령을 사용합니다. `--limit`나 추가 반복 없이 전체 `typed` 평가를 실행하세요. 상주 체크포인트 하나, 기록된 매개변수, 변경하지 않은 테스트 라벨을 유지합니다.

```sh
target/release/l2s1-tools laya-benchmark run \
  --prepared results/laya/typed --output results/laya/typed-gemma-e2b \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
  --context 8192 --batch 256 --threads 4 --model-load-mode read
```

평가기를 `llama-cuda`로 빌드하고 대응하는 네이티브 라이브러리를 제공하세요. 새 소스의 재현은 새 측정이며 고정 실행 파일의 시간 검증이 아닙니다.
