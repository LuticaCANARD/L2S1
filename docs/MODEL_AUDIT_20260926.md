# 모델 실측 증거 재검토 — 2026-09-26

[English](en/MODEL_AUDIT_20260926.md) · [한국어](ko/MODEL_AUDIT_20260926.md) · [日本語](ja/MODEL_AUDIT_20260926.md)

2026-09-23~25에 기록된 측정의 원본 자료를 재검토한 문서입니다. 새로운 추론 실행은 아닙니다. 코드로 표시한 증거 경로는 저장소 기준 로컬 자료이며, 공개 집계와 기존 보고서는 아래 링크로 제공합니다.

Raw는 정책 적용 전 최고 후보 정답/전체, 수락 정확도는 수락 정답/수락, coverage는 수락/전체입니다. 표에는 원본 건수를 함께 표시하며, 별도 표시가 없으면 p50 단위는 ms입니다.

## 동일 조건 RTX 3060 비교 — 2026-09-23

완료한 22개 설정 × JevBench 공개 231문항, 총 5,082응답을 재집계했습니다. 공개 목록은 20개 설정 / 4,620응답이며 raw 정답 2,924, 수락 3,122, 수락 정답 2,279, 수락 오답 843, 보류 1,498, 오류 0입니다. 원래 시도한 runtime 설정은 23개로, 22개가 완료했고 기본 GPT-OSS CUDA Graph 설정은 129응답 후 실패했습니다. 별도의 Graph 비활성 설정은 완료했습니다.

| 모델 | Raw 정답 / 231 | 정답 / 수락 | Coverage | p50 ms | GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 4B Q8_0 | 184/231 (79.65%) | 129/138 (93.48%) | 138/231 (59.74%) | 86.88 | 5039 |
| Qwen3.5 9B Q4_K_M | 180/231 (77.92%) | 154/167 (92.22%) | 167/231 (72.29%) | 130.55 | 5637 |
| Gemma4 E4B Q4_K_M | 179/231 (77.49%) | 164/194 (84.54%) | 194/231 (83.98%) | 86.62 | 3989 |
| Qwen3.5 4B Q4_K_M | 176/231 (76.19%) | 130/140 (92.86%) | 140/231 (60.61%) | 88.47 | 3381 |
| Gemma4 E2B Q8_0 | 157/231 (67.97%) | 151/209 (72.25%) | 209/231 (90.48%) | 46.79 | 3025 |
| Qwen3.5 2B Q8_0 | 144/231 (62.34%) | 83/98 (84.69%) | 98/231 (42.42%) | 40.99 | 2465 |

이 비교에서 Qwen3.5 4B Q8의 raw 정확도가 가장 높고, 수락한 138개 중 129개가 정답입니다. Gemma4 E4B Q4는 비슷한 p50과 1,050 MiB 적은 GPU 표본 최댓값으로 수락 194개 중 164개 정답을 반환합니다. Qwen3.5 9B Q4는 수락 정확도 92.22%와 coverage 72.29%를 함께 제공합니다. Gemma4 E2B Q8은 raw 67.97%, p50 46.79 ms입니다.

완료한 22개 설정은 request/evaluator hash, RTX 3060 12 GiB, context 8192, batch/ubatch 256, CPU threads 4, FlashAttention off, fresh/legacy prompt, 내장 모델 템플릿, 정책 0.8 / 0.05를 공유합니다. 시간은 Rust 직렬 판단이며 모델 로드와 워밍업 1회를 제외합니다. GPU MiB는 로드·워밍업을 포함해 200 ms마다 측정한 전체 보드 사용량 최댓값입니다. GPT-OSS는 `GGML_CUDA_DISABLE_GRAPHS=1`을 사용합니다.

Raw 정확도, 수락 건수, 다중 분류 Brier, 10구간 ECE, 직렬 p50/p95, request hash, GPU CSV 최댓값이 기록된 집계와 일치합니다. 다운로드 종료 후 별도 재실행한 Qwen3.5 4B Q8과 Gemma4 E2B는 모든 후보 확률이 같고, p50은 86.99 / 46.08 ms입니다. 원래 비교 시간과 별도로 보존합니다.

[공개 RTX 3060 집계](../benchmarks/rtx3060-20260926/summary.json) · [RTX 3060 보고서](RTX3060_BENCHMARK.md)

Qwen3.5 9B의 Q4_K_M과 Q8_0는 raw 정답이 모두 180/231 (77.92%)입니다. GPU 보드 표본 최댓값은 5,637 / 8,817 MiB로, 이 실행에서 Q4가 36.07% 적습니다. 집계 정확도를 비교한 값이며 출력 분포의 동일성을 뜻하지 않습니다.

## Gemma 31B / 26B CPU·GPU 혼합 배치 — 2026-09-23

두 실행은 서로 동일한 JevBench 231개 request와 evaluator를 사용합니다. evaluator는 앞의 모델 비교와 다릅니다. context 8192, batch/ubatch 256, CPU threads 8, fresh/legacy prompt, read 로드, 정책 0.8 / 0.05입니다.

| 모델 / 배치 | Raw 정답 / 231 | 정답 / 수락 | Coverage | p50 / p95 ms | 보드 GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 31B Q4_K_M / GPU 24 | 207/231 (89.61%) | 206/228 (90.35%) | 228/231 (98.70%) | 2312.37 / 24590.96 | 10897 |
| Gemma4 26B A4B UD-Q4_K_M / CPU expert 18 | 196/231 (84.85%) | 191/220 (86.82%) | 220/231 (95.24%) | 653.19 / 7338.46 | 10557 |

Gemma31B는 반복층 23개와 출력층을 GPU에, 반복층 37개를 CPU에 둡니다. 전체 실행의 engine peak RSS는 15.92 GiB, 프로세스 GPU 할당 표본 최댓값은 10.60 GiB, 프로세스 swap 표본은 0입니다. 표의 보드 메모리는 프로세스 할당과 구분하며 RSS에는 VRAM이 포함되지 않습니다. Gemma26B는 CPU expert 18층을 사용합니다. warm filesystem에서 로드 방식을 번갈아 실행한 6회 검사의 peak RSS 중앙값은 `auto` 16.17 GiB / `read` 9.35 GiB이며, 로드 방식에 따른 RSS 비교입니다.

## 전체 라벨 intent 분류 — 2026-09-23

RTX 3060에서 모델 4개 × BANKING77 영어 200개 / MASSIVE 한국어 200개, 총 1,600응답을 평가했습니다. 요청마다 77개 / 60개 전체 라벨을 제공합니다. Raw·수락 건수·p50이 native 원본과 일치합니다. context 8192, batch/ubatch 256, CPU threads 4, fresh/legacy, FlashAttention off, 정책 0.8 / 0.05이며 모델 로드와 워밍업은 제외합니다. Intent evaluator는 JevBench evaluator들과 별도입니다.

| 모델 | 데이터셋 | Raw 정답 / 200 | 정답 / 수락 | Coverage | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma4 E2B Q8_0 | BANKING77 en | 123/200 (61.50%) | 119/181 (65.75%) | 181/200 (90.50%) | 214.15 |
| Gemma4 E2B Q8_0 | MASSIVE ko | 103/200 (51.50%) | 100/181 (55.25%) | 181/200 (90.50%) | 157.52 |
| Gemma4 E4B Q8_0 | BANKING77 en | 130/200 (65.00%) | 127/173 (73.41%) | 173/200 (86.50%) | 371.72 |
| Gemma4 E4B Q8_0 | MASSIVE ko | 143/200 (71.50%) | 136/174 (78.16%) | 174/200 (87.00%) | 277.05 |
| Gemma4 E4B Q4_K_M | BANKING77 en | 130/200 (65.00%) | 124/172 (72.09%) | 172/200 (86.00%) | 383.61 |
| Gemma4 E4B Q4_K_M | MASSIVE ko | 138/200 (69.00%) | 129/168 (76.79%) | 168/200 (84.00%) | 287.00 |
| Qwen3 8B Q8_0 | BANKING77 en | 111/200 (55.50%) | 107/175 (61.14%) | 175/200 (87.50%) | 877.64 |
| Qwen3 8B Q8_0 | MASSIVE ko | 98/200 (49.00%) | 96/173 (55.49%) | 173/200 (86.50%) | 593.45 |

Gemma4 E4B Q8은 BANKING77에서 E4B Q4와 같은 65.0%, MASSIVE 한국어에서는 네 모델 중 가장 높은 71.5%입니다. Gemma4 E2B Q8의 영어 / 한국어 p50은 214.15 / 157.52 ms입니다.

[Intent 보고서](INTENT_BENCHMARK.md)

## 이미지 분류 baseline — 2026-09-24 / 25

원본 JPEG 한 장마다 로컬 HTTP 요청을 직렬 실행했습니다. 로드된 모델의 첫 요청을 포함하며 시작 시간은 제외하고 워밍업은 하지 않았습니다. 정책은 0.8 / 0.05이며 데이터셋마다 클래스와 고정 baseline 질문을 유지합니다. 9월 24일은 RTX 3060, 9월 25일 TrashNet은 RTX 3080입니다.

| 데이터셋 / GPU | 모델 | Raw 정답 / 전체 | 정답 / 수락 | Coverage | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cats/dogs / RTX 3060 | Gemma4 E2B Q8_0 | 69/70 (98.57%) | 69/70 (98.57%) | 70/70 (100%) | 99.18 |
| Caltech 30 classes / RTX 3060 | Gemma4 E2B Q8_0 | 122/150 (81.33%) | 122/148 (82.43%) | 148/150 (98.67%) | 195.28 |
| TrashNet 6 classes / RTX 3080 | Qwen3-VL 2B Q8_0 | 95/120 (79.17%) | 91/115 (79.13%) | 115/120 (95.83%) | 101.08 |
| TrashNet 6 classes / RTX 3080 | Gemma4 E2B Q8_0 | 64/120 (53.33%) | 64/113 (56.64%) | 113/120 (94.17%) | 79.12 |

고양이/개는 validation 70장(고양이 24 / 개 46), context 2048입니다. 선택지 순서를 뒤집은 실행도 69/70이며 p50 98.75 ms입니다. Caltech는 30클래스 × 5장, context 4096이며 GPU 3,855 MiB 기록은 한 시점 표본입니다. TrashNet은 6클래스 × 20장입니다. Qwen3-VL 2B Q8은 기록된 세 TrashNet baseline 모델 중 raw가 가장 높습니다. 후속 traits prompt와 batch 실험은 별도 탐색 프로토콜입니다. 각 자료는 독립 데이터셋·evaluator의 결과이며 HTTP 시간은 Rust 비교 시간과 구분합니다.

[이미지 평가 보고서](VISION_BENCHMARK.md)

## 합성 입력의 공유 상태 prefix 재사용 — 2026-09-25

RTX 3080, context 2048, batch/ubatch 32, CPU threads 4, state-first prompt, preparation cache 비활성입니다. 예시 상태에는 filler 200단어를 넣었고 질문 16개는 지시문과 ID가 서로 다릅니다. 순서를 번갈아 5라운드 측정했으며 각 경로에 워밍업 1회를 제외했습니다. 시간은 준비와 session 생성·해제를 포함한 전체 실행 ms 중앙값이며 모델 로드와 보고서 직렬화는 제외합니다.

| 모델 | Fresh ms | 공유 session ms | 비율 | 재사용 / 입력 토큰 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8_0 | 1461.77 | 531.49 | 2.75× | 3840/5789 |
| Gemma4 E2B Q8_0 | 2735.54 | 985.42 | 2.78× | 3840/5751 |

원본 JSON의 라운드별 기록으로 표의 중앙값을 재현했습니다. 모델별 210쌍의 판단에서 raw logit, 후보 확률, candidate mass, 선택 값, 보류 사유가 정확히 일치합니다. 실제 KV prefix를 재사용하는 명시적 native session의 합성 입력 평가이며, 과제 정확도와 유지 session 메모리는 측정하지 않았습니다.

[공유 상태 보고서](../benchmarks/shared-state-cache-20260925/REPORT.md)

## 재검토한 원본 경로

아래 Python 파일명은 당시 실행의 출처입니다. 중복 Python 구현은 삭제했으며, 현재 재집계와 실행은 각각 `l2s1-tools report-jevbench-matrix`, `l2s1-tools jevbench-public run`으로 수행합니다.

- RTX 3060: `results/jevbench-matrix-20260923/REPORT.json`, `environment.json`; `<configuration>/{manifest.json,summary.json,tasks-with-gold.jsonl,requests.jsonl,predictions.jsonl,gpu-memory.csv,inference.stderr.log}`. Recount: `scripts/report_jevbench_matrix.py`; GPU sampling: `scripts/jevbench_public.py`.

- Gemma31B: `results/large-model-20260923T141422Z/REPORT.json`, `remote/jevbench/{manifest.json,summary.json,predictions.jsonl,tasks-with-gold.jsonl,gpu-memory.csv}`. Gemma26B: `results/gemma26-lowrss-20260923T135227Z/REPORT.json`, `remote/jevbench-read/{manifest.json,summary.json,predictions.jsonl,tasks-with-gold.jsonl,gpu-memory.csv}`.

- Intent: `results/intent-wide-20260923/prepared/gold.jsonl`, `runs/<model>/{manifest.json,summary.json,predictions.jsonl,scored.jsonl}`.

- Images: `benchmarks/cats-dogs-vision-20260924/{summary.json,observations.jsonl}` and `dog-cat/`; `benchmarks/caltech101-vision-20260924/{summary.json,observations.jsonl,selection.json}`; `benchmarks/trashnet-vision-20260925/{gemma4,qwen3vl,smolvlm}/{summary.json,observations.jsonl}`.

- Shared state: `benchmarks/shared-state-cache-20260925/{qwen3-0.6b-sm86.json,gemma4-e2b-sm86.json}`; workload: `examples/benchmark_shared_state_cache.rs`.
