# 4B 이하 이미지 분류 검증 — 2026-10-01

복구한 전용 프롬프트·SDK 및 기본 보류 없음 변경이 포함된 `72b8ac6049025db84a93f83de4015e1ac29741ef`에서 실제 CUDA 이미지 분류를 실행했다. 엄격히 총 4B 이하인 두 모델 중 관측 최고는 **Qwen3-VL 2B: 95/120 (79.17%)**이다. Qwen3.5 4B급은 비교용이며 이미지 인코더까지 포함하면 4B를 넘는다.

## 결과

| 모델, Q8_0 | 모델+프로젝터 저장 텐서 원소 | 정답/전체 | raw top-1 | Wilson 95% 구간 | p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-2B-Instruct | 2.128B | 95/120 | 79.17% | 71.05–85.47% | 105.91 / 125.22 |
| Qwen3.5-2B | 2.213B | 85/120 | 70.83% | 62.16–78.22% | 99.51 / 108.40 |
| Qwen3.5-4B (참고) | 4.539B | 94/120 | 78.33% | 70.15–84.76% | 146.84 / 155.15 |

모든 모델은 **120/120 반환, coverage 100%, 보류 0건**이다. 따라서 accepted accuracy와 correct/all은 raw top-1과 같다. 상위 두 모델은 단 1장 차이이며 우열이 확정됐다는 의미는 아니다. 저장 텐서 원소 수는 GGUF tensor-info의 각 shape 곱을 합산한 값이며 활성 파라미터 수와 구분한다. Qwen3.5-4B 텍스트 GGUF만으로도 4.206B이다.

| 실제 클래스 (각 20장) | Qwen3-VL 2B | Qwen3.5 2B | Qwen3.5 4B급 |
| --- | ---: | ---: | ---: |
| cardboard | 19 | 20 | 18 |
| glass | 20 | 20 | 18 |
| metal | 19 | 18 | 18 |
| paper | 20 | 13 | 20 |
| plastic | 17 | 14 | 19 |
| trash | 0 | 0 | 1 |

주된 실패는 `trash`이다. 고정 질문은 주재질을 고르도록 요구하지만 TrashNet의 잔여 쓰레기 범주는 종이·플라스틱처럼 보이는 품목도 포함하므로 라벨 정의와 질문 사이의 불일치가 원인일 수 있다. 이는 혼동행렬에서의 추정이며, 이번에는 질문 변경이나 정답을 이용한 튜닝을 하지 않았다.

## 실행 조건과 한계

- 데이터: [garythung/trashnet](https://github.com/garythung/trashnet/tree/6fa2b878c6c1b4304b91109070ce0edf9279bb31), 기존 [selection.json](../trashnet-vision-20260925/selection.json)의 6종 × 20장. ZIP과 각 원본 JPEG SHA-256을 검증했다. 이 표본은 과거 실험에서도 관측됐으며 새 독립 holdout이나 실제 운영 정확도가 아니다. 신뢰구간도 데이터 선택·학습 중복 불확실성을 포함하지 않는다.
- 과거 baseline 질문과 선택지 설명을 고정했다. 파일명과 정답 라벨은 요청에서 제외하고 이미지 바이트·빈 state·동일한 6개 선택지만 전송했다. 캡션 생성이나 지도학습 분류기는 사용하지 않았다.
- 실제 loopback HTTP `/v1/decisions`, 순차 1장/요청, warmup 1회 후 120장 1회. 요청이나 CLI에서 정책 임계값을 지정하지 않고 매 응답의 기본값 `0/0`을 검증했다. 모든 요청에서 truncation 없음과 CUDA offload 요청을 확인했다.
- `fresh`, context 8192, batch/ubatch 256, threads 4, FlashAttention off, preparation cache 0, model-load-mode read. `Auto`가 모델 GGUF template을 선택했다. 이번에 복원한 `winnow`/`gemma4-decision`은 **텍스트 전용**이며 이미지에 적용하지 않았다.
- RTX 3080 10 GiB, driver 596.21, WSL2, i9-9900K(노출 논리 CPU 4개). 지연시간은 HTTP 직렬화된 요청 전송·이미지 처리·추론·응답 읽기를 포함하고 모델 로딩, warmup, 메모리 표본 채집을 제외한다. GPU clocks는 고정하지 않았다.
- Qwen3.5 실행의 native 로그에는 recurrent position의 `non-consecutive token position` 경고가 있다. 120개 응답은 모두 성공했지만 다른 엔진과 이미지 logits를 대조하지 않았으므로 해당 backend에서의 관측 결과로 한정한다. 원본 로그도 보존한다.

| 모델 | 프로세스 peak RSS MiB | 전체 GPU 실행 전 / 측정 중 최대 MiB |
| --- | ---: | ---: |
| Qwen3-VL 2B | 839.89 | 1937 / 9874 |
| Qwen3.5 2B | 1077.84 | 1522 / 9370 |
| Qwen3.5 4B급 | 1206.07 | 1520 / 9738 |

GPU 수치는 데스크톱 및 런타임 버퍼를 포함한 **전체 보드 사용량**이며 모델 전용 메모리나 최소 VRAM 요구량이 아니다. 프로세스 RSS와 합산하지 않는다.

## 과거 선택적 정책과의 차이

동일한 이번 logits에 과거 정책(top probability ≥ 0.8, candidate mass ≥ 0.05, 동률 거부)을 **사후 적용**한 진단이다. 선택적 정책으로 서버를 다시 실행한 결과는 아니다. 임계값 제거는 순위 정확도를 높이지 않는다.

| 모델 | 사후 채택/120 | 사후 채택 정답/채택 | 사후 채택 정답/전체 | 0/0으로 추가 반환된 정답 / 오답 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL 2B | 115 | 91/115 (79.13%) | 75.83% | 4 / 1 |
| Qwen3.5 2B | 98 | 79/98 (80.61%) | 65.83% | 6 / 16 |
| Qwen3.5 4B급 | 94 | 77/94 (81.91%) | 64.17% | 17 / 9 |

## 재현

`evaluate.py`는 표준 Python 라이브러리와 `nvidia-smi`를 사용한다. 각 모델에 대해 아래 명령의 model/mmproj/output을 바꿔 한 번씩 실행한다. 결과 디렉터리가 이미 있으면 덮어쓰지 않는다.

```sh
LD_LIBRARY_PATH=/path/to/matching/native/lib python3 benchmarks/vision-small-20261001/evaluate.py \
  --binary /path/to/l2s1 \
  --model models/Qwen3-VL-2B-Instruct-Q8_0.gguf \
  --mmproj models/mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf \
  --archive /path/to/dataset-resized.zip \
  --output /path/to/new-output
```

각 모델 폴더의 `responses.jsonl`에는 warmup(index -1)과 전체 120개 응답·latency가 있다. `summary.json`은 이미지별 정오답·혼동행렬·메모리 표본을 포함한다. 최상위 `summary.json`은 합산 및 사후 정책 진단, `provenance.json`은 matching native libraries·모델 프로젝터 revision/해시, `restoration-validation.json`은 프롬프트·SDK 복구 검증 기록이다. 대형 모델 가중치와 이미지 데이터는 포함하지 않는다.
