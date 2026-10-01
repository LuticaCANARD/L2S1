# 2B/4B 정확도 개선 및 보존 변경 통합 검증 — 2026-10-01

같은 Winnow E4B Q8 가중치에서 범용 `model/legacy`와 전용 `winnow/state-first`를 비교했다. 개발셋으로 설정을 선택한 뒤 고정했으며, 검증·테스트 결과를 보고 다시 선택하지 않았다. E2B는 개발 후보 다섯 개 모두 기존 설정 이하였으므로 기존 `model/minimal`을 유지한다. 두 모델 사이의 차이를 동일 모델의 개선으로 계산하지 않는다.

## E4B 결과

| 평가 | 기존 범용 | Winnow 전용 | 변화 | 상태 단위 bootstrap 95% 구간 |
| --- | ---: | ---: | ---: | ---: |
| 추가 검증 | 635/1000 (63.50%) | 715/1000 (71.50%) | +8.00%p | [+4.50, +11.30]%p |
| typed-decisions test split | 1282/2000 (64.10%) | 1444/2000 (72.20%) | +8.10%p | [+5.70, +10.45]%p |
| 공개 JevBench | 180/231 (77.92%) | 186/231 (80.52%) | +2.60%p | [-0.87, +6.49]%p |

모든 실행은 실제 CUDA 추론이며 동일 가중치·context8192·batch/ubatch256·스레드4·FlashAttention off·fresh 조건이다. 프로필 변경은 레이아웃, 선택지 ID 표현, 질문 형식과 답변 시작 경계를 함께 바꾼다. 특정 한 요소만의 인과 효과로 해석할 수 없다.

기권 임계값은 모든 구성에서 0/0이다. 아래 수락 수치를 통해 실제 반환을 검산하며, 후보 확률이나 기권 제거를 정확도 향상으로 세지 않았다.

| 평가/모델/설정 | 반환/전체 | 채택 정답/채택 | 채택 정답/전체 | 기권 |
| --- | ---: | ---: | ---: | ---: |
| holdout/e2b/baseline | 1000/1000 | 538/1000 | 53.80% | 0 |
| holdout/e4b/baseline | 1000/1000 | 635/1000 | 63.50% | 0 |
| holdout/e4b/improved | 1000/1000 | 715/1000 | 71.50% | 0 |
| typed-test/e2b/baseline | 2000/2000 | 1086/2000 | 54.30% | 0 |
| typed-test/e4b/baseline | 2000/2000 | 1282/2000 | 64.10% | 0 |
| typed-test/e4b/improved | 2000/2000 | 1444/2000 | 72.20% | 0 |
| regression/e2b/baseline | 231/231 | 159/231 | 68.83% | 0 |
| regression/e4b/baseline | 231/231 | 180/231 | 77.92% | 0 |
| regression/e4b/improved | 231/231 | 186/231 | 80.52% | 0 |

## 개발셋과 선택

| 모델 | 후보 | 정답/640 |
| --- | --- | ---: |
| e2b | model | 363 |
| e2b | winnow | 328 |
| e2b | gemma4-decision | 326 |
| e2b | model-typed | 347 |
| e2b | model-typed-examples | 336 |
| e4b | model | 419 |
| e4b | winnow | 450 |

E2B의 초기 전용 프로필이 악화되어 기존 typed/typed-examples 옵션을 개발 단계에서 추가 비교했다. 이 적응적 탐색은 `plan.json`에 기록했다. 이후 `selection.json`을 고정했고, 추가 검증과 typed test 실행의 manifest가 그 SHA-256을 보존한다. E2B는 baseline 한 번만 측정했고 비교 JSON의 동일 구성은 같은 관측이다.

## 검토 결과

- 이전 전용 프롬프트 구현은 별도 커밋에 있었고 현재 코드의 `Auto` 경로에는 없었다. 이를 현재 shared-input·네이티브 API에 맞게 복구했다. 범용 next-token 분류는 체크포인트에 맞는 분류 형식과 동일하지 않다.
- E2B에서 전용 형식이나 설명을 늘리는 변경은 개발 정확도를 낮췄다. 이 다섯 후보의 결과만으로 모든 2B 모델의 능력 한계를 단정할 수 없다.
- E4B의 추가 검증에서는 binary가 175/300→240/300, ordinal이 254/400→273/400, choice가 206/300→202/300이었다. 전체 개선이 모든 판단 유형에서의 개선을 뜻하지 않는다.
- 모델의 후보 softmax는 선택지 사이의 상대 확률이다. 높은 확률을 정답 보장으로 해석하면 오답에 과신하게 된다. 별도 온도 보정·학습·임계값 최적화는 수행하지 않았다.

JevBench의 +6/231 차이는 신뢰구간에 0이 포함돼 확실한 개선으로 단정하지 않는다. 기존 0.8/0.05 기권 기준을 사후 적용하면 반환174개 중 정답은160→157개로 감소한다. 이번 결과는 0/0 강제 응답 정책에 대한 것이며 선택적 응답 정책의 개선을 주장하지 않는다.

## 범위와 재현성

- 출처는 [Hugging Face LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)이다. `all/test-00000-of-00001.parquet`은 제공 test split이며 업계 표준 벤치마크 또는 실제 운영 데이터라는 뜻이 아니다. revision `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8`의 영어 합성 teacher 라벨과 비교한 수치다. 실사용 정답률·한국어·새 업무로 일반화하지 않는다.
- 추가 검증은 이전 개발128·보정128·제공 test400 상태를 exact-state hash로 제외한 train 영역의 200상태/1000문항이다. 체크포인트 학습 데이터와의 중복·근접 중복까지 배제하지 못하므로 독립적인 모델학습 holdout으로 부르지 않는다.
- typed-decisions 제공 test split은400상태/2000문항이며 이전 실험에서 관측된 데이터다. 현재 실험에서는 후보 고정 후에만 실행했다.
- JevBench는 pinned public231문항(공식 전체534가 아님)의 별도 전이/회귀 진단이다. 결과에 따라 프로필을 다시 선택하지 않았다.
- 5000회 paired state bootstrap은 한 상태의 질문들을 함께 재표집한다. 동일 업무·합성 라벨·후보 탐색 이력을 포함한 이 실험 범위의 구간이며 운영 보장이 아니다.
- GPU는 RTX3080 10GiB, WSL2, i9-9900K(노출 CPU4개). 실행당 첫 요청 warmup1회 제외, native in-process latency로 모델 로딩과 HTTP를 제외했다. 모든 추론은 순차이며 clocks를 고정하지 않았다.
- `provenance.json`에 모델·실행 파일·네이티브 라이브러리·입력·출력 해시와 최대 RSS/전체보드 GPU 표본을 보존한다. GPU 값에는 데스크톱과 로딩이 포함된다. 단일 패스 지연시간은 안정된 속도 개선의 증거가 아니다.

## 지연시간과 확률 진단

| 평가/모델/설정 | p50 ms | p95 ms | NLL | ECE15 |
| --- | ---: | ---: | ---: | ---: |
| holdout/e2b/baseline | 322.13 | 365.50 | 3.8166 | 0.4238 |
| holdout/e4b/baseline | 459.46 | 530.01 | 1.0060 | 0.1876 |
| holdout/e4b/improved | 477.56 | 586.80 | 0.7224 | 0.0774 |
| typed-test/e2b/baseline | 324.10 | 441.00 | 3.5652 | 0.4202 |
| typed-test/e4b/baseline | 467.98 | 543.14 | 0.9687 | 0.1770 |
| typed-test/e4b/improved | 502.00 | 666.40 | 0.7117 | 0.0610 |
| regression/e2b/baseline | 40.26 | 443.47 | 2.1374 | 0.2769 |
| regression/e4b/baseline | 64.95 | 685.07 | 0.5023 | 0.0932 |
| regression/e4b/improved | 60.91 | 777.52 | 0.4968 | 0.0787 |

Typed 지연시간은 질문5개 요청당, JevBench는 질문1개 요청당이다. `historical_gate`는 같은 예측에 0.8/0.05와 tie rejection을 적용한 사후 진단이며 실제 반환 정책과 구분한다.

## 적용

소스 빌드 CLI에서 Winnow E4B에는 `--prompt-profile winnow`를 명시한다. Python: `LoadOptions(prompt_profile="winnow", ...)`; TypeScript: `promptProfile: "winnow"`. `auto`는 그대로이며 E2B에 Winnow를 자동 적용하지 않는다. 새 프로필은 이 통합 작업본의 기능으로, 기존 배포된 바이너리를 교체하지 않았다.

```sh
cargo build --release --locked --features llama-cuda --bin l2s1 --example evaluate_jsonl
./target/release/l2s1 --model /path/to/Winnow-E4B-Q8_0.gguf \
  --device cuda --prompt-profile winnow --execution-mode fresh \
  --context 8192 --batch 256 --ubatch 256 --threads 4 --flash-attention off \
  --listen 127.0.0.1:11437
```

프로필은 Gemma4, 텍스트, minimal detail, shared 없는 입력, 선택지2–26개만 지원한다. 미지원 입력은 preflight에서 거부한다. `gemma4-decision`은 명시적인 실험용 옵션이며 이번 E2B 실험에서는 채택하지 않았다.

## 보존 변경과 검사

원래 작업 폴더의 tracked 변경과 untracked 파일을 대조해 통합했다. 평가 지표·shared_decision·학습 리포트·웹 시각화 변경은 현재 기반에 이미 존재했다. 충돌에서는 shared 필드와 최신 API 지원을 보존했으며, 동일한 변경을 중복 적용하지 않았다. 패키지 캐시·생성 산출물은 소스 커밋에 포함하지 않았다. 원래 작업 폴더는 덮어쓰지 않았다. 상세 감사는 로컬 `preserved-integration.json`과 원본 patch에 있다.

`validation.json`은 Rust, 네이티브, SDK, 학습 도구, 브라우저 검사의 정확한 통과/제외 범위를 기록한다. 브라우저 설치 안내의 고정0.1.4 기대값과 이전 기권 기본값을 가정한 테스트를 현재 계약에 맞게 수정했다. 웹 테스트에는 fixture/mock 검사가 포함되며 배포·실제 브라우저 GPU 성능 검증이 아니다.

실제 로컬 HTTP/CUDA 서버의 5개 판단 확률은 네이티브 평가와 정확히 일치했다. 미지원 shared/27개 선택지는 HTTP400으로 거부했다. 보존된 shared-decision의 E2B 실제 추론 검사도 통과했다.

재현 도구: `scripts/prepare_small_model_accuracy.py`, `scripts/compare_small_model_prompts.py`. 네이티브 평가기는 정답 없는 requests JSONL만 받으며 정답은 추론 종료 후 별도 채점한다. 기존 JevBench 예측 재채점에서도159/231 raw,153/213 accepted가 일치했다.
