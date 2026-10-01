# TrashNet 소형 모델 프롬프트 실험

## 고정한 프롬프트

> Classify the main waste item in the photo into one of the six TrashNet categories. Use the visible object and the category descriptions. Ignore the background. Choose exactly one category.

- cardboard: A cardboard box, corrugated cardboard, or thick paperboard packaging.
- glass: A glass bottle, glass jar, or another glass object.
- metal: A metal can, metal lid, metal container, or aluminum foil.
- paper: An ordinary paper sheet, newspaper, printed page, or paper bag.
- plastic: A plastic bottle, plastic container, plastic cap, or plastic bag.
- trash: A tissue, napkin, paper towel, disposable paper cup, snack wrapper, candy wrapper, or other residual waste.

학습 분할의 개발용 120장에서 공통 프롬프트를 선택·고정한 뒤 기존 test 481장을 평가했다.

| 모델 | 기존 정답/481 | 전용 프롬프트 정답/481 |
| --- | ---: | ---: |
| Qwen3-VL 2B Q8 | 439 (91.27%) | 427 (88.77%) |
| Qwen3.5 2B Q8 | 389 (80.87%) | 440 (91.48%) |
| Qwen3.5 4B Q8 | 426 (88.57%) | 412 (85.65%) |
| Gemma 4 E4B Q4 | 330 (68.61%) | 323 (67.15%) |

## 참조 이미지 6장 + 대상 1장

참조는 train에서 클래스당 1장, seed20261002로 추론 전에 고정했다. 기존 validation 481장(고유 SHA 480개, perceptual group 472개)을 동일한 새 런타임에서 대상만/참조 포함 조건으로 비교했다. 참조·대상 그룹은 겹치지 않는다. `media_ids`는 참조 6장 뒤에 대상을 지정하고, state에는 참조 라벨과 `target_image_position: 7`만 넣었다. 대상 정답·파일명·경로는 요청에서 제외했다.

| 모델 | 대상만 정답/481 | 참조 포함 정답/481 | 변화 %p (그룹 bootstrap 95% 구간) | HTTP p50 ms 대상만 → 참조 포함 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL 2B Q8 | 422 (87.73%) | 184 (38.25%) | -49.48 [-54.26, -44.79] | 115.4 → 617.1 |
| Qwen3.5 2B Q8 | 430 (89.40%) | 317 (65.90%) | -23.49 [-28.06, -19.17] | 111.8 → 579.5 |
| Qwen3.5 4B Q8 | 395 (82.12%) | 239 (49.69%) | -32.43 [-36.85, -28.16] | 177.5 → 881.6 |
| Gemma 4 E4B Q4 | 321 (66.74%) | 180 (37.42%) | -29.31 [-34.09, -24.33] | 145.9 → 616.7 |

이번 고정 참조·프롬프트 구성에서는 네 모델 모두 정확도가 낮아졌다. 이미지와 순서 설명을 함께 추가한 비교이며, 다른 참조 집합이나 이미지 예시 프롬프팅 전체에 일반화하지 않는다. 두 표는 서로 다른 평가 분할이다. 기존 지도학습 실험에서 사용한 분할이며 VLM 사전학습 중복은 알 수 없다.

모든 표는 정책 0/0, coverage 100%, 보류 0건이다. raw top-1, accepted accuracy, correct/all이 같다. 자유 생성이 아닌 next-token 6개 선택지 평가다. 선택적 정책을 적용한 생산 환경 정확도를 뜻하지 않는다. 그룹 구간은 3,000회 재표집하며 모델별 다중 비교를 보정하지 않았다.

환경은 RTX 3080 10 GiB, i9-9900K, WSL2, driver596.21이다. fresh/context8192/threads4/batch256/ubatch256, FlashAttention off, 조건별 warmup1 후 순서를 교대했다. HTTP 지연은 이미지 전송·인코딩·추론·응답을 포함하고 로딩·warmup은 제외한다. 참조 실험 peak RSS MiB는 모델 순서대로 871/1104/1231/2956, 전체 GPU 보드 표본 최대 MiB는 9860/9414/9837/6333이다. GPU 수치는 모델 전용 메모리가 아니며 RSS와 합산하지 않는다.

Gemma는 [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it), 4.5B effective / embedding 포함 8B이다. Gemma Q4와 Qwen Q8의 모델 간 비교에는 양자화 차이가 있다. 각 모델의 두 입력 조건은 같은 가중치를 썼다.

전체 원본 응답·행별 결과·실행 스크립트·해시는 로컬 `results/`에 보존하고 PR에는 요약만 포함한다. ZIP SHA256: `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`; 참조 실험 binary SHA256: `806b80ff8e80c2d882b7de38d719e54c3c03d5eb055a3cf61765265a7e3aff82`; llama.cpp revision: `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.

검증: 전체 Rust CUDA 테스트 129개 통과(44개 ignored), Qwen/Gemma 실제 7-image 테스트 통과, HTTP/stdio와 Python·TypeScript·C++에서 동일한 이미지 순서·7개 native chunk 확인. Python 18개, TypeScript 25개 및 타입 검사, C++ 계약, MCP 9개, native clippy와 Python mypy 통과. 참조 실험 응답 3,856개를 요청 바이트·순서·해시·통계·CUDA/fresh/무절단 여부까지 검산했다.
