<a id="caltech-101-30-way-vision-decision-benchmark"></a>
# Caltech-101: 30-way 비전 판단 벤치마크

[English](../../../en/benchmarks/caltech101-vision-20260924/README.md) · [한국어](README.md) · [日本語](../../../ja/benchmarks/caltech101-vision-20260924/README.md)

[English index](../../../en/README.md) · [한국어 색인](../../README.md) · [日本語索引](../../../ja/README.md)

이는 L2S1 HTTP 비전 API의 탐색적 제로샷 이미지 분류 실행입니다. 26 옵션의 1토큰 알파벳 제한을 넘어 이전 비전 PR에 도입된 30 옵션 코드 시퀀스 경로를 실행합니다.

<a id="data-and-protocol"></a>
## 데이터 및 프로토콜

- 출처: [Caltech-101 Kaggle 미러](https://www.kaggle.com/datasets/imbikramsaha/caltech-101), 소스 ZIP SHA-256 `c29bfcb9f72b1b03bdfb6b871907ba08ec436bd4049870c9e0424a6f489d4855`.
- 추론 전에 `scripts/prepare_caltech101_vision.py`는 시드 `20260924`를 사용하여 101 객체 클래스의 30를 샘플링한 다음 클래스당 5개의 이미지를 샘플링했습니다. `BACKGROUND_Google`만 제외되었습니다. 샘플 ZIP SHA-256는 `421e1a1ad05a288a57ad5e837a4f22d1941854ca561fe67a2b1be0131de4b1de`입니다.
- 원본 JPEG 바이트는 `image_base64`로 `POST /v1/decisions`에 하나씩 전송되었습니다. 각 요청은 동일한 알파벳순 30 옵션을 사용했습니다. 이미지 파일 이름이나 정답가 모델로 전송되지 않았습니다. 준비 요청이 사용되지 않았습니다.
- 이 미러는 이 실행에 대해 보류 분할을 제공하지 않습니다. 이러한 150 이미지는 제공된 아카이브에서 샘플링되었습니다. 사전 훈련 중복은 알 수 없습니다. 결과는 일반적인 분류 성능이 아닌 고정된 표본과 프롬프트를 측정합니다. 균형 잡힌 무작위 선택 기준선은 1/30(3.33%)입니다.

`selection.json`은 표본에 포함된 모든 파일 이름과 이미지 해시를 고정합니다. `observations.jsonl`에는 HTTP 응답마다 후보 점수 30개, 선택한 옵션, 원시 최상위 옵션, 지연 시간, 판단 보류 이유가 들어 있습니다. 이미지와 모델 가중치는 Git에 포함하지 않습니다.

<a id="runtime"></a>
## 런타임

| 항목 | 값 |
| --- | --- |
| 코드 | `7e33eca`(비전 와이드 코드) |
| 모델 | Gemma 4 E2B IT Q8_0 GGUF, SHA-256 `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a` |
| 프로젝터 | 멀티모달 GGUF, SHA-256 `9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292` 매칭 |
| GPU | 모든 HTTP 응답에서 NVIDIA GeForce RTX 3060, CUDA 오프로드가 확인되었습니다. |
| 추론 | 새로운 요청 컨텍스트, 컨텍스트 4096, 배치 256, 4 CPU 스레드, 모델 로드 모드 `read`, 기본 정책(`min_candidate_mass=0.05`, `min_top_probability=0.8`) |
| 실행 | 한 번에 하나의 로컬 HTTP 요청; 요청 대기 시간에서 시작이 제외됨 |

GPU는 3855 MiB를 사용했으며 실행이 끝날 무렵 74 °C를 보고했습니다. 서버가 5014 ms 이후에 `/healthz`에 도달했습니다. 이는 단일 실행 관찰입니다.

<a id="results"></a>
## 결과

| 지표 | 결과 |
| --- | ---: |
| 이미지/클래스 | 150 / 30 (5 각) |
| 모든 이미지를 수정하세요. | 122/150 (81.33%) |
| 수락률 | 148/150 (98.67%) |
| 승인된 판단 중 정답률 | 122/148 (82.43%) |
| 판단 보류 | 2/150, 탑 이미지의 `low_top_probability` 모두 |
| 원시 top-1, 판단 보류 정책 무시 | 122/150 (81.33%) |
| 로드된 모델 HTTP 대기 시간 | 196.609 ms를 의미합니다. p50 195.282 ms; p95 가장 가까운 순위 200.662 ms |
| 코드 접두사 평가 | 1 per image for all 150 requests |

클래스별 개수는 `summary.json`에 있습니다. 이 샘플에서는 세 가지 클래스(`flamingo_head`, `kangaroo` 및 `okapi`)가 0/5 점수를 받았습니다. 첫 번째 요청은 대기 시간(최대 308.351 ms)에 포함됩니다. 이전의 2등급 고양이/개 점수와의 비교는 암시되지 않습니다. 선택 사항과 표본이 크게 다릅니다.

<a id="reproduce"></a>
## 재현

Kaggle 미러를 ZIP으로 다운로드하고 위의 소스 해시를 확인하세요. CUDA 지원 `l2s1` 바이너리 및 일치하는 GGUF 파일을 사용하여 저장소 루트에서:

```sh
python3 scripts/prepare_caltech101_vision.py --source /path/to/dataset.zip --output /path/to/sample-dir
python3 scripts/benchmark_caltech101_vision.py \
  --selection /path/to/sample-dir/selection.json \
  --sample /path/to/sample-dir/sample-30x5.zip \
  --binary /path/to/l2s1 --model /path/to/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --output /path/to/new-result-dir
```

벤치마크는 기존 출력 디렉터리 재사용을 거부하고 점수를 매기기 전에 샘플 ZIP 및 이미지당 SHA-256 해시를 확인합니다. 원시 관찰, 요약 및 서버 로그를 출력 디렉터리에 기록합니다. 재방송을 비교하려면 모델/프로젝터/소스 해시와 정확한 클래스 순서를 먼저 확인하세요.
