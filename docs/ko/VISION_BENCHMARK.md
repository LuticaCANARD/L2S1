<a id="direct-vision-http-benchmark"></a>
# 직접 비전 HTTP 벤치마크

[English](../en/VISION_BENCHMARK.md) · [한국어](VISION_BENCHMARK.md) · [日本語](../ja/VISION_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

이는 2026년 9월 24일 `lucatagpu`(RTX 3060 12 GiB, 드라이버 595.71.05)에서 측정한 **로컬 합성 스모크 벤치마크**입니다. 같은 CUDA 지원 L2S1 실행 파일, Gemma 4 E2B Q8_0 GGUF, 대응 `mmproj`, 요청 스키마, 호스트의 CPU·CUDA 경로를 비교합니다. 모델·프로젝터 SHA-256은 [CPU 원시 보고서](../../benchmarks/vision-20260924/cpu.json)와 [CUDA 원시 보고서](../../benchmarks/vision-20260924/cuda.json)에 기록되어 있습니다. 모델 가중치는 저장소에 없습니다.

[실행기](../../scripts/benchmark_vision_http.py)는 장치마다 격리된 루프백 HTTP 서버를 시작하고 `/healthz`를 기다린 뒤 측정에서 제외하는 워밍업 요청 4회를 보냅니다. 이후 저장소의 빨강·파랑 64×64 PNG를 번갈아 사용해 직렬 `POST /v1/decisions` 30회의 시간을 잽니다. JSON·base64는 측정 전에 준비합니다. 측정 구간에는 HTTP 요청·응답 전송, 이미지 인코딩, 모델 추론, 점수 계산, 응답 직렬화가 포함됩니다. 각 실행은 로드된 모델 하나로 처리하며 병렬 클라이언트나 요청 간 배치는 측정하지 않습니다. 둘 다 컨텍스트 2048, 배치 256, CPU 스레드 4개, `--model-load-mode read`를 사용했습니다. GPU 실행을 CPU보다 먼저 수행했습니다.

| 장치 | p50 | p95 최근접 순위 | 평균 | 시작부터 `/healthz`까지 | 픽스처 선택 | 로드 중 GPU 메모리 | 프로세스 최대 RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 3060 CUDA | 96.054 ms | 96.236 ms | 96.021 ms | 5.013 s | 30/30 | 3,813 MiB | 3.47 GiB |
| 같은 호스트의 CPU | 3,066.295 ms | 3,162.112 ms | 3,079.159 ms | 5.263 s | 30/30 | 20 MiB | 5.94 GiB |

이 반복되는 두 이미지 작업의 CPU/CUDA p50 비율은 **31.9×**였습니다. 30회 선택은 라벨이 있는 단색 이미지 두 개의 반복입니다. 30/30은 인터페이스 스모크 확인이며 독립 예제 30개나 일반적인 비전 정답률 결과가 아닙니다. 프로세스 RSS와 GPU 메모리는 다른 측정값이므로 더하거나 총 시스템 메모리로 취급할 수 없습니다. 시작 시간에는 프로세스 시작, 모델·프로젝터 로드, 준비 상태 확인이 포함됩니다. 지연 시간 열은 시작과 워밍업을 제외합니다. 이 단일 직렬 실행은 동시 처리량이나 운영 지연 시간을 보장하지 않습니다.

<a id="reproduce"></a>
## 재현

`--features llama-cuda`로 L2S1을 빌드하고 대응하는 비전 GGUF와 `mmproj`를 준비합니다. 빌드와 맞는 네이티브 공유 라이브러리를 `LD_LIBRARY_PATH`로 사용할 수 있게 하세요. 두 장치 모드에 같은 실행 파일을 사용합니다. 실행기는 기존 보고서를 덮어쓰지 않으며 모든 워밍업·측정 관측값과 모델·프로젝터·실행 파일·이미지 해시를 저장합니다.

```sh
python3 scripts/benchmark_vision_http.py \
  --binary target/release/l2s1 \
  --model /path/to/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --red-image tests/fixtures/vision_red_64.png \
  --blue-image tests/fixtures/vision_blue_64.png \
  --device cuda --warmup 4 --iterations 30 \
  --output results/vision-http-cuda.json
```

다른 출력 경로와 사용 가능한 `--listen` 주소로 `--device cpu`를 반복합니다. 저장소의 보고서는 PR #12로 분리하기 전 비전 구현에서 생성되었습니다. 실행 파일 SHA-256이 해당 빌드를 식별합니다. PR #12 소스는 격리된 CPU 계약 테스트를 통과하지만, 이 시간 수치는 기록된 호스트 빌드와 설정에 관한 것입니다.
