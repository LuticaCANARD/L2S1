<a id="verification-guide"></a>
# 검증 가이드

[English](../en/VERIFICATION.md) · [한국어](VERIFICATION.md) · [日本語](../ja/VERIFICATION.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

사용하려는 정확한 체크포인트, 런타임 및 실행 구성에 대한 검사를 실행하세요. 단위 테스트, 네이티브 계약 테스트 및 레이블이 지정된 작업 평가는 다양한 질문에 답합니다.

<a id="build-and-general-checks"></a>
## 빌드 및 일반 점검

고정된 llama.cpp 개정판은 `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`입니다. CMake FetchContent는 첫 번째 기본 빌드에서 이를 다운로드하고 아카이브 해시를 확인합니다. sys 종속성은 일치하는 소스, 헤더 및 공유 라이브러리를 빌드합니다. CPU 및 CUDA 빌드는 Linux에서 실행됩니다. `llama-metal`는 macOS에서 Metal 장치를 사용하여 동일한 네이티브 API를 빌드합니다. 오프라인 확인을 위해 로컬 체크아웃을 사용하거나 다른 업스트림 개정판을 테스트하려면 `L2S1_LLAMA_CPP_SOURCE`를 설정하십시오.

```sh
cargo fmt --all -- --check
cargo test --locked --offline
cargo test --release --locked --offline --features llama
cargo clippy --release --locked --offline --all-targets --features llama -- -D warnings
python3 -m unittest discover -s scripts -p 'test_*.py'
```

GPU 검사의 경우 `--features llama-cuda`로 빌드하고 네이티브 테스트에는 `SKID_CUDA=1`를 사용합니다. CPU 전용 빌드는 CUDA 요청을 거부합니다.

Mac에서는 `cargo build --release --locked --features wgpu --bin l2s1-wgpu` 및 `cargo build --release --locked --features llama-metal --bin l2s1`를 사용하여 Metal 경로를 모두 구축합니다. Gemma 4의 경우 `WGPU_BACKEND=metal --require-metal`로 첫 번째를 실행하고 일반 GGUF 모델의 경우 `--device metal`로 두 번째를 실행합니다. 실제 Metal 추론과 동등성을 위해서는 일치하는 GGUF 파일이 있는 Mac이 필요합니다. Gemma 4 요청-로컬 모드가 동등한 경우 `L2S1_WGPU_GEMMA_MODEL` 및 `L2S1_WGPU_GEMMA_MMPROJ`를 설정한 다음 `cargo test --release --locked --features wgpu --test wgpu_execution -- --ignored`를 실행합니다.

CUDA의 다른 GGUF 제품군의 경우 새 모델 경로와 동일한 바이너리를 사용합니다. 작업 품질을 비교하기 전에 `--inspect`, `--preflight --input examples/warehouse.json` 및 실제 판단 요청을 실행하세요. 아래의 다중 모델 적합성 테스트에서는 콜론으로 구분된 GGUF 경로와 `SKID_CUDA=1`를 허용합니다. 성공적인 로드 또는 유한 후보 확률 질량은 레이블이 지정된 작업의 정확성이 아니라 이 판단 경로와의 호환성을 설정합니다.

직접 정지 이미지 입력의 경우 호환 가능한 비전 GGUF 및 이에 맞는 `mmproj` GGUF를 제공하세요. 무시된 테스트는 두 개의 서로 다른 PNG를 사용하고, 이미지 콘텐츠가 원시 logits에 영향을 미치는지 확인하고, 유효하지 않은 이미지 이후의 복구를 확인합니다. 작업 정답률을 설정하지 않습니다.

```sh
SKID_VISION_MODEL=/path/to/vision-model.gguf \
SKID_VISION_MMPROJ=/path/to/mmproj.gguf \
cargo test --locked --offline --features llama --test vision -- --ignored
```

허용된 CUDA 호스트에서 동일한 계약에 대해 `--features llama-cuda` 및 `SKID_CUDA=1`를 사용하세요. HTTP 경로에는 `/healthz` 및 `/v1/capabilities`의 실시간 루프백 확인, `media`라는 이름의 유효한 `POST /v1/decisions` 및 HTTP를 반환하는 잘못된 형식의 요청이 추가로 필요합니다. 400와 `error.code` 및 `error.request_id`.

Offline Cargo 명령에는 이전에 채워진 CMake 소스 캐시 또는 로컬 체크아웃으로 설정된 `L2S1_LLAMA_CPP_SOURCE`도 필요합니다. 일반 테스트는 모델 파일이 필요한 테스트를 건너뛰고 실행합니다. 가중치를 다운로드하거나 실제 모델 호환성을 설정하지 않습니다. 업스트림 C++ 도우미 경고는 Rust 린트 결과와 별개입니다.

<a id="model-contract-checks"></a>
## 모델 계약 확인

```sh
L2S1_CONFORMANCE_MODELS=/path/to/model-a.gguf:/path/to/model-b.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/l2s1-conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

CUDA에 대해 `SKID_CUDA=1`를 설정합니다. 적합성 제품군은 의미 체계 ID, 모든 판단 종류, 토큰 매핑, 아티팩트 바인딩, 컨텍스트 오류, 복구, 스냅샷 제한 및 실행 진단을 확인합니다. 0.02의 기존 확률/질량 허용 오차, 변경되지 않은 상위 ​​선택 및 변경되지 않은 허용 결과를 사용하여 최적화된 모드를 새로운 실행과 비교합니다. 병렬 차이는 별도로 보고됩니다. 완료된 보고서는 자동으로 동등성 합격이 아닙니다.

CUDA 호스트에서 검증된 하이브리드 Bonsai 경로의 경우 실제 모델 상태 복원 회귀를 실행합니다.

```sh
L2S1_BONSAI_MODEL=/path/to/Bonsai-27B-Q1_0.gguf \
  cargo test --release --locked --features llama-cuda \
  --test state_restore_bonsai -- --ignored --nocapture
```

새로운 점수 및 정책 판단, 실제 재사용된 토큰, 스냅샷 바이트 제한 및 강제 대체 후 복구에 대해 16-판단 공유 상태 요청을 확인합니다. 이는 일반 작업 정답률이 아닌 하나의 픽스처에 대한 실행 동등성입니다.

특정 실행 경로에 대해 [접두사 재사용 검사](SEMIF_ALGORITHM.md#reproduce) 및 [병렬 계약 검사](PARALLEL_EXECUTION.md#validation-and-measurement)를 사용하세요. 병렬 모드에는 수치적 차이가 있으며 여전히 선택되어 있습니다. 사전 검증를 로드하거나 통과하는 모델에는 여전히 추론 및 레이블이 지정된 워크로드 평가가 필요합니다.

<a id="optional-optimization-checks"></a>
## 선택적 최적화 확인

```sh
SKID_MODEL=/path/to/model.gguf \
  cargo test --release --locked --offline --features llama \
  --test native_compact --test optimization_contract --test shared_state \
  --test worker_native -- --include-ignored --test-threads=1

cargo run --release --locked --offline --features llama \
  --example benchmark_optimizations -- \
  --model /path/to/model-a.gguf --model /path/to/model-b.gguf \
  --output /tmp/l2s1-optimizations.json --repeats 3
```

각 체크포인트에 대해 계약 확인을 반복합니다. CUDA를 측정할 때 테스트에는 `SKID_CUDA=1`를 사용하고 벤치마크에는 `--cuda`를 사용합니다. 계약에는 정확한 캐시된 토큰 준비, 동적 지침 및 옵션 순서, 압축/전체 증거 동등성, 세션 오류 정리 및 네이티브 작업자 티켓 상관 관계가 포함됩니다. 컴팩트 전송은 전체 어휘 노멀라이저를 유지합니다. Rust에서 호스트 측 버퍼 복사본을 제거합니다. 어휘 프로젝션이나 GPU에서 호스트로의 전송은 제거하지 않습니다.

벤치마크는 짧고 확장된 합성 상태로 1, 4 및 16 질문을 측정하고, 실행 순서를 회전하고, 로드를 제외하고, 원시 응답, ID, 단계 타이밍 및 캐시 통계를 기록합니다. 캐시 측정에서는 준비 후 의도적으로 반복 입력을 사용합니다. 공유 상태 측정은 한 세션 내에서 다양한 질문이 포함된 별도의 호출을 발행합니다. 각 최적화를 동일한 레이아웃의 새로운 기준과 비교합니다. 상태 우선 프롬프트는 재사용과 관계없이 예측을 변경할 수 있습니다. 질문당 분할 평균 시간은 독립형 요청 대기 시간이 아니며 이러한 워크로드는 작업 정답률 또는 운영 환경 처리량을 설정하지 않습니다. 보고서는 덮어쓰기를 거부하고 무시된 로컬 출력 디렉터리에 속합니다.

<a id="task-quality-and-evidence"></a>
## 작업 품질 및 증거

- [합성 벤치마크](BENCHMARK.md): 규칙 픽스처에 대한 정확성, 판단 보류, 일관성 및 대기 시간.
- [AG 뉴스 평가](KAGGLE_BENCHMARK.md): 동결 분류 프로토콜.
- [JevBench 평가](JEVBENCH.md): 공개 작업 매핑, 업스트림 채점 및 별도의 승인 지표.

무시된 로컬 출력 디렉터리의 원시 예측 및 레이블과 함께 체크포인트, 런타임, 프롬프트 및 구성 ID를 유지합니다. 대기 시간, 수락률, 수락된 판단의 정답률 및 후보 간 상대 신뢰도를 별도로 유지하세요. 생성된 보고서 및 호스트별 실행 계획은 소스 문서가 아닙니다. README에는 간단한 모델 비교 요약이 포함되어 있습니다. 이는 공식 순위표나 운영 환경-정답률 주장이 아닙니다.
