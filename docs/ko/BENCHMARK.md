<a id="local-model-comparison-benchmark"></a>
# 국내 모델 비교 벤치마크

[English](../en/BENCHMARK.md) · [한국어](BENCHMARK.md) · [日本語](../ja/BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

`decision-rules-v1` 벤치마크는 **12 요청의 기존 GGUF 체크포인트를 36 라벨이 붙은 판단**와 비교합니다. 규칙 준수 정확성, 판단 보류, 반복 출력 일관성 및 추론 대기 시간을 측정합니다. 모델을 다운로드하거나 배포하지 않습니다.

기본값은 레거시 v1 프롬프트입니다. 선택적 v2 상태 우선 프롬프트 및 선택적 접두사 재사용은 [SEMIF_ALGORITHM.md](SEMIF_ALGORITHM.md)에 문서화되어 있습니다. 비교를 위해 별도의 출력 디렉터리를 사용하여 v2의 경우 `--prompt-layout state-first`를 전달한 다음 `--execution-mode fresh`(기본값) 또는 `--execution-mode prefix-reuse`를 Rust 실행기에 전달합니다. 보고서는 요청된 모드, 논리적 입력 토큰, 재사용된 접두사 토큰 및 실제 평가된 토큰을 기록합니다. 동일한 프롬프트 버전, 모델, 장치 및 배치 설정을 비교합니다. 프롬프트를 변경하면 캐시 재사용과 관계없이 정답률이 변경될 수 있습니다.

`tests/fixtures/decision_benchmark.json`의 영어 픽스처는 두 가지 도메인을 다룹니다.

- 창고: 0, 6, 7, 24, 25 및 48 시간에 보관 선택, 저온 유통 요건 및 파견 우선순위.
- 액세스 제어: 역할 간 매핑, 부울 편집 권한, 0, 1, 2, 3 및 4 시도 시 시도 실패 검토 우선순위.

모든 요청에는 하나의 선택 사항, 하나의 바이너리, 하나의 서열 판단이 있습니다. 선택 옵션 순서는 다양합니다. 서열 값은 API의 요구에 따라 정렬된 상태로 유지됩니다. 일부 경우에는 관련 없는 메모가 포함되어 있습니다. 예상되는 답변은 A/B/C 토큰 위치와 무관한 의미론적 옵션 ID입니다. 일반 Rust 테스트는 시나리오 규칙의 모든 레이블을 독립적으로 다시 계산합니다.

이는 일반적인 추론, 코딩, 다국어, 안전 또는 운영 환경-정답률 평가가 아닌 소규모 합성 규칙 준수 벤치마크입니다. 이러한 레이블을 조정하지 말고 결과 숫자를 보류된 정답률로 표시하지 마십시오. `tests/native.rs`의 실제 모델 일관성 테스트는 별도로 유지됩니다.

<a id="run-the-five-model-matrix"></a>
## 5개 모델 매트릭스 실행

전제 조건: 이미 캐시된 Rust 종속성, CMake, C++17 컴파일러 및 로컬에서 획득한 체크포인트. 번들 sys 종속성은 고정된 llama.cpp 소스를 빌드합니다. CUDA 실행에는 추가로 CUDA 툴킷이 필요합니다. 모델 경로는 `tests/fixtures/benchmark_models.json`에 나열되어 있습니다. 러너는 Hugging Face 클라이언트, 계정 또는 네트워크 요청을 사용하지 않습니다. 모델을 구입하기 전에 별도의 라이선스를 검토하세요.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools benchmark-models \
  --device cpu cuda \
  --iterations 3 \
  --warmup 1
```

Runner는 요청 시 `--locked --offline` 및 CUDA 기능을 사용하여 릴리스 테스트 실행 파일을 한 번 빌드하고 각 모델을 해시하며 한 번에 하나의 모델/장치를 실행합니다. CPU가 기본 장치입니다. CUDA는 명시적이며 자동으로 CPU로 대체될 수 없습니다. 네이티브 백엔드는 전체 실행에 대해 하나의 로드된 모델을 재사용합니다. Rust 프로세스 시작, 화물 편집, 적재 및 워밍업은 정상 상태 타이밍에서 제외됩니다.

하위 집합을 선택하거나 설정을 변경합니다.

```sh
target/release/l2s1-tools benchmark-models \
  --model gemma4 --model qwen3 \
  --device cuda \
  --iterations 10 --warmup 2 \
  --context 2048 --batch 256 --threads 4 \
  --min-top-probability 0.8 --min-candidate-mass 0.05 \
  --timeout 1800 \
  --output results/benchmark/my-comparison
```

`--iterations`는 모든 12 요청에 대해 전체 측정 통과를 의미합니다. `--warmup`는 별도의 시간이 설정된 첫 번째 요청에 추가로 완전히 제외된 패스를 의미합니다. 케이스 순서는 측정된 패스 간에 결정론적으로 회전합니다. 반복은 타이밍과 일관성을 특성화하는 데 도움이 됩니다. 독립적인 정답률 예제의 수는 증가하지 않습니다.

오래된 보고서를 현재 결과로 착각할 수 없도록 출력 디렉터리는 새로운 것이어야 합니다. 기본값은 무시된 `results/benchmark/` 아래의 타임스탬프 디렉터리입니다. `--timeout`는 워밍업을 포함하여 각 모델/장치에 별도로 적용됩니다. 누락된 파일, 네이티브 오류 및 시간 초과는 실패한 행으로 계속 표시됩니다. 실행기는 다른 모델을 계속 사용하고 실행이 실패하면 0이 아닌 종료 코드를 반환합니다. 낮음 정답률은 테스트 실패가 아닌 측정으로 보고됩니다.

사용자 정의 체크포인트의 경우 `--manifest /path/to/models.json`를 전달합니다.

```json
{
  "models": [
    {"id": "my-model", "path": "/absolute/path/to/chat-model.gguf"}
  ]
}
```

상대 체크포인트 경로는 사용자 정의 매니페스트를 포함하여 저장소 루트에 대해 확인됩니다. ID는 문자, 숫자, 하이픈, 밑줄만 포함하는 고유한 소문자 이름이어야 합니다. 벤치마크는 백엔드의 자동 프롬프트 프로필을 사용하므로 모델은 해당 템플릿을 통해 동일한 의미 작업을 받습니다. 템플릿/토큰 수와 모델 크기를 비교하세요.

<a id="reports-and-metric-definitions"></a>
## 보고서 및 측정항목 정의

각 디렉터리에는 `summary.md`, `summary.json`, 빌드 로그, 모델/장치별 JSON 및 네이티브 로그가 포함됩니다. 체크포인트 SHA256, 픽스처 SHA256, 실행 가능한 SHA256, 사용 가능한 저장소/llama.cpp 개정, 장치, 양자화 설명, 설정, 정책, 개별 레이블, 예측, 확률 및 대기 시간 샘플이 유지됩니다. 원시 로컬 보고서에는 로컬 모델 경로가 포함될 수 있습니다. 게시하기 전에 검토하세요.

| 지표 | 정의 |
| --- | --- |
| 수락률 | 받아들인 판단 / 측정된 모든 판단 |
| 판단 보류 환율 | 판단 보류된 판단 / 신중한 모든 판단 |
| 수락된 판단의 정답률 | 승인된 판단/승인된 판단을 수정합니다. `null` / `n/a`(아무 것도 허용되지 않는 경우) |
| 수락된 정답 / 전체 | 승인된 판단/모든 판단을 수정합니다. 판단 보류은 올바른 것으로 간주되지 않습니다 |
| 원시 상단-1 정답률 | 정책 판단 보류 이전에 레이블 대비 가장 높은 점수를 받은 의미론적 옵션; 동점이 잘못되었습니다 |
| 올바른 허용됨 | 올바른 승인된 판단/측정된 추론 시간(초) |
| 판단/초 | 판단 보류을 포함한 모든 완료된 판단/측정된 추론 시간(초) |
| p50/p95 요청 | 전체 `decide` 호출의 가장 가까운 순위 백분위수(각 호출에는 세 가지 순차적 판단이 포함됨) |
| 입력 토큰/초 | 처리된 입력 토큰 / 추론 시간(초). 생성 토큰/초와 구분함 |
| 반복 일관성 | 허용된 출력 및 원시 상위 1 변경 사항과 동일한 사례/판단에 대한 첫 번째 패스 비교 |

품질 개수 및 분수도 도메인 및 판단 종류별로 표시됩니다. 전체 샘플은 사례별 검사를 허용합니다. 반복 출력은 점수가 변경되는 동안 동일한 선택된 옵션을 유지할 수 있습니다. 이 보고서는 네이티브 제품군의 확률/일괄 검사를 대체하지 않습니다.

로드 및 첫 번째 요청은 별도로 보고됩니다. 타이밍에는 신속한 준비, 사전 작성, logits 전송 및 채점이 포함되지만 JSON 보고서 직렬화는 제외됩니다. OS 페이지 캐시는 플러시되지 않습니다. 로드 시간은 콜드 디스크 측정이 아닙니다. 전체 컨텍스트 용량, 프로세스 RSS, 최대 VRAM 및 동시 제공 처리량은 측정되지 않습니다. 하드웨어, 양자화, 컨텍스트, 배치, 스레드 수 및 정책은 의미 있는 비교를 위해 일관성을 유지해야 합니다. 임의의 하드웨어 속도 임계값은 없습니다.

<a id="test-the-benchmark-without-models"></a>
## 모델 없이 벤치마크 테스트

```sh
cargo test --locked --offline --test benchmark
cargo test --locked --offline -p l2s1-tools
```

이는 동점 및 완전 판단 보류 출력을 포함하여 픽스처 레이블 및 측정 단위 분모를 검증하고 매니페스트 검증 및 오류 보존 보고서를 제공합니다. 그들은 실제 추론 정확성을 확립하지 않습니다.

실제 모델 벤치마크는 `tests/benchmark.rs`에서 무시된 Rust 테스트입니다. 직접 실행할 수도 있습니다.

```sh
SKID_MODEL=/absolute/path/to/model.gguf \
SKID_CUDA=1 \
SKID_BENCH_ITERATIONS=3 \
SKID_BENCH_WARMUP=1 \
SKID_BENCH_OUTPUT=results/benchmark/single-model.json \
  cargo test --release --locked --offline --features llama --test benchmark \
  native::model_decision_benchmark -- --exact --ignored --nocapture
```

직접 실행에서는 `SKID_CONTEXT`, `SKID_BATCH`, `SKID_THREADS`, `SKID_MIN_TOP_PROBABILITY` 및 `SKID_MIN_CANDIDATE_MASS`를 허용합니다. Rust 실행기는 체크포인트 해시와 비교 테이블을 추가합니다. 이전 `tests/performance.rs`는 단일 창고 예의 반복 측정에 계속 사용할 수 있습니다.
