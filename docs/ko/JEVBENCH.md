<a id="jevbench-public-evaluation"></a>
# JevBench 공개 평가

[English](../en/JEVBENCH.md) · [한국어](JEVBENCH.md) · [日本語](../ja/JEVBENCH.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

Rust `l2s1-tools jevbench-public` 명령은 기존 `evaluate_jsonl` 예제를 공용 [JevBench](https://github.com/fstandhartinger/jevbench) 작업에 연결합니다. 공개 채점자는 검토된 업스트림 개정을 따릅니다. 이전에 기록된 Gemma 4 E2B Q8_0 및 다중 모델 매트릭스 결과는 과거 Python 어댑터에서 나왔습니다. 새로운 실행에서는 Rust를 사용합니다.

검토된 업스트림 개정은 `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`입니다. 준비는 다른 개정이나 더러운 업스트림 체크아웃을 거부합니다. 게시된 세 개의 JSONL 파일에는 231 판단(쉬운 48, 원본 72, 하드 111)이 포함되어 있습니다. 이는 전체 534 항목 평가나 공식 리더보드 제출이 아닌 공개 하위 집합 평가입니다.

<a id="reproduce"></a>
## 재현

요구 사항: 고정된 개정판의 깔끔한 업스트림 체크아웃, 호환 가능한 GGUF 및 CUDA 지원 프로젝트 빌드. 평가자는 CUDA 오프로드를 검증합니다. 특정 GPU를 추가로 요구하려면 `--expected-gpu 'RTX 3060'`(또는 의도된 장치 이름)를 사용하십시오. 하네스 및 공용 작업 파일은 업스트림 MIT 공지를 유지합니다. 모델 용어는 별도로 유지됩니다.

```bash
# On a CUDA host with the Rust and native toolchains configured:
cd /path/to/L2S1
cargo build --release --locked --features llama-cuda --bin l2s1 --example evaluate_jsonl
cargo build --release --locked -p l2s1-tools

git clone https://github.com/fstandhartinger/jevbench.git ../upstream-new
git -C ../upstream-new checkout --detach f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84

target/release/l2s1-tools jevbench-public run \
  --jevbench ../upstream-new \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --context 8192 --expected-gpu 'RTX 3060' --output ../run-gemma4-new
```

출력 디렉터리가 아직 존재하지 않아야 합니다. 외부 추론 API, API 키 또는 청구 가능한 서비스는 사용되지 않습니다.

CPU/GPU 중량 배치의 경우 하네스는 `--gpu-layers N`, `--cpu-moe-layers N` 및 `--threads N`도 허용합니다. RTX 3060의 Gemma 4 26B A4B Q4 체크포인트의 경우 `--cpu-moe-layers 18 --threads 8 --context 8192`를 사용합니다. 매니페스트는 이러한 설정을 기록하고 하네스는 보고된 배치를 확인합니다. 이러한 옵션을 생략하면 기본 배치와 4스레드 설정이 유지됩니다. 옵션인 `--model-load-mode read`는 모델 로딩만 변경합니다. 기본 `auto`는 자동 로딩을 유지합니다. 요청된 모드가 기록되고 확인됩니다.

<a id="mapping-and-measurement-contract"></a>
## 매핑 및 측정 계약

- `state`, 지침 및 판단 기준만 추론에 전달합니다. 예상 답변, 근거, 골드 분포 및 출처를 별도의 채점 파일에 보관하세요.
- 정식 `labels` 순서를 유지합니다. `choice`를 정렬된 옵션에 매핑하고, `noul`를 false/true에 매핑하고 확률은 다시 no/yes에 매핑하며, `score`를 숫자 서열 수준에 오름차순으로 매핑합니다.
- 모델의 후보 간 상대 확률을 직접 사용합니다. 공식 argmax 점수는 프로젝트의 기본 판단 정책이 판단 보류하는 경우에도 계산됩니다. 해당 정책의 수락된 판단의 정답률, 잘못된 허용 답변 및 수락률를 별도로 보고합니다.
- Rust 득점자는 고정된 업스트림 `score_task` 및 `summarize` 정의를 따릅니다. Brier는 두 바이너리 레이블을 모두 포함하여 전체 다중 클래스 합계를 사용합니다. ECE는 10개의 동일 너비 신뢰 구간을 사용합니다. 저장된 공개 예측 및 집계 결과를 업스트림 Python 채점기와 비교했습니다.
- 기본 구성: 레거시 프롬프트, 새로운 실행, 컨텍스트 8,192, 배치/마이크로배치 256, 4개의 스레드, 한 번에 하나의 요청, FlashAttention 꺼짐. LoRA, 출력 헤드, 보정 아티팩트 또는 이러한 항목에 대한 교육이 없습니다.
- 입력 오버플로는 오류로 남아 있으며 잘리지 않습니다. 추론 실패를 유지합니다. 누락, 중복 또는 알 수 없는 결과 ID 및 후보 매핑 불일치를 거부합니다.
- 로드 및 하나의 워밍업이 제외된 로컬 Rust 추론 호출 대기 시간을 보고합니다. 여기에는 신속한 준비 및 추론이 포함되지만 HTTP/네트워크 경로는 제외됩니다. 공식 보드의 원격 엔드포인트 대기 시간이 아닙니다.
- 로컬 비용을 알 수 없습니다(`null`). 0이 아닙니다. 전체 데이터 세트, 공식 배포 대기 시간 및 지원되는 가격 기준을 사용할 수 없으므로 전체 JevBench 점수 또는 순위가 할당되지 않습니다.

출력에는 정확한 추론 요청, 금이 포함된 원래 작업, 원시 예측, 정규화된 JevBench 레코드, 공개 계층/제품군별 공식 지표 요약, 선택적 정책 결과, 로그, 명령 및 데이터/모델/평가자 해시가 포함됩니다. `native_candidate_softmax`는 확률 소스를 식별합니다. 이러한 값은 보정된 정확성 확률로 주장되지 않습니다.

매핑 테스트:

```bash
python3 -m unittest discover -s scripts -p 'test_jevbench_public.py'
```

<a id="local-rtx-3080-rerun-2026-09-23"></a>
## 로컬 RTX 3080 재실행(2026-09-23)

새로운 5개 모델 실행에서는 16,384 및 `--expected-gpu 'RTX 3080'` 컨텍스트를 사용했습니다. 모든 1,155 예측은 추론 오류나 잘림 없이 완료되었으며 정답률, Brier, ECE 및 판단 보류 총계의 독립적인 저장된 예측 재계산을 통과했습니다. Gemma 4 E2B Q8_0은 159/231(68.83%)를 획득했습니다. Gemma 3 1B Q8_0 93/231, TinyLlama 1.1B Q4_K_M 77/231, Qwen3 0.6B Q8_0 73/231 및 SmolLM2 135M Q8_0 71/231. 이 점수는 기본 판단 보류 정책 이전에 argmax를 사용합니다.

공개 데이터 세트 및 채점 모듈은 [Open-Jev 보고서](https://zefan-cai.github.io/open-jev/benchmarks/)에서 참조하는 개정판과 바이트가 동일합니다. 이는 로컬 GGUF를 사용하여 L2S1를 평가합니다. Open-Jev의 훈련된 체크포인트가 실행되지 않았습니다. 로컬 전체 보고서(로컬 아티팩트: `results/jevbench-local-20260923T083530Z/REPORT.md`, 커밋되지 않음)는 계층 점수, 선택적 정답률, 타이밍 제한 및 재생 명령을 기록합니다. 원시 예측, 소스 스냅샷 및 재생 증거는 무시된 결과 디렉터리의 해당 보고서 옆에 남아 있습니다. 이는 아래의 RTX 3060 매트릭스와는 별개입니다.

<a id="prompt-order-diagnostics-on-that-rtx-3080-run"></a>
### 해당 RTX 3080 실행에 대한 즉각적인 진단

4개의 사후 진단 스트림은 JSON 필드 순서(`state-first`)를 변경하거나 해당 ID와 판단 기준 쌍을 사용하여 139 Choice 옵션 목록을 반전하는 동안 각 모델, 공개 작업 세트 및 추론 설정을 고정된 상태로 유지했습니다. 원래 모델 점수는 위의 5개 모델 실행의 기준입니다. 아래 정답은 의미 ID이므로 표시된 옵션 변경 위치 자체는 변경된 답변으로 간주되지 않습니다.

| 모델 | 변화 | 기준선 / 231 | 새 측정 / 231 | 의미론적 답변이 변경되었습니다. | 오답→정답 / 정답→오답 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8_0 | 먼저 상태 | 73 | 78 | 73/231 | 29 / 24 |
| Qwen3 0.6B Q8_0 | 선택 순서가 바뀌었습니다. | 73 | 76 | 131/139 | 29 / 26 |
| Gemma 4 E2B Q8_0 | 먼저 상태 | 159 | 151 | 50/231 | 15 / 23 |
| Gemma 4 E2B Q8_0 | 선택 순서가 바뀌었습니다. | 159 | 157 | 33/139 | 9 / 11 |

추론 오류나 잘림 없이 4개 스트림이 모두 완료되었습니다. 이는 프롬프트 및 옵션 표시가 이러한 체크포인트의 출력을 변경할 수 있다는 증거입니다. 이는 독립적으로 지속되는 개선 테스트나 상태 우선이 다른 모델에 더 낫다는 주장이 아닙니다. 로컬의 gitignored `results/jevbench-diagnosis-20260923/REPORT.md`는 전체 진단, 명령 및 저장된 예측을 유지합니다. 레거시 순서는 기본값으로 유지됩니다.

<a id="multi-model-comparison"></a>
## 다중 모델 비교

[기록된 모델 비교](MODEL_RESULTS.md#recorded-model-comparison)는 완료된 9월 23 RTX 3060 매트릭스: 22 GGUF 체크포인트를 요약합니다. 231 항목, 완료된 실행에 오류가 없습니다. 23 런타임 구성이 있었습니다. CUDA 그래프가 활성화되어 GPT-OSS가 실패하고 `GGML_CUDA_DISABLE_GRAPHS=1`로 완료되었습니다. 원래 실패한 시도는 증거로 남아 있습니다. 지연 시간 값은 해당 설정에 속하며 별도의 실행과 혼합되어서는 안 됩니다.

전체 보고서, CSV, 원시 예측, 모델 계획, 런타임 해시 및 측정된 소스 스냅샷은 로컬의 gitignored `results/jevbench-matrix-20260923/` 디렉터리에 있습니다. 이 저장소에는 포함되지 않습니다. Qwen3.5-4B Q8_0은 184/231(79.65%)에서 이 선택된 매트릭스를 이끌었습니다. 모델 다운로드가 완료된 후 전체 재실행은 해당 모델 및 Gemma4 E2B에 대한 모든 확률을 정확하게 재현했습니다. 스냅샷은 후속 작업 트리 변경과 관계없이 이 비교에 사용된 고정된 빌드를 기록합니다.

`l2s1-tools jevbench-matrix download/run`는 고정된 계획을 다운로드하거나 공개 평가자를 통해 사용 가능한 체크포인트를 순차적으로 실행합니다. `l2s1-tools report-jevbench-matrix`는 골드 라벨에 대해 저장된 예측을 다시 계산하고 보고서를 생성하기 전에 공유 요청/평가자 해시를 확인합니다. 이를 재생성하려면 로컬 모델 계획, 전체 모델별 실행 디렉터리 및 `matrix-status.json`가 필요합니다. 생성된 보고서, 실행 아티팩트 및 호스트별 계획은 소스 버전으로 관리되지 않고 로컬로 유지됩니다.

<a id="gemma-4-26b-a4b-separate-run-2026-09-23"></a>
## Gemma 4 26B A4B 별도 실행(2026-09-23)

UD-Q4_K_M 체크포인트(SHA-256 `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`)는 18 CPU 전문가 레이어, 8개 스레드, 컨텍스트가 포함된 RTX 3060에서 실행되었습니다. 8192 및 배치/ubatch 256. 고정된 공개 득점자는 196/231 정답(84.85%): Easy 48/48, Original 70/72, Hard 78/111로 계산했습니다. 기본 정책에서는 191가 정확하고 29가 잘못되었으며  11가 판단 보류된 220 판단을 수락했습니다. Brier는 0.278098 및 ECE 0.132128였습니다. 추론 오류나 잘린 입력이 없었습니다.

측정된 신규/레거시 추론 p50/p95는 787.01/8814.01 ms였으며 모델 로드와 하나의 워밍업을 제외했습니다. 이 실행에서는 매트릭스와 동일한 공개 작업 ID를 사용했지만 다른 평가기 빌드 및 요청 직렬화를 사용했습니다. 지연 시간은 원래 매트릭스 또는 이후 31B 실행과 제어된 비교가 아닙니다. 저장된 예측은 독립적으로 계산되었습니다. 로컬의 gitignored `results/jevbench-gemma26-20260923T132214Z/` 폴더에는 소스, 모델, 요청, 평가자 및 예측 해시와 원시 증거가 보관됩니다.

<a id="gemma-4-rebuild-confirmation-2026-09-23"></a>
## Gemma 4 재구축 확인(2026-09-23)

현재 Rust 소스와 네이티브 C++ 브리지는 고정된 llama.cpp CUDA 라이브러리를 재사용하여 RTX 3060 호스트의 별도 디렉터리에서 새로 컴파일되었습니다. Gemma 4 E2B Q8_0, E4B Q8_0 및 E4B Q4_K_M은 각각 위의 기본 설정을 사용하여 동일한 공개 231 항목을 완료했습니다. 이들은 각각 157/231(67.97%),  177/231(76.62%) 및 179/231(77.49%) 점수를 받았습니다. 모든 693 후보 확률 벡터와 판단 값은 이전 RTX 3060 행렬과 정확히 일치했습니다. 추론 오류나 잘림이 발생하지 않았습니다.

재구축 보고서(로컬 아티팩트: `results/gemma4-rebuild-20260923T103912Z/REPORT.md`, 커밋되지 않음)는 전후 비교, 대기 시간, 소스 및 바이너리 해시, 빌드 로그를 기록합니다. 기본 테스트는 49/49를 통과했습니다. 라마 기능 모음은 18를 무시하고 59 테스트를 통과한 후 별도로 실행된 Gemma 4 CUDA 통합 테스트와 6개의 매핑 테스트를 통과했습니다. 이러한 결과에는 고정된 소스 및 기준 경로가 포함됩니다. 이번 재실행에는 선택적 최적화 모드가 활성화되지 않았습니다.
