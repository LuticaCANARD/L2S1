<a id="layajev-task-adapter-and-cache-evaluation"></a>
# Laya/Jev 작업 어댑터 및 캐시 평가

[English](../en/LAYA_BENCHMARK.md) · [한국어](LAYA_BENCHMARK.md) · [日本語](../ja/LAYA_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

Rust `l2s1-tools laya-benchmark` 명령은 로컬 L2S1 JSONL 평가기와 함께 [Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark)에서 사용되는 공용 작업을 평가합니다. Laya를 로드하거나 평가 레이블을 학습하거나 별도의 JevBench 종합 점수를 생성하지 않습니다.

| 평가 묶음 | 고정 평가 데이터 | 범위 |
| --- | --- | --- |
| `phish` | `AreLit/PhishNChips`, `core_emails.csv` | 2,000 emails, balanced binary labels |
| `typed` | `LocalLLaMA/typed-decisions`, `all/test` | 400 cases, 2,000 binary/choice/ordinal decisions |
| `probes` | Luni의 리터럴 프로브 정의 | 9가지 요청, 14 고유 질문: 접지, 모순 및 라우팅 변형 |

데이터 세트 및 벤치마크 개정은 어댑터에 고정되어 있습니다. 레코드 소스 URL과 파일 SHA256 값을 가져옵니다. 프로브 케이스는 고정된 `probe.py`의 고정 렌더링입니다. 준비는 SHA256을 확인하고 다운로드한 코드를 실행하지 않습니다. 데이터 세트 및 생성된 증거는 소스 패키지가 아닌 무시된 `results/`에 속합니다. 소스 데이터 세트는 자체 용어를 유지합니다. 모델 체크포인트가 다운로드되지 않았습니다.

<a id="prepare-and-run"></a>
## 준비하고 실행하세요

어댑터와 스코어링은 Rust입니다. Typed Parquet 준비에서는 Rust Apache Parquet 리더를 사용합니다. 이러한 명령에는 Python 및 `pyarrow`가 필요하지 않습니다.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools laya-benchmark fetch --suite typed --output results/laya/typed-source
target/release/l2s1-tools laya-benchmark prepare --suite typed \
  --source results/laya/typed-source --output results/laya/typed

cargo build --release --locked --features llama-cuda --example evaluate_jsonl
target/release/l2s1-tools laya-benchmark run \
  --prepared results/laya/typed --output results/laya/fresh \
  --evaluator target/release/examples/evaluate_jsonl --model /path/to/model.gguf
```

동일한 가져오기/준비 흐름에서 `phish` 또는 `probes`를 사용합니다. CUDA의 경우 `--features llama-cuda`를 사용하여 평가기를 빌드한 다음 `--cuda` 및 체크포인트에 필요한 배치(예: `--gpu-layers 24 --model-load-mode read`)를 제공합니다. `llama` 기능은 CPU 전용 런타임을 빌드합니다. [빌드 가이드](GUIDE.md#build)를 참조하세요. 소규모 통합 확인에는 `prepare --limit 4`를 사용할 수 있습니다. 매니페스트와 보고서는 결과를 하위 집합으로 명시적으로 표시합니다.

각 JSONL 케이스에는 하나의 상태에 대한 모든 판단이 포함됩니다. 평가자는 더 이상 각 질문을 별도의 요청으로 강제하지 않습니다. 사례 ID, 상태, 질문 지침 및 판단 기준만 추론을 입력합니다. 골드 라벨, 배포판, 작업 흐름 태그 및 근거는 별도의 파일에 유지됩니다. 옵션 순서, 바이너리 극성 및 서열 레벨은 유지됩니다. 엔진의 기본값은 레거시 프롬프트, 새로 실행 및 비활성화된 준비 캐싱으로 유지됩니다.

모든 실행은 평가자/모델/입력 해시, 명령, 백엔드 설정, 모든 예측, 오류, 타이밍 및 준비 캐시 카운터 스냅샷을 기록합니다. 누락, 중복, 예상치 못한 출력, 잘린 출력 또는 잘못된 형식의 출력은 정답률을 자동으로 개선할 수 없습니다. 전체 정답률은 계획된 분모에 대해 누락/실패한 레이블 판단을 계산합니다. 유효한 전용 정답률은 별도로 이름이 지정됩니다. 실패한 실행은 증거를 유지하고 성공적으로 종료됩니다.

<a id="scoring-boundaries"></a>
## 점수 경계

- `accuracy` / `raw_top1_accuracy`는 판단 보류 이전에 계획된 모든 레이블 판단으로 나눈 원시 올바른 값입니다. `valid_accuracy`는 실패한 판단을 제외하며 해당 분모를 대체해서는 안 됩니다. `accepted_accuracy`는 업스트림 원시 argmax 타이 규칙이 아닌 실제 네이티브가 선택한 답변의 점수를 매깁니다. `correct_accepted`, `wrong_accepted`, `coverage`(승인/모두) 및 `accepted_correct_all`(정답/전체 승인)는 별개입니다. 결과에는 계획된 분모가 있는 `by_type`(`binary`/`choice`/`ordinal`) 및 `by_workflow`가 모두 포함됩니다.
- 바이너리 연결은 업스트림 `p_true >= 0.5` 규칙을 사용합니다. Choice/서열 타이는 원래 옵션 순서를 사용합니다. 네이티브 승인은 여전히 ​​엔진 정책을 따릅니다.
- 피싱은 분류를 인식하는 AUROC, 재현율, 정밀도 및 하드 라벨 Brier를 보고합니다. 업스트림 손으로 쓴 AUROC는 임의로 동점 순위를 매깁니다. 대신 이 어댑터는 동점 절반의 크레딧을 제공합니다.
- 입력된 작업은 바이너리 및 선택 질문에 대해 하드 정답률, 소프트 타겟 Brier/TVD/KL/soft 정답률, 서열 질문에 대해 예상 수준 MAE/within-one을 보고합니다. 둥근 소프트 타겟은 정규화됩니다. 하드 라벨 Brier는 별도이며 클래스별 평균이 아닌 라벨에 대한 합계를 사용합니다. `nll_hard`는 하드 골드 라벨에 대한 자연 로그 손실로, 유효한 라벨이 붙은 판단에 대해 평균을 낸 것이며, 확률은 `1e-12`(최대 27.631 nats)에 있습니다. 이는 서열 라벨을 포함하며 소프트 골드 KL 또는 소프트 타겟 교차 엔트로피가 아닙니다.
- `ece_hard`는 argmax 정확성에 대해 10개의 동일한 너비 bin을 사용합니다. 이는 소프트 교사 분포와의 일치를 측정하는 것이 아니며 지정되지 않은 업스트림 ECE 구현과 상호 교환 가능한 것으로 가정되지 않습니다. `ece_hard_15`는 15개의 빈으로 동일한 정의를 추가로 보고합니다. 신뢰도는 엔트로피 신뢰도가 아니라 가장 큰 후보 확률입니다. 둘 다 유효한 레이블이 지정된 판단만 사용하고 정확한 빈 가장자리를 상위 빈에 배치하며 최종 빈에는 1 확률이 있습니다.
- 프로브 실패는 검사별로 보고됩니다. 보완 및 안정성 테스트는 경험적 방법입니다. 일부 쌍을 이루는 질문은 엄격한 논리적 보완이 아니며 라우팅 변형도 기준표 문구를 변경합니다. 접지 출력은 L2S1 엔트로피 신뢰도를 명시적으로 사용하는 과신도 검사에 재사용됩니다. 보편적인 "11-test 정답률" 또는 교사 합의 상한선은 추론되지 않습니다.
- 소스에서 게시한 GPU 5090 타이밍은 로컬 CPU/3060 실행과 직접 비교할 수 없습니다. 사례별 대기 시간에는 모든 질문과 전체 배치 완료가 포함됩니다. 질문이나 배치 크기로 나누는 것은 대기 시간이 아닌 분할 평균 작업입니다.

<a id="compare-caching-without-changing-the-task"></a>
## 작업을 변경하지 않고 캐싱 비교

캐시 진단에만 즉시 반복을 사용하십시오. 0을 반복하면 품질 결과가 됩니다. 이후의 반복은 추가적인 독립 사례로 간주되지 않습니다.

```sh
target/release/l2s1-tools laya-benchmark prepare --suite typed \
  --source results/laya/typed-source --output results/laya/cache-input \
  --limit 4 --repeats 2

target/release/l2s1-tools laya-benchmark run \
  --prepared results/laya/cache-input --output results/laya/state-fresh \
  --evaluator target/release/examples/evaluate_jsonl --model /path/to/model.gguf \
  --prompt-layout state-first --execution-mode fresh --batch 64 \
  --cache-bytes 67108864

target/release/l2s1-tools laya-benchmark run \
  --prepared results/laya/cache-input --output results/laya/state-reuse \
  --evaluator target/release/examples/evaluate_jsonl --model /path/to/model.gguf \
  --prompt-layout state-first --execution-mode prefix-reuse --batch 64 \
  --cache-bytes 67108864

target/release/l2s1-tools laya-benchmark compare --prepared results/laya/cache-input \
  --baseline results/laya/state-fresh --candidate results/laya/state-reuse \
  --output results/laya/cache-comparison.json
```

비교에는 동일한 체크포인트, 평가자, 입력, 프롬프트 ID, 컴퓨팅 설정 및 정책이 필요합니다. 모든 확률, 후보 확률 질량, 원시 최고 선택 및 허용된 선택을 확인합니다. 레이아웃이나 배치 크기를 변경하려면 새로운 기준이 필요합니다. 이는 캐시 전용 비교가 아닙니다. 0.02 확률/대량 회귀 허용오차는 변경된 상위 선택 또는 허용된 선택을 결코 변명하지 않습니다. 정확한 평등도 보고됩니다. 워밍업은 시간 제한이 없으며 해당 준비 항목은 측정 전에 지워집니다. KV는 여전히 일반 요청 경계에서 지워집니다.

<a id="why-a-cpu-workload-can-show-little-cache-benefit"></a>
## CPU 워크로드가 캐시 이점을 거의 표시할 수 없는 이유

CPU 배치는 KV 재사용을 비활성화하지 않습니다. 네이티브 구현은 작업을 CPU/GPU 레이어에 전달하기 전에 정확한 토큰 접두사를 비교합니다. 구별해야 할 작업에는 세 가지 종류가 있습니다.

| 방식 | 줄인 작업 | 현재 지원 범위 |
| --- | --- | --- |
| 준비 캐시 | 신속한 컴파일/토큰화 및 답변 경계 매핑 | 프롬프트 항목에 대한 정확한 상태와 판단; 모델/구성 범위 및 바이트 제한 |
| 접두사 KV 재사용 | CPU 레이어를 포함한 공유 입력 토큰에 대한 변환기 상태 재계산 | 요청 내의 질문 또는 명시적인 불변 상태 세션 |
| 결과 메모이제이션 | 전체 추론 | 이 어댑터에서는 활성화되지 않습니다. 다른 작업 부하를 측정합니다. |

256의 기본 사전 채우기 배치를 사용하면 255 토큰의 공통 접두사가 **zero** 토큰을 저장합니다. 재사용은 수치적 동작을 보존하기 위해 완전한 원래 배치로 반올림됩니다. 최종 토큰은 항상 평가됩니다. 따라서 요청당 단일 질문, 일반적인 개별 호출, 레거시 명령 우선 레이아웃, 변경된 초기 토큰 또는 지원되지 않는 순환/하이브리드 메모리로 인해 재사용이 0이 될 수 있습니다. 준비 히트만으로는 변압기 작업을 피했다는 것을 알 수 없습니다. 캐시 적중 횟수뿐만 아니라 `reused_prefix_tokens` 및 `native_ms`도 검사하세요.

9월 24, 2026의 CPU 통합 검사에서는 i5-12600KF에서 SmolLM2 135M Q8_0을 사용했으며, 4개의 스레드, 상태 우선 프롬프트, 처음 2개의 유형화된 테스트 사례(각각 5개의 판단), 그리고 한 번의 즉시 반복. 배치 256에서 준비 캐싱은 준비를 16.45에서 7.93 ms로 줄였지만 전체 추론은 3.4초 주변에 머물렀고 KV 재사용은 0이었습니다. 배치 64에서 신규 대 접두사 재사용에는 3.899 대 2.389초(1.63x)가 걸렸으며 2,048는 재사용된 토큰을 사용했습니다. 모든 20 확률 벡터, 후보 질량, 상위 선택 및 허용된 선택은 동일한 배치 비교 내에서 정확히 동일했습니다. 이 작은 직렬 검사는 31B 또는 운영 환경 속도 보장이 아닌 CPU 재사용을 보여줍니다. 배치 64는 또한 배치 256보다 새로운 경로를 느리게 만들었습니다.

동일한 서버는 또한 CPU/GPU 분할(24 오프로드 레이어, 8개 스레드, 읽기 로딩)을 사용하여 Gemma 4 31B Q4_K_M을 확인했습니다. 동일한 두 가지 유형의 사례와 해당 반복에서 배치 64 신규/재사용에는 108.552/69.904초(1.55x)가 걸렸습니다. 배치 128는 62.438/42.770초(1.46x)를 사용했습니다. 두 동일한 배치 캐시 비교 모두 모든 20 확률 벡터, 후보 질량 및 선택을 정확하게 보존했습니다. 그러나 일치하는 배치-256 새 케이스에는 41.677초가 소요되었습니다. 더 작은 배치 캐시 구성도 해당 기준을 능가하지 않습니다. 배치 크기 전반에 걸쳐 최대 확률 변화는 0.00165/0.00504였으며 이 작은 샘플에서는 선택 사항이 변경되지 않았습니다. 일괄 256를 기본값으로 유지합니다. CPU 캐시 효율성은 사전 채우기 형상을 변경한 후 순 이득을 의미하지 않습니다. 모든 15 통합 구성(350 판단 레코드(반복 포함))이 오류나 잘림 없이 완료되었습니다. 여기에는 어댑터가 준비한 4,000 라벨 판단에 대한 전체 추론이 아닌 소규모 유형/피싱 샘플과 전체 행동 프로브 세트가 포함됩니다.

<a id="speed-work-in-priority-order"></a>
## 우선순위에 따라 속도 작업

1. **하나의 모델을 상주하고 케이스 수준 그룹화를 유지합니다.** 이제 어댑터가 이 작업을 수행합니다. API 호출 전반에 걸쳐 불변 상태에 대한 질문을 변경하려면 `SharedStateSession`를 사용하세요. 일반 요청은 격리된 상태로 유지됩니다.
2. **동일한 레이아웃의 새로운 기준선을 사용하여 상태 우선을 측정합니다.** 질문별 접미사 앞에 공유 증거를 노출합니다. 레이아웃을 변경하면 모델 예측이 변경될 수 있으므로 선택 상태로 유지됩니다.
3. **짧은 CPU 상태에서 더 작은 사전 채우기 배치를 테스트합니다.** 배치 64/128는 짧은 접두사를 재사용 가능하게 만들 수 있지만 행렬 곱셈 효율성도 감소시킵니다. 각 크기별로 최신 상태와 캐시된 상태를 비교하세요. 배치 정렬 가드를 제거하지 마십시오.
4. **반복 입력 및 고정 스키마에 대해 제한된 준비 캐싱을 사용합니다.** 후보 매핑은 변경된 상태에서 적중할 수 있지만 정확한 프롬프트 항목은 누락됩니다. 네이티브 추론이 지배적인 경우 토큰 준비 비용 절감만으로는 총 대기 시간을 실질적으로 변경할 수 없습니다. 또한 `boundary_and_other_ms`를 검사합니다. 여기에는 유효성 검사, 요청 경계 지우기 및 단계 타이머 외부의 기타 작업이 포함됩니다. 네이티브 `sd_clear` 0 KV 버퍼(호스트 버퍼 포함) 및 일반 배치/요청 경계는 이를 반복적으로 호출할 수 있습니다. 중복된 지우기를 제거하거나 물리적 제로화에서 논리적 재설정을 분리하는 것은 격리/오류 복구 테스트가 필요한 후속 최적화입니다. 이 어댑터는 이러한 의미를 변경하지 않습니다.
5. **실제 메모리 예산 하에서 병렬 폭을 평가합니다.** 처리량을 향상할 수 있지만 KV 메모리가 늘어나고 수치 결과가 변경될 수 있습니다. 이미 12 GiB VRAM 근처에 있는 31B 모델에서는 더 많은 오프로드된 레이어 또는 시퀀스가 ​​실패할 수 있습니다. `--request-batch-size`만으로는 신규 모드에서 네이티브 병렬성을 생성하지 않습니다.
6. GPU 오프로드를 늘리기 전에 측정된 입력 길이에 **Match 컨텍스트 할당.** 짧은 분류 입력에는 8,192 토큰 KV 버퍼가 필요하지 않을 수 있습니다. 더 작은 명시적 컨텍스트는 더 많은 레이어를 위해 GPU 메모리를 확보하여 CPU 계산을 줄일 수 있습니다. 모든 입력 및 응답 코드 접두어를 확인하십시오. 절대 조용히 자르지 마세요. 새로운 컨텍스트/배치 조합으로 품질과 할당을 다시 확인하세요.
7. **나중 기능으로 명시적 고정 스키마 접두사 세션을 고려합니다.** 일반 교차 요청 KV 재사용은 여기서 구현되지 않습니다. 이러한 API에는 정확한 토큰 접두사 확인, 모델/템플릿/어댑터 ID, 제한된 메모리, 명시적 소유권 및 오류 무효화가 필요합니다. 의미상 유사한 입력에 대해 답변 캐시와 혼동하거나 캐시된 답변을 재사용하지 마십시오.

변환기 추론의 양이 아닌 로딩 변경 할당 및 시작 RSS를 읽으십시오. 양자화, CPU 스레드 수와 레이어 배치는 별도의 튜닝 축입니다. 각각은 독립적으로 확인된 품질 기준을 유지해야 합니다. 31B 대체 기능을 갖춘 더 작은 1단계 모델도 추론 정책을 변경하고 자체 정답률/수락률 평가가 필요합니다.
