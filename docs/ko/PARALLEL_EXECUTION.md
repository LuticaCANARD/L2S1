<a id="parallel-question-execution"></a>
# 병렬 질문 실행

[English](../en/PARALLEL_EXECUTION.md) · [한국어](PARALLEL_EXECUTION.md) · [日本語](../ja/PARALLEL_EXECUTION.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

지원되는 `parallel` 모드는 별도의 llama.cpp 시퀀스 ID를 사용하여 독립적인 질문을 평가합니다. 로드된 모델을 공유하고, 웨이브당 한 번씩 정확한 공통 프롬프트 접두사를 계산하고, 여러 질문의 접미사 토큰을 각 디코드 배치에 넣습니다. 하나의 라마 컨텍스트를 동시에 변경하는 여러 스레드를 시작하지 않습니다.

```sh
cargo run --release --locked --features llama-cuda -- \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --input examples/warehouse.json --device cuda \
  --prompt-layout state-first --execution-mode parallel --parallel-width 4
```

기본값은 `legacy` 프롬프트와 `fresh` 실행으로 유지됩니다. 프롬프트 레이아웃과 실행 모드는 별도의 선택입니다. 상태를 이동하면 프롬프트가 변경됩니다. 병렬성은 실행 일정을 변경합니다. 동일한 레이아웃에서 모든 실행 모드를 비교합니다.

<a id="implementation"></a>
## 구현

- Rust는 각 질문의 토큰과 A–Z 단일 토큰 연속을 준비하고 검증합니다. 사용자 상태 및 질문 데이터는 특수 토큰 구문 분석 보호를 유지합니다.
- 최대 `parallel_width` 질문은 각 웨이브에 입력됩니다(기본값 4, 1–32 허용). 마지막 파도는 더 작을 수 있습니다. 프롬프트가 다른 디코드 배치로 완료되는 경우를 포함하여 결과에는 요청 순서와 ID가 유지됩니다.
- 질문당 하나 이상의 접미사 토큰을 유지하면서 전체 웨이브에서 정확한 토큰 ID를 비교하여 공통 접두사를 찾습니다. 이는 완전한 원래의 사전 충전 배치로 반올림됩니다. 첫 번째 시퀀스에서는 이를 평가합니다. `llama_memory_seq_cp`는 KV 항목을 다른 시퀀스와 공유합니다.
- 접미사 토큰에는 독립적인 시퀀스 ID와 원래 절대 위치가 있습니다. 그들의 관심은 다른 질문의 접미사가 아니라 자신의 순서와 공유 접두사를 봅니다. 최종 전체 어휘 logits는 관련 디코드 배치에서 각 질문의 최종 토큰 인덱스를 사용하여 복사됩니다.
- 웨이브, API 호출 사이 및 실패 후에 메모리가 지워집니다. 지속적인 교차 요청 캐시가 없습니다. 순환/하이브리드 모델은 이 모드에서 명시적으로 지원되지 않습니다.
- 모델은 한 번 로드된 상태로 유지됩니다. 기본적으로 시퀀스 용량이 변경되면 명목 토큰 용량 `context * width`를 사용하여 컨텍스트가 느리게 재생성됩니다. `context`는 질문당 입력 제한을 유지합니다. 너비가 높을수록 KV/attention 메모리가 증가하고 할당이 실패할 수 있습니다. 자동 CPU 대체 또는 입력 잘림이 없습니다.
- `backend.parallel_width`는 구성된 웨이브 제한을 기록합니다(직렬 모드는 1를 보고합니다). `reused_prefix_tokens`는 공유 접두사에 대한 비용을 지불한 웨이브의 첫 번째 질문과 팔로어에 대한 공유 토큰 수에 대해 0입니다. 실제 제출된 토큰에 대한 `input_tokens - reused_prefix_tokens` 계정을 합산했습니다.

이는 고정된 llama.cpp [batch, 시퀀스, 메모리 복사본 및 토큰별 logits API](https://github.com/ggml-org/llama.cpp/blob/3d82ef62d47fd74e18f36c5eccbdcf965b617b17/include/llama.h)를 사용합니다. 이는 대체 모델 아키텍처나 보정된 판단 교육이 아닌 직렬 웨이브를 사용한 독립적인 시퀀스 일괄 처리입니다.

<a id="dynamic-parallel-kv-context"></a>
### 동적 병렬 KV 컨텍스트

`--parallel-context-dynamic`는 `parallel` 모드에 대한 옵트인 메모리 설정입니다. 각 웨이브를 토큰화한 후 최대 `sum(input_tokens) + batch` KV 슬롯을 예약하며 레거시 `context * width` 예약으로 제한됩니다. llama.cpp는 요청된 크기를 채울 수 있습니다. 합계는 공유 접두사를 두 번 이상 계산하므로 이는 보수적입니다. `--context`는 여전히 각 개별 질문의 유효성을 검사합니다. 입력이 잘리지 않습니다.

이후 웨이브에 더 많은 슬롯이 필요할 때 네이티브 컨텍스트가 커집니다. 반복적인 컨텍스트 재구성을 피하기 위해 동일한 유효 질문 수로 이후 웨이브에 대한 높은 수용력을 유지합니다. 기본 모드와 동적 모드 사이를 전환하면 컨텍스트가 다시 생성되고 두 모드 모두 추론 후 request-local KV를 지웁니다. `backend.parallel_context_dynamic`는 옵트인 결과를 식별합니다. `backend.parallel_context_tokens`는 동적 모드에서 현재 할당된 패딩된 컨텍스트를 보고합니다. 모델 가중치는 로드된 상태로 유지됩니다.

이는 llama.cpp 컨텍스트 크기를 변경하고 logits 또는 판단을 변경할 수 있습니다. 판단을 위해 메모리 설정을 사용하기 전에 기본 병렬 모드와 정확한 모델, 프롬프트, 정책 및 입력 세트를 비교하십시오. 이는 메모리 최적화입니다. 처리량 향상을 약속하지 않습니다.

<a id="independent-request-batches"></a>
### 독립적인 요청 배치

`backend.decide_batch(&requests)`는 상태를 병합하거나 질문 프롬프트를 변경하지 않고 여러 요청을 평가합니다. 병렬 모드는 질문을 격리된 시퀀스로 평면화하고 경계파를 처리하며 원래 요청/판단 그룹화를 복원합니다. 다른 요청에서 반복되는 판단 ID가 허용됩니다. 이 명시적 호출에서는 정확한 토큰 접두사만 공유할 수 있습니다. KV는 통화에서 살아남지 않습니다. 실패하면 부분 응답 결과 없이 전체 배치에 대한 오류가 반환됩니다. 빈 배치는 빈 결과 목록을 반환합니다.

JSONL 평가자는 `--execution-mode parallel --parallel-width 16 --request-batch-size 16`와 선택적 `--warmup` 배치를 지원합니다. 이를 통해 동일한 독립 기사 요청에 대한 공정한 비교가 가능해집니다. 각 기사의 전체 배치 완료 대기 시간, 배치 ID 및 분할 평균 컴퓨팅 시간을 별도로 기록합니다. 배치의 경과 시간을 크기로 나누면 개별 응답 대기 시간이 아닌 분할 평균 비용을 측정합니다.

<a id="independent-image-batches"></a>
### 독립적인 이미지 배치

일치하는 비전 프로젝터가 로드되면 `backend.decide_vision_batch(&requests, &images)`는 자체 상태와 하나의 원본 이미지에서 각 요청의 점수를 매깁니다. `parallel` 모드에서 판단은 제한된 네이티브 파동을 입력합니다. `fresh` 모드는 이를 순차적으로 평가합니다. HTTP는 각 판단이 `media_ids`를 통해 하나의 이미지를 참조하고 리스너가 `--execution-mode parallel --parallel-width 4`를 사용할 때 동일한 경로를 노출합니다. 4개의 독립적인 판단이 있는 4개의 요청 미디어 항목은 단순히 HTTP 응답을 공유하는 대신 네이티브 실행을 공유합니다.

이미지 디코딩 및 다중 모드 토큰화는 신뢰할 수 있는 프롬프트 제어 토큰 경계를 유지합니다. 호환되는 프로젝터 청크는 llama.cpp mtmd의 배치 인코더에 입력됩니다. 프로젝터 모양 및 토큰 제한으로 인해 이를 더 작은 인코더 배치로 나눌 수 있습니다. 모든 웨이브 질문은 자체 전체 어휘 최종 logits 또는 컴팩트 후보 점수 및 전체 어휘 노멀라이저와 함께 독립적인 디코더 시퀀스 ID 및 시퀀스별 이미지 위치를 사용합니다. Vision은 현재 프롬프트 접두사 KV를 공유하지 않으므로 `reused_prefix_tokens`는 0으로 유지됩니다. 비인과적 이미지 청크는 구성된 토큰 배치 및 마이크로배치에 전체적으로 맞아야 합니다. 지원되지 않는 레이아웃이나 반복 모델은 직렬 실행으로 전환하는 대신 오류를 반환합니다. 26 이상의 옵션에는 여전히 `fresh` 비전 실행이 필요합니다.

이미지 웨이브는 별도의 KV 스트림을 사용합니다. 동적 컨텍스트 크기 조정은 질문당 제한에 따라 가장 긴 다중 모드 입력과 각 스트림의 헤드룸 토큰 배치 1개를 예약합니다. 토큰 수와 위치 범위 모두 해당 제한과 비교하여 확인됩니다. 각 질문은 `--context`로 제한됩니다. 메모리 및 이미지 임베딩은 일괄 처리 또는 오류 후에 지워집니다. 최종 부분파 보존 요청 및 판단 주문. 일괄 실패는 부분 응답을 반환하지 않습니다.

`backend.vision_batch_metrics()`는 최신 네이티브 wave의 카운터를 제공합니다. 병렬 이미지 HTTP 응답도 `backend.details.vision_batch` 아래에 `scope: "last_native_wave"`로 표시해 다음을 포함합니다. `projector_encode_calls`, `projector_batch_max`, `decoder_calls`, `decoder_batch_max_sequences`, `projector_reused_chunks`, `kv_clear_calls`, `kv_clear_skipped`. 두 clear 카운터는 `kv_clear_scope: "since_last_native_vision_start"`를 사용합니다. 다음 네이티브 비전 호출이 초기화하기 전에 후속 정리·설정 호출이 이 값을 늘릴 수 있습니다. 함께 있는 준비 캐시 카운터는 설정 이후 백엔드 수명을 대상으로 하는 별도 `preparation_cache_scope`를 가집니다. 이 카운터가 실제 인코더·디코더 배치 형태를 입증하며, HTTP 요청의 이미지 수만으로는 입증할 수 없습니다. 전체 배치 완료 지연 시간과 이미지당 분할 평균 밀리초를 따로 측정하고 같은 이미지의 `fresh`와 점수·최상위 선택·판단 보류·작업 정답률을 비교하세요.

[120-image TrashNet 측정](benchmarks/trashnet-vision-20260925/REPORT.md#native-four-image-batching-2026-09-26)는 RTX 3080에서 4개의 네이티브 디코더 시퀀스를 확인했으며, 분할 평균 처리 시간은 더 낮았지만 세 가지 모두 모두 CUDA 체크포인트가 기존 수치 동등성 판단 기준에 실패했습니다. 선택한 값 또는 원시 순위가 변경되었습니다. 이미지 일괄 처리는 해당 모델 및 레이아웃 제한 내에서 지원됩니다. 네이티브 배치 카운터 및 격리 테스트는 점수 동일성 또는 작업 정답률을 설정하지 않습니다.

<a id="optimization-components"></a>
### 최적화 구성요소

최적화된 4개의 이미지 프로필에는 정확한 프롬프트 준비 캐싱, 압축 증거 전송 및 더티 전용 물리적 KV 삭제가 포함됩니다. 이러한 구성 요소는 변경되지 않은 컴퓨팅 구성에서 기존 setter를 사용하여 별도로 테스트됩니다. 점수 평등은 새로운 대 병렬 패리티를 설정하지 않습니다. 직렬 캐시/컴팩트 비교에서는 속도가 향상되지 않았습니다. 이러한 구성 요소는 `--vision-optimized`에 포함되어 있습니다.

새로운 이미지 HTTP 응답은 백엔드 수명 캐시 스냅샷을 사용하여 `backend.details.vision_preparation`를 노출하고 마지막 네이티브 호출에 대해 `vision_kv_clear`를 노출합니다. 모든 직렬 미디어 그룹은 동일한 최종 스냅샷을 수신하므로 공통 응답 메타데이터는 일관성을 유지합니다. 이러한 필드는 이미지 또는 KV 재사용을 주장하지 않습니다.

`--vision-projector-reuse`는 병렬 모드에서 명시적인 프로젝터 재사용을 위해 지원되는 설정입니다. 각 고유 청크를 독립적으로 인코딩하고 하나의 웨이브 내에서 동일한 청크를 재사용합니다. 직렬 디코더의 수치 실행은 보존되지 않습니다.

<a id="combined-vision-optimizations"></a>
### 결합된 비전 최적화

`--vision-optimized`에는 일치하는 프로젝터와 명시적으로 선택된 CUDA 또는 Metal 장치가 필요합니다. 병렬 너비 4, 동적 스트림별 컨텍스트, 토큰 배치/마이크로배치 1024, Flash Attention 켜기, 압축 증거, 8 MiB 준비 캐시 및 동일 이미지 프로젝터 재사용을 선택합니다. 질문당 컨텍스트의 기본값은 4096입니다. Rust 호출자는 `ComputeOptions::vision_optimized()`를 사용하여 백엔드를 구성하고 프로젝터를 로드한 다음 `enable_vision_optimizations()`를 호출합니다.

네이티브 메모리는 부분 쓰기 후 실패한 작업을 포함하여 모든 디코드 또는 스냅샷 복원을 시작하기 전에 쓰기를 추적합니다. `sd_clear`는 항상 논리적으로 캐시된 토큰을 무효화하지만 더러워지면 물리적 KV만 0으로 만듭니다. 성공적인 비전 호출은 네이티브 정리를 소유합니다. Rust 가드는 검증/준비/점수 오류 및 해제를 지웁니다. 요청 및 세션 격리를 유지하면서 중복 지우기를 건너뜁니다.

준비 캐시는 정확한 렌더링 상태/판단 프롬프트 및 답변 경계 매핑을 유지합니다. 이미지 바이트 또는 KV를 저장하지 않습니다. 동일한 바이트와 일치하는 순서의 청크/위치 메타데이터가 있는 이미지는 하나의 웨이브 내에서 불변의 프로젝터 임베딩을 공유할 수 있습니다. 이 재사용 모드에서는 각 고유 청크가 단일 청크 인코더 배치를 사용하므로 다른 이미지를 변경해도 인코더 배치 모양이나 슬롯이 변경되지 않습니다. 이는 프로젝터 일괄 처리에 대한 대안을 선택합니다. 디코더 일괄 처리가 유지됩니다. 디코더 KV 스트림과 이미지 위치는 독립적으로 유지됩니다. `backend.vision_projector_reuse`는 활성화된 설정을 보고하고 `projector_reused_chunks`는 실제 재사용된 청크를 보고합니다. 이는 전역 이미지와 타일을 생성하는 프로젝터의 이미지 수를 초과할 수 있습니다.

컴팩트 전송은 각 판단의 후보자와 전체 어휘 정규화 프로그램만 Rust에 복사합니다. llama.cpp는 여전히 전체 어휘를 ​​계산하고 그 출력을 호스트로 전송합니다. 최적화된 프로필은 텍스트 전용 KV 접두사 세션 또는 스냅샷 복원을 독립적인 비전 스트림과 결합하지 않습니다. 이는 대체 실행 모드입니다. 하이브리드/반복 모델 및 광범위한 응답 코드는 이 프로필에서 명시적으로 거부됩니다.

병렬 attention과 프로젝터 배치 형태는 모두 수치 결과를 바꿀 수 있습니다. fresh를 기본으로 유지하며 원래 입력에서 통합 프로파일과 fresh를 비교하세요. 같은 계산 설정의 전체·축약 동등성은 fresh·최적화 동등성과 따로 검증합니다. Metal 실행·성능은 실제 하드웨어 테스트가 여전히 필요합니다.

<a id="validation-and-measurement"></a>
## 검증 및 측정

시력 격리와 수치적 동등성은 별도의 테스트입니다. `SKID_VISION_MODEL`, `SKID_VISION_MMPROJ` 및 선택적으로 `SKID_CUDA=1`를 설정한 후 다음을 실행합니다.

```sh
cargo test --release --locked --features llama-cuda --test vision \
  real_vision_batch_preserves_image_state_order_and_recovers_after_errors \
  -- --ignored --test-threads=1
cargo test --release --locked --features llama-cuda --test vision \
  real_vision_batch_equivalence_on_color_fixture -- --ignored --test-threads=1
python3 benchmarks/trashnet-vision-20260925/evaluate_batch.py --help
```

격리 테스트를 통과하더라도 CUDA에서 동등성 테스트가 실패할 수 있습니다. 평가자는 허용오차를 늘리거나 배치 결과를 직렬 추론으로 대체하는 대신 비교에서 이러한 실패를 유지합니다.

```sh
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  cargo test --release --locked --offline --features llama-cuda \
  --test parallel real_model_parallel_contract -- --ignored --nocapture
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  SKID_PARALLEL_WIDTH=4 SKID_PARALLEL_OUTPUT=/tmp/parallel.json \
  cargo test --release --locked --offline --features llama-cuda \
  --test parallel real_model_parallel_measurement -- --ignored --nocapture
```

계약 테스트는 혼합 유형 질문, 출력 ID/순서, 동일하지 않은 프롬프트 길이, 부분 웨이브, 공유 접두사 재사용, A-Z 후보 매핑, 신뢰할 수 없는 토큰형 텍스트, 교차 질문 격리, 변경된 상태 요청 격리, 이전 웨이브 완료 후 오류 복구, 유효하지 않은 너비 및 새로운 실행과 너비 1 동등성을 다룹니다.

측정에서는 단기 및 장기 창고 상태에 대한 1, 4, 16 및 32 질문을 사용합니다. 질문은 확장을 측정하기 위해 세 개의 창고 판단 기준을 반복합니다. 이는 32 고유의 실측 작업이 아닙니다. 각 모드에는 시간 제한이 없는 워밍업(컨텍스트 할당 포함) 1회와 시간 제한 반복 3회가 있습니다. 모드 순서는 구성 간에 번갈아 나타납니다. 상세한 JSON는 모든 점수와 대기 시간을 유지합니다.  0.02는 기존 확률/질량 비교 임계값으로 유지되며 변경된 상위 1 또는 허용된 선택은 보고된 동등성 판단 기준에 실패합니다.

이러한 측정에서는 정상 상태 대기 시간에서 모델/컨텍스트 시작을 제외하고 원격 API/네트워크를 포함하지 않으며 Jev 성능 패리티 또는 보정된 신뢰도를 설정하지 않습니다.

<a id="numerical-limitations"></a>
## 수치적 제한

병렬 실행은 배치 형태를 변경하고 확률, 최고의 선택 및 허용된 판단을 변경할 수 있습니다. 이전의 픽스처 검사에서는 새로운 판단 보류가 잘못된 답변으로 받아들여지는 것을 관찰했습니다. 더 넓은 픽스처는 기존 0.02 확률/질량 허용 오차에 실패했습니다. 이 모드는 `--execution-mode parallel`를 사용하여 명시적으로 지원되고 선택됩니다. 동일한 정책 및 동등성 임계값을 유지하면서 정확한 체크포인트 및 워크로드에 대한 새로운 실행과 비교해 보세요.
