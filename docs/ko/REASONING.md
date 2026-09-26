<a id="direct-scoring-and-bounded-thinking"></a>
# 직접 점수 계산과 제한된 사고

[English](../en/REASONING.md) · [한국어](REASONING.md) · [日本語](../ja/REASONING.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

Direct 모드는 기본값이며 기존 프롬프트 바이트를 유지합니다. 닫힌 빈 사고 블록 뒤에서 후보 코드의 점수를 계산하고 추론 시퀀스를 생성하지 않습니다. 명시적인 `thinking` 모드는 토큰을 greedy 방식으로 생성하고 모델이 고유한 `</think>` 토큰을 출력할 때까지 기다린 다음 최종 구분자를 평가하고, 전체 어휘 logits로 같은 타입의 후보 집합을 평가합니다. 최종 답변 코드는 생성하지 않고 점수만 계산합니다.

초기 thinking 지원은 채팅 템플릿이 `enable_thinking`을 지원하는 **dense Qwen3 텍스트 GGUF**로 제한되며, 지원 모델에서 자동 선택되는 `qwen3` 프롬프트 프로파일을 사용합니다. 이미지 thinking, 다른 아키텍처, 26개를 넘는 후보, 프리픽스 재사용, 병렬 실행, 상태 복원, 준비 캐시, 축약 근거, LoRA, 출력 헤드, 학습된 보정, 숨은 특징 내보내기는 거부합니다. 이미지 판단은 호환되는 비전 모델과 프로젝터로 direct 모드를 계속 사용합니다.

<a id="cli"></a>
## CLI

```sh
cargo build --release --locked --features llama --bin l2s1
target/release/l2s1 --model models/Qwen3-0.6B-Q8_0.gguf \
  --input request.json --reasoning-mode thinking --max-reasoning-tokens 128
```

기존 실행 경로는 `--reasoning-mode direct`로 선택합니다. 사고 예산의 기본값은 128이며 1–1024를 허용합니다. 입력, 전체 토큰 예산, 최종 구분자가 설정된 컨텍스트에 들어가야 합니다. 입력 잘라내기는 비활성화되어 있습니다. 예산 안에서 사고를 닫지 못하면 생성 토큰 수와 예산을 담은 `reasoning_limit`으로 실패합니다. 닫히기 전에 생성 종료 토큰이 나오면 `reasoning_incomplete`로 실패합니다. 어느 경로도 합성 종료 토큰을 삽입하거나 미완료 시퀀스를 답으로 바꾸지 않습니다.

<a id="library-and-http"></a>
## 라이브러리와 HTTP

```rust,ignore
backend.set_reasoning(l2s1::ReasoningOptions {
    mode: l2s1::ReasoningMode::Thinking,
    max_tokens: 128,
})?;
let response = backend.decide(&request)?;
backend.set_reasoning(l2s1::ReasoningOptions::default())?;
```

JSON HTTP 요청은 `"reasoning":{"mode":"thinking","max_tokens":128}`을 받습니다. 어댑터는 실패를 포함해 각 요청 뒤에 이전 추론 모드를 복원합니다. 지원 모델 정보는 capability 엔드포인트에서 확인할 수 있습니다. 비전 요청에 thinking을 선택하면 명시적인 오류가 발생합니다.

성공한 CLI·라이브러리 결과에는 `reasoning`이 포함됩니다. HTTP 결과에서는 `usage.reasoning`에 배치합니다. 예시는 다음과 같습니다.

```json
{"mode":"thinking","generated_tokens":88,"completed":true}
```

개수에는 모델이 생성한 종료 토큰이 포함되지만, 신뢰된 최종 구분자와 생성하지 않은 답변 코드는 제외됩니다. `input_tokens`는 준비된 입력 프롬프트를 계속 셉니다. 사고 내용은 반환하지 않습니다. 프롬프트·아티팩트 식별 정보에는 `thinking-greedy-v1`과 예산이 포함되어 direct 모드 보정이 조용히 재사용되지 않게 합니다.

<a id="verified-boundary"></a>
## 검증 범위

2026년 9월 26일, 실제 Qwen3-0.6B Q8_0 CPU 런타임은 고양이 텍스트의 이진 질문을 생성 토큰 88개로 완료했습니다. 1토큰 예산은 명시적으로 실패했습니다. 해당 실패 뒤와 완료된 사고 뒤에 direct 모드로 돌아오면 같은 백엔드 인스턴스의 모든 후보 확률이 정확히 재현되었습니다. thinking 반복 실행도 토큰 사용량과 확률을 정확히 재현했습니다. 잘못된 캐시·실행 조합은 거부되었고 올바르게 복구되었습니다.

같은 질문은 RTX 3080의 실제 CUDA HTTP 브라우저 데모에서도 생성 토큰 96개로 완료되었습니다. 1토큰 예산은 사용자 지정 `reasoning_limit` 문구와 함께 HTTP 422를 반환했습니다. CPU와 CUDA의 생성 토큰 수가 같다고 가정하지 않습니다. 이는 완료 확인이며, 통제된 지연 시간이나 장치 간 수치 동등성 측정이 아닙니다.

이는 네이티브 생성과 격리 회귀 검증이며 품질 벤치마크 또는 thinking이 정답률을 높인다는 주장이 아닙니다. 공개 typed-decisions 측정은 direct 모드를 사용합니다. 사고 후에도 후보 확률과 `candidate_mass`는 기존의 조건부 점수 의미를 유지합니다.

실제 모델 회귀 검증은 명시적으로 실행하세요.

```sh
L2S1_TEST_REASONING_MODEL=/path/to/Qwen3-0.6B-Q8_0.gguf \
  cargo test --locked --features llama --test reasoning_llama -- --nocapture
```

로컬 direct·thinking·limit 원시 근거는 gitignore된 `results/reasoning-native-20260926/` 디렉토리에 있습니다.
