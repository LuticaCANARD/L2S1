<a id="frozen-deployment-output-heads"></a>
# 모델 본체를 고정하는 배포용 출력 헤드

[English](../en/OUTPUT_HEAD.md) · [한국어](OUTPUT_HEAD.md) · [日本語](../ja/OUTPUT_HEAD.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

선택적으로 사용하는 `--output-head <JSON>`은 명시적으로 지정한 하나의 choice 작업을 작은 학습 분류기로 평가합니다. GGUF 본체는 고정됩니다. 요청하지 않으면 어댑터를 로드하지 않으며 일반 판단은 원래 점수 경로를 유지합니다.

두 종류의 헤드를 지원합니다.

- `logit_affine`: 의미를 나타내는 후보 ID로 대응시킨 기본 후보 logits 위에 행렬과 편향을 학습합니다. 단위 행렬은 temperature만 사용하는 보정을 구현합니다.
- `hidden`: Gemma4의 최종 출력 정규화 뒤 은닉 상태 위에 행렬과 편향을 학습합니다. E2B 특징은 1,536차원이며 3클래스 헤드는 매개변수 4,611개입니다.

둘 다 `softmax((W x + b) / temperature)`를 계산합니다. 아티팩트의 행은 의미를 나타내는 후보 ID를 사용하므로 후보 순서를 바꾸면 코드와 클래스의 대응이 올바르게 바뀝니다. 프롬프트 순서 변경은 기본 모델 특징에도 영향을 줄 수 있어 별도로 측정합니다.

`candidate_mass`는 원래의 전체 어휘 LM 후보 질량을 유지합니다. 별도로 유지하는 호환성 기준이며 **새 분류기가 제공하는 확률이 아닙니다.** Temperature는 후보 분포만 보정합니다. 응답은 `calibration_id`에 헤드를 식별하고 `backend.output_head_path`에 경로를 기록하며 `learned_hidden_softmax_with_base_mass_v1` 또는 `learned_logit_affine_softmax_with_base_mass_v1`을 사용합니다. `scores[].raw_logit`은 temperature 적용 전 헤드 점수이고 `option_probability`는 temperature를 포함합니다.

<a id="runtime-contract"></a>
## 런타임 계약

아티팩트는 정확한 GGUF SHA-256, 장치 설명, 계산 옵션, 프롬프트 버전, 작업 ID, 지시문, 의미를 나타내는 후보 ID·기준에 연결됩니다. 후보 순서 변경은 허용합니다. 다른 작업 ID는 기본 모델을 사용합니다. 학습된 ID에 변경된 지시문이나 후보 의미를 사용하면 명시적으로 실패합니다. 작업 ID는 호출자의 선언이므로, 다른 도메인 텍스트를 그 ID로 전달하는 경우 자동 탐지하지 않습니다.

헤드가 로드되면 fresh 실행만 지원합니다. LoRA와 출력 헤드는 조합할 수 없습니다. 로드 후 프롬프트·실행 설정이 바뀌면 추론 전에 다시 확인합니다. 은닉 특징 내보내기는 현재 Gemma4만 지원합니다. 같은 고정 llama.cpp 리비전을 사용하고 런타임 변경 뒤에는 다시 검증·학습하세요. JSON은 공유 라이브러리나 GPU 드라이버의 지문을 기록하지 않습니다.

네이티브 브리지는 고정된 llama.cpp staging API `llama_set_embeddings_nextn(..., true, false)`와 `llama_get_embeddings_nextn_ith`로 Gemma4의 post-norm 특징을 읽습니다. 마스킹 없는 추출은 기본 그래프 형태를 유지하며 최종 decode 배치의 은닉 행을 전달합니다. Rust는 마지막 행만 받습니다. 일반 embedding 모드는 모든 입력 토큰에 어휘 출력을 강제하므로 의도적으로 사용하지 않습니다. 유지된 질량 기준에는 전체 어휘 투영이 여전히 필요하므로 LM 헤드를 제거하거나 Jev/RLCD 지연 시간을 재현한다고 주장하지 않습니다.

<a id="reproduce-the-pilot"></a>
## 파일럿 재현

포함된 llama.cpp 커밋 `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`과 CUDA feature로 빌드하세요.

```bash
cargo build --release --offline --features llama-cuda --examples --bin l2s1
cargo build --release --offline -p l2s1-tools

target/release/l2s1-tools prepare-output-head \
  --source results/kaggle-airline-20260922 \
  --previous results/finetune-20260923/data \
  --output results/output-head-20260923/data

mkdir -p results/output-head-20260923/features
for split in train dev calibration test probe; do
  target/release/examples/export_decision_features \
    --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
    --input "results/output-head-20260923/data/$split.jsonl" \
    --output "results/output-head-20260923/features/$split.jsonl"
done

# Python environment with PyTorch (CPU training) and the standard library.
python3 scripts/train_output_head.py \
  --data results/output-head-20260923/data \
  --features results/output-head-20260923/features \
  --output results/output-head-20260923/heads

target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda --warmup \
  --output-head results/output-head-20260923/heads/selected.json \
  --input results/output-head-20260923/data/test.jsonl \
  --output results/output-head-20260923/selected-test.jsonl
```

선택한 아티팩트의 실제 주 CLI 호출은 다음과 같습니다.

```bash
target/release/l2s1 --model models/gemma-4-E2B-it-Q8_0.gguf \
  --device cuda --output-head results/output-head-20260923/heads/selected.json \
  --input results/output-head-20260923/example-request.json
```

주 CLI도 `--device cuda --output-head ...`로 같은 아티팩트를 받습니다. 입력은 JSONL 벤치마크 래퍼가 아닌 `DecisionRequest`입니다. `LlamaBackend::extract_features`는 일반 판단 응답과 함께 fresh 기본 요청 하나를 내보내며 `load_output_head`는 명시한 아티팩트를 활성화합니다.

파일럿은 이전 두 항공사 실험에 쓰인 모든 정규화된 완전 일치 텍스트를 제외합니다. 학습 텍스트 900개(세 순환 후보 순서 = 2,700회 호출), 개발 300개, 보정 400개, 테스트 400개를 고정합니다. 테스트 텍스트 60개의 순열 프로브는 180회 호출이지 독립 사례 180개가 아닙니다. 분할은 균형적이며 원래 모집단 분포가 아닙니다. 의미상 근접 중복은 제외하지 않습니다.

특징 표준화는 학습 데이터만 사용합니다. Float64 CPU LBFGS는 교차 엔트로피에 `L2/2 * ||W||²`를 더해 학습합니다. 고정 탐색값은 0.001, 0.01, 0.1, 1입니다. 각 헤드의 패널티와 선택 헤드는 보정 전 최소 개발 NLL로 정합니다. Temperature는 보정 데이터만 사용합니다. 표준화는 내보낸 가중치와 편향에 합쳐집니다. 최종 테스트 라벨은 하이퍼파라미터, 헤드, temperature 선택에 쓰지 않습니다. 생성된 `selection.json`과 `heads/training.json`에 선택·학습 세부 사항을 기록합니다.

모델 가중치, 원시 데이터, 특징, 학습된 아티팩트는 무시된 `models/`와 `results/`에 둡니다. 데이터셋·모델 약관은 이 저장소의 소스 코드 라이선스와 별개입니다.
