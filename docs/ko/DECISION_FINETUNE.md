<a id="supervised-decision-lora-pilot"></a>
# 감독된 판단 LoRA 파일럿

[English](../en/DECISION_FINETUNE.md) · [한국어](DECISION_FINETUNE.md) · [日本語](../ja/DECISION_FINETUNE.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

이 실험은 기존 항공 감정 판단 인터페이스에 대해 Gemma 4 E2B IT를 교육합니다. 이는 유형화된 확률적 판단의 공공 목표를 따릅니다. TypeSafe의 독점 RLCD 또는 Jev 아키텍처를 재현하지 않습니다. 새로운 병렬 주의 아키텍처를 추가하지 않으며 일반적인 판단 기능을 주장하지 않습니다.

<a id="frozen-protocol"></a>
## 동결 프로토콜

- 출처: `target/release/l2s1-tools kaggle-airline`가 준비한 고정된 Kaggle Twitter US Airline Sentiment 아카이브. 데이터 세트 텍스트 및 가중치는 무시된 `results/`에 남아 있습니다. 이 저장소에서는 재배포되지 않습니다.
- 정규화된 텍스트로 모든 800 이전 보정/평가 예제를 제외합니다. 정확하게 정규화된 중복 항목과 충돌하는 라벨 텍스트 그룹을 제거합니다.
- Seed 20260923: 900 고유 교육 트윗(클래스당 300),  400 새로운 보정 트윗 및 400 새로운 테스트 트윗. 분할은 정규화된 텍스트에 의해 분리됩니다. 이는 자연적인 클래스 확산이 아닌 의도적으로 균형을 맞춘 단일 도메인 샘플입니다.
- 각 훈련 트윗은 세 가지 순환 옵션 순서(2,700 훈련 입력) 모두에 나타납니다. 고정된 60 케이스 테스트 하위 집합은 세 가지 순서 모두에서 평가됩니다(180 진단 호출, 이는 추가 독립 케이스가 아님).
- `export_decision_tokens`는 실제 Rust/GGUF 프롬프트 렌더러 및 토크나이저를 사용합니다. Transformers는 운영 환경 프롬프트를 다시 구현하지 않고 정확한 입력 및 후보 토큰 ID를 사용합니다. 라벨은 별도로 저장됩니다.
- 기본 모델: `google/gemma-4-E2B-it`, 개정판 `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. 교육에서는 bf16 컴퓨팅과 함께 NF4 기반을 사용하고 bf16 및 fp32 정규화/어댑터의 고정된 대형 행렬을 사용합니다.
- LoRA 순위 8, 알파 16, 드롭아웃 0, 언어 모델 주의 Q/V 투영 전용입니다. 한 시대, 마이크로배치 1, 축적 12, 학습률 0.0001가 0으로 선형적으로 감소하고, 가중치 감소가 0인 AdamW, 1에서 그래디언트 노름 클리핑.
- 판단 위치에서의 손실: 후보 교차 엔트로피에 0.1 곱하기 음수 로그 전체 어휘 후보 확률 질량. 프롬프트 토큰이나 생성된 텍스트에는 손실이 없습니다.
- 최종 체크포인트, 테스트 기반 체크포인트/하이퍼매개변수 선택이 수정되었습니다. 훈련 전용 스모크 실행은 GPU/메모리/그라디언트 호환성을 먼저 확인합니다. 해당 어댑터는 폐기되고 실제 실행은 원래 베이스에서 시작됩니다.
- 각 체크포인트의 온도를 새로운 보정 분할에만 맞춥니다. 임계값 0.6–1.0에서 원시 top-1, NLL, Brier, ECE, 수락률 및 수락된 판단의 정답률을 보고합니다. 후보 확률 질량은 변경되지 않고 0.05 게이트는 활성 상태로 유지됩니다.
- 동일한 런타임/정밀도에서 기본 체크포인트와 훈련된 체크포인트를 비교합니다. Transformers NF4와 llama.cpp Q8을 별개의 비교 대상으로 간주합니다. 전환 효과는 훈련 이득이 아닙니다. 이 파일럿에서는 애플리케이션 기본값이 변경되지 않습니다.
- GGUF 베이스 및 어댑터에 대한 추가 도메인 외부 회귀 검사로 이전 AG News 400 사례 개발 벤치마크를 실행합니다. 손대지 않은 새로운 테스트 세트가 아니며 이 실험의 어댑터나 온도에 맞지 않습니다.

공개 사전 훈련 노출 및 거의 중복된 텍스트는 제외되지 않습니다. 단일 훈련 시드는 무작위 초기화 전반에 걸쳐 재현성을 설정할 수 없습니다. 새로운 테스트 데이터는 동일한 과거 데이터 세트에서 나오며 도메인 외부 또는 운영 환경 정답률을 설정하지 않습니다. 현재 API 보정은 오프라인 아티팩트이며 애플리케이션 응답에 자동으로 적용되지 않습니다.

<a id="entry-points"></a>
## 진입점

1. `target/release/l2s1-tools prepare-decision-finetune`는 데이터와 프로토콜을 동결합니다.
2. `examples/export_decision_tokens.rs`는 운영 환경 입력/후보 ID를 내보냅니다.
3. `scripts/train_decision_lora.py`는 고정된 모델을 다운로드하고, 스모크 테스트 또는 전체 쌍 실험을 실행하고, 어댑터와 원시 측정값을 저장합니다.
4. `target/release/l2s1-tools report-decision-finetune`는 보정 전용 온도에 적합하며 보유 측정항목 및 옵션 주문 진단을 보고합니다.

CLI 및 JSONL 평가자는 `--lora path/to/adapter.gguf`를 허용합니다. 일치하는 llama.cpp `convert_lora_to_gguf.py --base <config-dir>` 도구를 사용하여 PEFT 어댑터를 변환합니다. 1 규모의 백엔드당 하나의 어댑터가 지원됩니다. 병렬 실행으로 인해 컨텍스트 크기가 조정될 때마다 다시 연결됩니다. 응답 메타데이터는 `lora_path`를 기록합니다. 이 옵션이 없으면 기존 기본 모델 경로는 변경되지 않습니다. 어댑터를 로드해도 온도 보정이 자동으로 적용되지 않습니다.

생성된 아티팩트, 종속성 잠금 및 정확한 실행 스크립트를 무시된 로컬 `results/` 디렉터리에 보관하세요.

새 아티팩트 디렉터리의 경우(새 테스트 세트가 아닌 동일한 분할을 재현함):

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools prepare-decision-finetune \
  --source results/kaggle-airline-20260922 --output results/decision-pilot-new/data
for split in train calibration test probe; do
  target/release/examples/export_decision_tokens \
    --model models/gemma-4-E2B-it-Q8_0.gguf \
    --input "results/decision-pilot-new/data/$split.jsonl" \
    --output "results/decision-pilot-new/data/$split-tokens.jsonl"
done
target/release/l2s1-tools prepare-decision-finetune \
  --output results/decision-pilot-new/data --seal-tokens
# Run in the locked training environment on the GPU host:
python scripts/train_decision_lora.py --data results/decision-pilot-new/data \
  --output results/decision-pilot-new/pilot --cache results/decision-pilot-new/hf-cache
target/release/l2s1-tools report-decision-finetune --data results/decision-pilot-new/data \
  --run results/decision-pilot-new/pilot
```

<a id="frozen-synthetic-accuracy-study"></a>
## 냉동 합성 정답률 연구

새로운 연구는 항공사 조종사 및 이전 창고 설비와는 별개입니다. 6개 도메인에 걸쳐 480 train, 120 dev, 120 보정 및 180 테스트 논리 사례를 생성합니다. 임계값과 문구 템플릿 모음은 모두 분할에서 분리됩니다. 각 사례에는 자연어와 기호 판단 기준가 쌍을 이룹니다. 이는 독립 표본이 아닌 하나의 사례에 대한 두 가지 표현입니다. 선택, 바이너리 및 서열 타겟은 정확한 경계와 인접한 정수/소수 경계를 포함하여 균형을 이루고 있습니다. 서열 값은 정렬된 순서를 유지합니다.

다음 워크플로에는 구성된 네이티브 빌드와 위에 고정된 개정의 기존 로컬 Gemma 4 E2B IT 체크포인트가 필요합니다. 교육에는 호환되는 PyTorch, Transformers, 비트샌드바이트 및 PEFT가 포함된 GPU Python 환경이 추가로 필요합니다. 실험을 통해 종속성 버전을 기록합니다. 모든 데이터, 토큰 내보내기, 교사 응답, 어댑터 및 보고서를 무시된 `results/`에 보관합니다. 무시된 `models/` 아래에 기본 GGUF 가중치를 유지합니다. 새 스크립트는 출력 파일/디렉터리 덮어쓰기를 거부하고 체크포인트를 다운로드하지 않습니다.

**dev 전용**에 대한 연구 준비 및 프롬프트 구성 비교:

```sh
study=results/accuracy-study-new
model=models/gemma-4-E2B-it-Q8_0.gguf
checkpoint=/path/to/local/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7
variant=natural
mkdir -p "$study"
target/release/l2s1-tools prepare-accuracy-study --output "$study/data"
target/release/l2s1-tools prepare-accuracy-study --output "$study/data" --verify
cargo build --release --locked --features llama-cuda \
  --example evaluate_accuracy --example export_decision_tokens

target/release/examples/evaluate_accuracy --model "$model" --cuda \
  --input "$study/data/dev-$variant-requests.jsonl" \
  --output "$study/dev-$variant-predictions.jsonl" \
  --prompt-details minimal,typed,typed-examples --layouts legacy,state-first \
  --all-rotations
target/release/l2s1-tools report-accuracy-study --data "$study/data" \
  --predictions "$study/dev-$variant-predictions.jsonl" \
  --split dev --variant "$variant" --output "$study/dev-$variant-report.json" \
  --select "$study/dev-$variant-selection.json"
```

쌍을 이루는 표현 실험에는 `symbolic`를 사용합니다. 두 변형을 모두 비교하는 경우 보정/테스트 예측을 읽기 전에 두 개발 보고서를 모두 완료하고 승리한 변형을 고정하세요. 선택은 원시 top-1 순위를 매긴 다음 모든 사례에 대해 허용된 올바른 비율, 결정론적 최종 타이 브레이크를 사용하여 계산 경로 대기 시간 중앙값을 매깁니다. 전체 분할을 포괄하는 구성만 적합합니다. 회전 2는 세 가지 옵션 작업만 포함하며 전체 분할 구성에 대해 승리할 수 없습니다. `--select`는 보정을 거부하고 분할을 테스트합니다. 모델 로딩 및 출력 직렬화는 타이밍에서 제외됩니다. 이는 종단 간 서비스 대기 시간이 아닙니다.

고정된 선택 사항을 읽고 코드 회전을 포함하여 **훈련 요청만**로 내보냅니다. 내보내기는 운영 환경 GGUF 토크나이저를 사용합니다. 트레이너의 `--detail`는 밑줄을 사용합니다. 네이티브 CLI 열거형 값은 하이픈을 사용합니다.

```sh
selection="$study/dev-$variant-selection.json"
detail=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["setting"]["prompt_detail"])' "$selection")
detail_cli=$(python3 -c 'import sys; print(sys.argv[1].replace("_", "-"))' "$detail")
layout=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["setting"]["prompt_layout"].replace("_", "-"))' "$selection")
target/release/examples/export_decision_tokens --model "$model" \
  --input "$study/data/train-$variant-requests.jsonl" \
  --output "$study/train-tokens.jsonl" --all-rotations \
  --prompt-detail "$detail_cli" --prompt-layout "$layout"

python3 - "$study" "$variant" "$detail" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
seal = dict(schema_version=1, split='train', variant=sys.argv[2], detail=sys.argv[3],
            manifest_sha256=digest(root/'data/manifest.json'),
            token_sha256=digest(root/'train-tokens.jsonl'),
            hf_model='google/gemma-4-E2B-it',
            hf_revision='3e22461f65e89153144f8adb70e3b8c2cc9845a7')
with (root/'train-token-seal.json').open('x') as out:
    json.dump(seal, out, sort_keys=True, indent=2)
    out.write('\n')
PY
```

사고 교사 답변을 생성하고 폐기된 스모크 검사를 실행한 다음 새로운 어댑터를 교육합니다. 독립적인 **train** 라벨과 일치하는 교사의 최종 답변만 인정됩니다. 생성된 근거는 기록되지만 결코 학생 목표로 사용되지 않습니다. 누락되거나 잘못된 교사 답변은 오라클 라벨로 대체되지 않습니다. 동일한 허용 예제와 프로토콜이 스모크와 최종 교육을 바인딩해야 합니다.

`--teacher-batch-size`는 왼쪽 패딩 교사 생성을 활성화합니다(기본값 1). 각 답변은 검증 전 첫 번째 EOS에서 독립적으로 잘립니다. 별도의 제한된 파일럿을 사용하여 GPU 메모리 크기를 조정합니다. 교사 보고서는 최대 CUDA 할당 및 배치 타이밍을 기록합니다. 불완전한 파일럿은 교육 입력으로 허용되지 않습니다.

```sh
train_stage() {
  python3 scripts/train_accuracy_lora.py "$@" \
    --data "$study/data" --variant "$variant" --detail "$detail" \
    --tokens "$study/train-tokens.jsonl" --token-seal "$study/train-token-seal.json" \
    --checkpoint "$checkpoint"
}
train_stage teacher --output "$study/teacher" --max-new-tokens 384
train_stage smoke --teacher "$study/teacher" --output "$study/smoke"
train_stage train --teacher "$study/teacher" \
  --smoke-report "$study/smoke/complete.json" --output "$study/train"
python3 "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" "$study/train/adapter" \
  --base "$checkpoint" --outfile "$study/adapter.gguf" --outtype f16
```

이 변환 명령은 `LLAMA_CPP_DIR`에서 전체 llama.cpp 체크아웃이 필요합니다. 번들로 제공되는 네이티브 빌드 스냅샷에는 변환 도구가 생략되어 있습니다. 추론 빌드는 `L2S1_LLAMA_CPP_SOURCE` 또는 레거시 `LLAMA_CPP_DIR` 재정의가 설정되지 않은 한 번들 소스를 사용합니다.

성공적인 훈련 손실, 스모크 확인 또는 변환은 네이티브 정답률 개선을 설정하지 않습니다. 동일한 GGUF, 컴퓨팅 구성 및 고정된 프롬프트 설정을 기본으로 사용하여 변환된 어댑터를 평가합니다. 예를 들면:

```sh
target/release/examples/evaluate_accuracy --model "$model" --cuda \
  --lora "$study/adapter.gguf" \
  --input "$study/data/test-$variant-requests.jsonl" \
  --output "$study/test-$variant-adapter.jsonl" \
  --prompt-details "$detail_cli" --layouts "$layout" --all-rotations
target/release/l2s1-tools report-accuracy-study --data "$study/data" \
  --predictions "$study/test-$variant-adapter.jsonl" \
  --split test --variant "$variant" --output "$study/test-$variant-adapter-report.json"
```

`--lora` 없이 동일한 네이티브 평가를 별도의 기본 출력 파일로 실행합니다. 이전에 고정된 단일 회전 또는 앙상블만 비교합니다. 추가 회전 행은 테스트 시 선택할 수 있는 새로운 기회가 아니라 진단입니다. 보정 또는 판단 보류 임계값을 맞추는 경우 최종 테스트 평가 전에 보정 분할을 사용하여 해당 작업을 완료하세요. 기자 자체도 둘 다에 맞지 않습니다. 원시 top-1, NLL/Brier, 수락률, 수락된 판단의 정답률, Accepted-corright/all 및 회전 감도를 함께 보고합니다. NF4 교사/훈련 결과와 GGUF 어댑터 결과는 런타임/정밀도 경계가 다릅니다. 이 워크플로는 자체적으로 측정된 정답률 또는 성능 주장을 하지 않습니다.
