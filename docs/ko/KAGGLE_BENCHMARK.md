<a id="kaggle-ag-news-evaluation"></a>
# Kaggle AG 뉴스 평가

[English](../en/KAGGLE_BENCHMARK.md) · [한국어](KAGGLE_BENCHMARK.md) · [日本語](../ja/KAGGLE_BENCHMARK.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

이는 합성 규칙 픽스처 대신 자연 뉴스 기사에 대한 기존 판단 엔진을 평가합니다. 이는 훈련된 Kaggle 대회 제출이 아닌 제로샷 분류 실험입니다.

<a id="data-and-frozen-protocol"></a>
## 데이터 및 동결 프로토콜

출처: Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset), 버전 2의 [AG 뉴스 분류 데이터 세트. 다운로드한 아카이브에는 120,000 훈련 행과 7,600 테스트 행이 포함되어 있으며 세계, 스포츠, 비즈니스 및 과학/기술의 4가지 클래스가 있습니다.

준비 스크립트는 다운로드된 아카이브 SHA256을 확인하고, 정규화된 제목 및 설명이 교육 기사와 일치하는 테스트 기사를 제거하고, 중복 테스트 텍스트를 제거하고, 시드 `20260921`가 있는 클래스당 샘플 100 테스트 기사를 제거합니다. 이 아카이브에는  10 테스트 기사가 교육과 겹쳤습니다. 동결 평가에는 400 고유 기사와 25%의 항상 단일 클래스 기준이 있습니다.

훈련 예제, 데모, 라벨 또는 예상 답변이 모델로 전송되지 않습니다. 요청에는 제목, 설명, 고정된 지침 및 동일한 4가지 범주 설명만 고정된 순서로 포함됩니다. 예상되는 레이블은 `selection.json`에 별도로 저장됩니다. 샘플, 프롬프트, 범주 순서 및 임계값은 추론 전에 고정되었으며 결과에 맞춰 조정되지 않았습니다. 공개 과거 데이터가 모델 사전 학습에 나타날 수 있습니다. 이 데이터세트의 학습/테스트 중복을 제거해도 사전 학습 오염이 없음을 입증할 수는 없습니다.

각 로컬 GGUF는 CUDA, 컨텍스트 2048, 배치 256 및 4개의 CPU 스레드를 사용하여 동일한 400 아티클에 대해 한 번 실행됩니다. 모델은 순차적으로 실행됩니다. 기본 판단 보류 임계값은 변경되지 않습니다. 상위 후보 확률 0.8 및 전체 어휘 후보 확률 질량 0.05. 추론 코드는 변경되지 않습니다. Qwen3.8는 UD-IQ2_XXS를 사용합니다. Gemma 3 및 4는 Q8_0을 사용합니다. GPT-OSS는 MXFP4를 사용합니다. 이는 원래 모델군을 통제된 비교가 아닌 다양한 모델 크기와 양자화입니다.

<a id="metrics"></a>
## 측정항목

- **Cordirect / all:**는 모든 400 기사로 나눈 올바른 예측을 허용했습니다. 판단 보류, 오류 및 누락된 출력은 분모에 남아 있습니다.
- **Accepted 정답률:**는 허용된 올바른 예측을 허용된 예측으로 나눕니다. 모든 예측이 판단 보류하는 경우 정의되지 않습니다.
- **Coverage:**에서 허용된 예측을 400로 나눕니다.
- **Raw top-1 정답률:** 애플리케이션이 판단 보류하더라도 라벨에 대한 최고 확률 후보입니다. 동점이 잘못되었습니다. 이는 애플리케이션의 반환 응답 정답률이 아닙니다.
- **혼란 행렬:** 클래스별 예측, 판단 보류, 오류 및 누락된 결과.
- **95% 윌슨 구간:** 보고된 각 비율에 대한 설명적 불확실성, 이 샘플/프로토콜에 따른 조건; 다른 도메인이나 모델 학습 오염에 대한 보장은 아닙니다.
- **Latency:** 엔드 투 엔드 `decide` 시간(프롬프트 구성, 미리 채우기, logits 전송 및 점수 매기기 포함)(모델 로드 제외). 첫 번째 기사가 포함되어 있습니다. 생성된 토큰은 측정되지 않습니다.

모든 기사 결과는 즉시 JSONL로 플러시됩니다. 모델에 30분 시간 제한이 있습니다. 부분 결과에는 누락된 분모가 유지됩니다. 실패하거나 불완전한 실행은 완료된 벤치마크로 표시되어서는 안 됩니다. 기존 출력 파일은 덮어쓰지 않습니다.

<a id="reproduce"></a>
## 재현

데이터 및 자세한 예측은 무시된 `results/` 디렉터리에 유지됩니다. 소스 저장소에는 포함되지 않습니다. Kaggle에서 정확한 아카이브를 다운로드하세요. 해시가 검토된 버전과 다르면 준비가 실패합니다.

```sh
mkdir -p results/kaggle-ag-news
curl --fail --location \
  https://www.kaggle.com/api/v1/datasets/download/amananandrai/ag-news-classification-dataset \
  --output results/kaggle-ag-news/dataset.zip
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools kaggle-ag-news prepare

cargo build --release --locked --features llama-cuda --example evaluate_jsonl
target/release/l2s1-tools kaggle-ag-news run
target/release/l2s1-tools kaggle-ag-news report
```

`--model gemma4`를 사용하여 하나의 모델만 실행하거나 `--model`를 반복합니다. 이전 결과를 보존하려면 `--folder`와 함께 새 디렉토리를 사용하십시오. Rust 명령은 `crates/l2s1-tools/src/kaggle_ag_news.rs`에 문서화된 기존 로컬 모델 파일 이름을 예상합니다. 가중치를 다운로드하지 않습니다.

준비된 소스 아카이브, 개별 CSV, 고정 요청, 선택한 행 ID, 모델 파일, 실행 파일 및 소스 파일은 `results/kaggle-ag-news/`에 SHA256 출처가 있습니다. `runtime.json`는 llama.cpp 개정판과 더티 작업 트리 상태를 기록합니다. Git 커밋만으로는 테스트된 모든 코드를 설명할 수 없습니다. `runner-used.py`는 보고서 생성 명령이 추가되기 전에 이 실험에 사용된 정확한 러너를 보존합니다.

평가자의 검증:

```sh
python3 -m unittest discover -s scripts -p test_kaggle_ag_news.py -v
cargo clippy --release --locked --features llama-cuda --example evaluate_jsonl -- -D warnings
```

채점 테스트에는 분모 처리, 모두 판단 보류한 출력, 중복/알 수 없는 ID, 분할 중복 제거, 결정론적 샘플링 및 신뢰 구간 경계 사례가 포함됩니다. [VERIFICATION.md](VERIFICATION.md)에서 모델별 검사를 별도로 실행하세요. 성공적인 고정 배치 분류 실행은 실행 모드 전반에 걸쳐 일관성을 설정하지 않습니다.
