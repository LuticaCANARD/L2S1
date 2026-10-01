<a id="l2s1-experiment-tools"></a>
# L2S1 실험 도구

[English](../../../en/crates/l2s1-tools/README.md) · [한국어](README.md) · [日本語](../../../ja/crates/l2s1-tools/README.md)

[English index](../../../en/README.md) · [한국어 색인](../../README.md) · [日本語索引](../../../ja/README.md)

`l2s1-tools`는 고정된 데이터세트를 준비하고, 로컬 GGUF 평가자를 실행하고, 저장된 예측을 독립적으로 계산하기 위한 저장소 전용 Rust CLI입니다. llama.cpp를 연결하지 않습니다. 추론을 수행하는 명령은 저장소의 `evaluate_jsonl` 예제를 시작합니다. 다음을 사용하여 빌드하세요.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools --help
```

저장소 루트에서 명령을 실행하거나 `--root /path/to/L2S1`를 전달합니다. 바이너리는 크레이트 게시에서 제외됩니다. 생성된 데이터를 기록하고 사용자가 선택한 경로(일반적으로 무시되는 `results/` 아래)에 보고합니다. 기존 출력 경로는 일반적으로 동결된 증거를 보존하기 위해 거부됩니다. 훈련 및 직접 PyTorch 모델 프로브는 Python에 남아 있습니다.

| 기존 Python 스크립트 | Rust 하위 명령 |
| --- | --- |
| `benchmark_models.py` | `benchmark-models` |
| `kaggle_ag_news.py` | `kaggle-ag-news prepare/run/report` |
| `kaggle_airline.py` | `kaggle-airline prepare/run/report` |
| `calibrate_ag_news.py` | `calibrate-ag-news` |
| `prepare_decision_finetune.py` | `prepare-decision-finetune` |
| `prepare_accuracy_study.py` | `prepare-accuracy-study` |
| `prepare_output_head.py` | `prepare-output-head` |
| `evaluate_decision_lora.py` | `evaluate-decision-lora` |
| `evaluate_intents.py` | `evaluate-intents` |
| `evaluate_intents_wide.py` | `evaluate-intents-wide` |
| `analyze_intent_rotation.py` | `analyze-intent-rotation` |
| `jevbench_public.py` | `jevbench-public prepare/score/run` |
| `jevbench_matrix.py` | `jevbench-matrix download/run` |
| `laya_benchmark.py` | `laya-benchmark fetch/prepare/score/compare/run` |
| `probe_airline_order.py` | `probe-airline-order` |
| `report_accuracy_study.py` | `report-accuracy-study` |
| `report_airline.py` | `report-airline` |
| `report_decision_finetune.py` | `report-decision-finetune` |
| `report_intents.py` | `report-intents` |
| `report_intents_wide.py` | `report-intents-wide` |
| `report_jevbench_matrix.py` | `report-jevbench-matrix` |
| `report_output_head.py` | `report-output-head` |
| `summarize_output_head_runtime.py` | `summarize-output-head-runtime` |
| `tune_compute.py` | `tune-compute` |

Rust로 대체된 Python 준비·평가·리포트 실행기 21개와 해당 구현 전용 테스트 7개를 삭제했습니다. `kaggle_ag_news.py`, `calibrate_ag_news.py`, `prepare_accuracy_study.py`는 Python 학습 코드가 해시·보정·고정 데이터 검증 도우미를 가져다 쓰므로 유지합니다. 이들의 테스트, PyTorch 학습·진단, 현재 비전 HTTP 실행기도 유지합니다. 과거 스크립트는 Git 이력에 남아 있습니다. 새 데이터 준비·벤치마크·리포트에는 Rust를 사용하세요.

고정된 정답률 연구 시드는 모든 18 고정 파일을 바이트 단위로 재생합니다. AG 뉴스, 항공사, 판단 미세 조정, 출력-헤드 및 의도 준비를 기존 로컬 아티팩트와 비교하여 확인했습니다. JevBench 및 Laya 준비 및 저장된 예측 점수도 고정된 공개 데이터와 비교하여 확인되었습니다. 이러한 검사는 새로운 CLI에 대한 새로운 모델 추론, 네트워크 다운로드 또는 GPU 실행을 설정하지 않습니다. 해당 경로에는 해당 호스트 및 모델 파일이 필요합니다.

일부 JSON 보고서는 부동 소수점 감소 순서 또는 키 형식만 다릅니다. 보고서에는 수락률, 실패 및 후보 확률에 대한 별도의 분모가 포함됩니다. 로컬 합성/공개 점수는 운영 환경 정답률 청구가 아닙니다. JevBench는 고정된 공개 득점자의 Rust 기록을 사용하며 명시적으로 전체 공식 종합 점수를 계산하지 않습니다. Laya 프로브 케이스는 고정된 `probe.py` 정의의 작은 고정 렌더링입니다. 준비에서는 다운로드한 소스 해시를 사용하기 전에 확인합니다.
