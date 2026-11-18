# 로그 분류 경험과 평가 루프

`scripts/log_experience.py`는 CPU 모델의 로그 분류를 SQLite 또는 PostgreSQL에
저장합니다. 사람이 원인을 검토한 뒤 정답률을 평가하고 검토된 경험을 JSONL로
내보냅니다. 실행할 조치를 선택하거나 실행하지 않으며, 가중치 학습은 수행하지 않습니다.
Python 3.11 이상과 빌드된 L2S1 실행 파일, GGUF 모델이 필요합니다.

## SQLite로 시작

저장소 루트에서 실행합니다. 기본 DB는 `results/log-experience.sqlite3`입니다.

```sh
.venv/bin/python scripts/log_experience.py init
.venv/bin/python scripts/log_experience.py classify --log-file /path/to/error.log
.venv/bin/python scripts/log_experience.py list --pending
.venv/bin/python scripts/log_experience.py review EXPERIENCE_ID \
  --label network --note 'DNS 오류로 확인'
.venv/bin/python scripts/log_experience.py evaluate
.venv/bin/python scripts/log_experience.py export --output results/reviewed-logs.jsonl
```

라벨은 `network`, `dependency`, `compile`, `test`, `other`입니다.
`classify`의 출력 ID를 `review`에 전달합니다. 정답은 별도로 확인해서 입력해야 합니다.
실제 후속 조치 결과가 확인된 경우에만 `--action-success yes` 또는 `no`를
지정합니다. 기본값 `unknown`은 성공·실패 어느 쪽으로도 집계하지 않습니다.
조치 성공은 분류 정답과 별개이며 원인이나 인과적 보상을 입증하지 않습니다.

기본 모델은 `models/Qwen3-4B-Q4_K_M.gguf`, 실행 파일은 `target/release/l2s1`,
장치는 CPU, 스레드는 8개입니다. `--model`, `--binary`, `--threads`, `--timeout`으로
변경할 수 있습니다. 기본 수락 정책은 `--min-probability 0.8 --min-mass 0.05`입니다.
이는 정확도 보장이 아니며 높은 점수의 오답도 가능합니다. 보류는 정답으로 대체하지
않고 그대로 저장합니다. 추론 실패는 경험으로 기록하지 않고 오류로 종료합니다.

## PostgreSQL

운영자가 만든 전용 데이터베이스를 사용합니다. 계정에는 해당 스키마의 테이블 생성과
조회·삽입 권한이 필요합니다. 기존 데이터는 삭제하지 않습니다.

```sh
uv pip install --python .venv/bin/python -r scripts/requirements-log-experience.txt
# 연결 문자열은 비밀 관리 도구 또는 셸 환경에서 설정합니다.
export L2S1_EXPERIENCE_DB='postgresql://USER:PASSWORD@HOST:5432/DATABASE'
.venv/bin/python scripts/log_experience.py init
.venv/bin/python scripts/log_experience.py classify --log-file /path/to/error.log
.venv/bin/python scripts/log_experience.py evaluate
```

`--db`로 파일 경로나 PostgreSQL URL을 지정할 수도 있습니다. 공용 환경에서는
명령행 이력에 자격 증명을 남기지 않도록 환경 변수를 권합니다. 이 도구는 DB 서버나
계정, TLS, 백업을 설정하지 않습니다. 원격 DB는 배포 환경에 맞게 `sslmode`를 지정하세요.

## 저장과 평가

- 경험에는 원본 로그, 입력 SHA-256, 요청, 모델·런타임 식별 정보와 후보 점수를 포함한
  전체 진단 출력, 선택·보류, 원시 최고 후보, 실행 시간이 저장됩니다. 시간에는 모델
  로딩과 프로세스 시작도 포함되므로 순수 추론 지연과 구분해야 합니다.
- 피드백은 수정할 때 이전 기록을 덮어쓰지 않고 새 행으로 추가합니다. 평가에는 최신
  피드백만 사용합니다. 검토되지 않은 입력은 정확도 분모에 넣지 않습니다.
- `evaluate`는 원시 최고 후보 정답률, 수락률, 보류율, 수락한 판단의 정답률,
  혼동행렬과 별도 조치 성공률을 반환합니다. 분모가 없으면 `null`입니다.
  `--model-sha256 HASH`로 특정 모델만 평가할 수 있습니다.
- 내보내기는 검토된 경험만 포함합니다. 보류의 `classification_reward`는 `null`,
  선택한 정답은 1, 오답은 0입니다. 이는 학습 재료이며 강화학습 알고리즘이 아닙니다.
  기존 출력 파일은 덮어쓰지 않습니다.
- 동일 로그 해시를 같은 `train`/`validation` 파티션으로 묶어 단순 중복 누수를
  방지합니다(해시 기반 약 80/20). 유사 로그·같은 장애의 변형까지 분리하지는 않으므로
  실제 학습에서는 장애·프로젝트·시간별 분리와 상충하는 정답 검토가 추가로 필요합니다.
  반복 판단 수와 고유 입력 수를 구분하고, 충분한 미사용 평가 데이터를 확보하세요.

원본 로그와 피드백은 DB와 내보내기 파일에 평문으로 저장됩니다. 토큰·비밀번호·개인정보는
입력 전에 제거하세요. 수집 데이터는 기본적으로 Git에서 제외된 `results/`에 보관합니다.
이번 버전에서는 DB 스키마 버전 업그레이드나 자동 정책 갱신을 제공하지 않습니다.

## 검증

```sh
.venv/bin/python -m unittest discover -s scripts -p test_log_experience.py -v
# 별도 테스트 DB: 테스트마다 고유 스키마를 만들고 해당 스키마만 삭제합니다.
L2S1_TEST_POSTGRES_URL='postgresql://USER:PASSWORD@localhost/TEST_DATABASE' \
  .venv/bin/python -m unittest discover -s scripts -p test_log_experience.py -v
```
