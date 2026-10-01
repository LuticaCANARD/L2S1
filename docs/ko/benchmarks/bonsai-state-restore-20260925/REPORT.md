<a id="bonsai-request-local-state-restoration-on-rtx-3060"></a>
# Bonsai 요청 - RTX 3060에서 로컬 상태 복원

[English](../../../en/benchmarks/bonsai-state-restore-20260925/REPORT.md) · [한국어](REPORT.md) · [日本語](../../../ja/benchmarks/bonsai-state-restore-20260925/REPORT.md)

[English index](../../../en/README.md) · [한국어 색인](../../README.md) · [日本語索引](../../../ja/README.md)

드라이버 595.71.05가 포함된 NVIDIA GeForce RTX 3060 12 GiB, `lucatagpu`의 2026-09-25에서 측정되었습니다. 하이브리드 체크포인트는 `prefix-reuse` 모드에서 새로운 실행으로 돌아갑니다. 명시적 `state-restore` 모드는 공통 접두사의 전체 llama.cpp 시퀀스 상태를 저장하고 해당 상태에서 첫 번째 접미사를 실행하며 이후의 각 접미사에 대해 이를 복원합니다. 상태는 하나의 네이티브 요청 중에만 존재합니다. 스냅샷을 사용할 수 없으면 실행이 다시 새로 실행됩니다.

<a id="setup"></a>
## 설정

- L2S1는 커밋 `a12a99b`에서 빌드를 측정했습니다(`7be9a29`와 동일한 구현을 기반으로 함). CUDA 바이너리 SHA-256 `edeb9b1e61afb72a0db94d71201f6d0baf63e4cfd3eb3c60bb7cf3c1e4edbdde`. llama.cpp 소스 개정 `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.
- Ternary Bonsai 2 27B Q1_0 GGUF SHA-256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`, CUDA로 완전히 오프로드되었습니다.
- 고정된 [16-판단 창고 요청](../../../../benchmarks/bonsai-state-restore-20260925/request-16.json) 및 첫 번째 판단만. 픽스처는 공유 상태 접두사 및 선택 판단을 사용합니다. 플래그: `--device cuda --context 4096 --batch 256 --ubatch 256 --threads 8 --prompt-layout state-first`; 두 번째 경로는 `--execution-mode state-restore`를 추가합니다. 둘 다 전체 증거 전송을 사용합니다.
- 모드당 하나의 상주 모델이 있는 카운트 및 모드당 1개의 워밍업과 3개의 측정된 루프백 HTTP 요청입니다. 타이밍에는 HTTP 및 JSON 처리가 포함되지만 시작 및 모델 로드는 제외됩니다. 모드는 순차적으로 실행되었습니다. GPU 메모리는 각 실행 중에 샘플링되었습니다. [raw 요약](../../../../benchmarks/bonsai-state-restore-20260925/summary.json)에는 측정된 벽 시간과 결과 비교가 모두 포함되어 있습니다.

<a id="results"></a>
## 결과

| 모드 | 판단 1개 p50 | 판단 16개 p50 | 판단 16개 속도 향상 | 판단 16개의 재사용 프리픽스 토큰 | 최대 GPU 메모리 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh/가득한 | 757.0 ms | 12,207.9 ms | 1.00× | 0 | 4,309 MiB |
| 상태 복원/전체 | 766.1 ms | 6,976.4 ms | **1.75×** | 3,840 | 4,309 MiB |

3개의 16-판단 클라이언트 시간은 신규의 경우 12,156.0/12,207.9/12,226.3 ms이고 상태 복원의 경우 6,973.4/6,976.4/6,981.5 ms였습니다. 복원된 요청은 173,678,124 바이트 상태 스냅샷을 저장하고 이후 15 판단을 위해 15번 로드했습니다. 첫 번째 판단은 원래 미리 채워진 상태를 사용했습니다. 진단 타이밍은 338.7 ms 미리 채우기,  257.2 ms 저장,  1,225.7 ms 복원 및 5,158.5 ms 접미사 실행이었습니다. one-판단 요청에는 재사용된 토큰이 없습니다.

측정된 모든 요청에 대해 복원된 경로에는 **zero** 변경된 선택 사항과 원시 상위 선택 사항이 있었고 **zero** 옵션 확률과 신규 대비 후보 확률 질량의 최대 차이가 있었습니다. 모든 16 원시 최고 선택은 픽스처의 간단한 규칙과 일치했습니다. 기본 정책은 두 경로 모두에서 15/16 판단을 허용했습니다. 실제 모델 회귀는 옵션별 토큰 ID, logits, 확률, 후보 확률 질량, 값 및 판단 보류 이유를 추가로 비교했습니다. 이 체크포인트를 전달했습니다. 0바이트 스냅샷 제한을 사용하여 `snapshot_memory_budget`를 보고하고 재사용된 토큰 없이 새로 실행하여 동일한 결과를 생성했습니다. 정상 한도를 복원하면 다시 재사용이 허용됩니다.

별도의 [3-판단 CUDA 적합성 결과](../../../../benchmarks/bonsai-state-restore-20260925/conformance.json)는 Bonsai에 대한 바이너리, 선택 및 서열 판단을 다룹니다. 상태 복원은 167,384,364 바이트 스냅샷을 두 번 로드했으며, 옵션 확률과 후보 질량 차이가 0이고 변경된 최상위 선택이나 허용된 값이 fresh하지 않았습니다. 0바이트 제한으로 인해 다시 `snapshot_memory_budget`가 생성되고 0이 복원되었습니다. 이는 또 다른 프롬프트 모양을 확인합니다. 이는 라벨이 붙은 정답률 세트가 아닙니다.

<a id="reproduction-and-limits"></a>
## 재생산 및 한계

동일한 GGUF가 있는 CUDA 호스트에서 `L2S1_BONSAI_MODEL=/path/to/Bonsai-27B-Q1_0.gguf cargo test --release --locked --features llama-cuda --test state_restore_bonsai -- --ignored --nocapture`를 사용하여 무시된 회귀를 실행합니다. `L2S1_CONFORMANCE_MODELS=/path/to/Bonsai-27B-Q1_0.gguf SKID_CUDA=1 cargo test --release --locked --features llama-cuda --test conformance -- --ignored --nocapture`로 다른 계약을 실행하세요. CLI 진단의 경우 연결된 요청을 가리키는 `--input`와 함께 위의 플래그를 사용하고 `--diagnostics`를 추가합니다. 전체 벤치마크 출력 및 테스트 로그는 `~/personal/skid/state-restore-stable-20260925/results/` 아래의 측정된 호스트에 남아 있습니다.

이는 하나의 GGUF 및 GPU에서 하나의 합성 픽스처에 대한 실행 동등성 및 대기 시간입니다. 레이블이 지정된 작업 품질, 기타 하이브리드 체크포인트, 교차 요청 재사용 또는 운영 환경 대기 시간 분포는 측정하지 않습니다. 스냅샷 제한은 총 프로세스 메모리가 아닌 스냅샷 버퍼를 제어합니다. `state-restore`는 계속 옵트인 상태로 유지됩니다. `prefix-reuse`는 여전히 Bonsai에 대한 하이브리드 폴백을 보고합니다.
