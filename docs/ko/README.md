# 문서 색인

[English](../en/README.md) · [한국어](../ko/README.md) · [日本語](../ja/README.md)

빌드와 첫 판단 실행은 [한국어 README](../../README.ko.md)부터 시작하세요. 이 색인은 모든 공개 부속 가이드, 패키지 문서, 벤치마크 보고서를 안내합니다. 모든 가이드·패키지 문서·벤치마크 보고서의 영어·한국어·일본어 본문을 제공합니다. 문서의 언어 링크로 같은 문서의 다른 언어판으로 이동할 수 있습니다. 코드·명령·식별자·측정값은 원문 값을 유지합니다.

## 사용과 통합

| 문서 | 내용 |
| --- | --- |
| [사용 가이드](GUIDE.md) | 빌드, 스키마, 점수, CLI, Rust, HTTP, 이미지, 백엔드 |
| [TypeScript 라이브러리](typescript/README.md) | 상주 모델 프로세스, HTTP 클라이언트, 패키징, 검증 |
| [Python SDK](python/README.md) | 타입이 있는 비동기 호출, 고정 질문 재사용, wheel, 공통 Rust 런타임 |
| [배치 API 검토](BATCHING_API_REVIEW.md) | 반복 상태 입력, native 배치 경계와 HTTP 연결 방안 |
| [배포 파이프라인](RELEASE_PIPELINE.md) | GitHub Release·npm·PyPI·native Cargo 자동 게시 |
| [AI 에이전트 통합](AGENT_INTEGRATION.md) | 범용 스킬, stdio MCP, 요청 검증, 상주 추론 |
| [브라우저 WebGPU 데모](WEBGPU_DEMO.md) | 로컬 Qwen3 ONNX 추론, WebGPU, 제한된 사고, 정책 |
| [이미지·텍스트 데모](IMAGE_DEMO.md) | 기록된 응답, 정책, 실패 설명, 로컬 서버 |
| [Pages 배포](PAGES_DEPLOYMENT.md) | Cloudflare Pages와 n2s1.luticalab.net DNS |
| [모델 교체](MODEL_INTERCHANGEABILITY.md) | 모델 식별, 사전 검증, 보정, 진단, 워커, 메모리 |
| [검증](VERIFICATION.md) | 빌드와 모델별 검증 명령 |
| [라이선스](LICENSING.md) | 소스, 의존성, 모델 라이선스의 구분 |

## 실행과 특화

| 문서 | 내용 |
| --- | --- |
| [추론 모드](REASONING.md) | 직접 판단·네이티브 사고, 토큰 한도, 런타임 범위 |
| [병렬 실행](PARALLEL_EXECUTION.md) | 텍스트·이미지 배치, 동적 컨텍스트, 프로젝터 재사용, 비전 프로파일 |
| [프리픽스 알고리즘](SEMIF_ALGORITHM.md) | 프리픽스 준비와 재사용 |
| [판단 미세 조정](DECISION_FINETUNE.md) | LoRA 작업 절차와 기록된 평가 |
| [출력 헤드](OUTPUT_HEAD.md) | 작업별 헤드와 아티팩트 연결 |

## 평가 방법과 결과

| 문서 | 내용 |
| --- | --- |
| [모델 측정 결과](MODEL_RESULTS.md) | 체크포인트 비교, 범위, 측정 한계 |
| [합성 벤치마크](BENCHMARK.md) | 규칙 기반 픽스처의 정답, 판단 보류, 일관성, 지연 시간 |
| [JevBench](JEVBENCH.md) | 공개 작업 매핑, 원본 채점, 수락 지표 |
| [Ollaya 비교](OLLAYA_COMPARISON.md) | 근거에 기반한 차이와 남은 품질·제품 격차 |
| [typed-decisions](TYPED_DECISIONS_BENCHMARK.md) | 전체 2,000개 판단 테스트, 보정, 측정 절차 |
| [의도 분류](INTENT_BENCHMARK.md) | 넓은 답변 코드, BANKING77, MASSIVE 한국어 |
| [AG News](KAGGLE_BENCHMARK.md) | 고정된 분류 절차 |
| [Laya/Jev 작업과 캐시](LAYA_BENCHMARK.md) | 작업 변환과 CPU·GPU 캐시 검증 |
| [비전 벤치마크](VISION_BENCHMARK.md) | 직접 이미지 추론과 시간 측정 범위 |

## 작업·하드웨어 보고서

| 문서 | 내용 |
| --- | --- |
| [Caltech-101](benchmarks/caltech101-vision-20260924/README.md) | 정지 이미지 분류 |
| [고양이와 개](benchmarks/cats-dogs-vision-20260924/REPORT.md) | 이진 이미지 분류 |
| [TrashNet과 비전 처리량](benchmarks/trashnet-vision-20260925/REPORT.md) | 이미지 분류와 실행 모드 |
| [범용 GGUF CUDA 스모크](benchmarks/gguf-cuda-20260925/README.md) | CUDA 모델 스모크 측정 |
| [Bonsai 상태 복원](benchmarks/bonsai-state-restore-20260925/REPORT.md) | 상태 복원 측정 |
| [공유 상태 캐시](benchmarks/shared-state-cache-20260925/REPORT.md) | 캐시 측정 |
| [빌드 캐시](benchmarks/build-cache-20260925/REPORT.md) | 네이티브 빌드 캐시 측정 |
| [typed-decisions 아티팩트](benchmarks/typed-decisions-20260926/README.md) | 저장된 요약, 기록, 재현 메타데이터 |

## 소스와 도구

| 문서 | 내용 |
| --- | --- |
| [아키텍처와 구성 요소](GUIDE.md#architecture) | 소스 구성 |
| [네이티브 llama.cpp 의존성](crates/l2s1-llama-sys/README.md) | 고정된 네이티브 런타임과 브리지 |
| [데이터셋·보고서 도구](crates/l2s1-tools/README.md) | 데이터셋 준비, 평가, 재집계 |
| [창고 요청 예제](../../examples/warehouse.json) | 이진·선택·서열 요청 |
| [문서 웹사이트](web/README.md) | Svelte 사이트, 데모, 공개 문서 내보내기 |
| [npm 공개 배포](typescript/PUBLISHING.md) | 플랫폼 런타임 패키징, 공개 조건, 검증 범위 |
| [실험 도구 고지](crates/l2s1-tools/NOTICE.md) | JevBench·Laya 출처와 데이터셋 범위 |
| [L2S1 스킬](skills/l2s1/SKILL.md) | 통합 지침과 근거 처리 |
| [스킬 요청 참조](skills/l2s1/references/decisions.md) | 요청 예제, 타입 결과, HTTP·MCP 필드 |
| [스킬 인터페이스 참조](skills/l2s1/references/interfaces.md) | 빌드, CLI, HTTP, MCP, Rust 통합 |

소스·의존성 고지: [LICENSE](../../LICENSE), [THIRD_PARTY_LICENSES.txt](../../THIRD_PARTY_LICENSES.txt). 모델 가중치는 별도로 준비하며 각각의 약관을 따릅니다. 보고서는 명시된 범위의 실험 기록입니다. 정답률, 수락된 판단의 정답률, 수락률을 구분하세요.
