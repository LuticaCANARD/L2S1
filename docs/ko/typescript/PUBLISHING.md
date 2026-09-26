<a id="npm-publishing-review"></a>
# npm 공개 배포 검토

[English](../../en/typescript/PUBLISHING.md) · [한국어](PUBLISHING.md) · [日本語](../../ja/typescript/PUBLISHING.md)

[English index](../../en/README.md) · [한국어 색인](../README.md) · [日本語索引](../../ja/README.md)

래퍼와 플랫폼 런타임은 공개 npm 패키지로 배포할 수 있습니다. 구현은 일반 npm tarball, ESM export, TypeScript 선언, 선택적 플랫폼 의존성을 사용합니다. 최종 사용자는 Rust·C++ 컴파일러가 필요하지 않습니다. 실제 공개는 네이티브 플랫폼 빌드 통과와 npm scope·배포 권한 확보를 전제로 합니다.

<a id="package-layout"></a>
## 패키지 구성

| 패키지 | 내용 | 런타임 |
| --- | --- | --- |
| `@l2s1/node` | JavaScript, 선언, 소스, README, MIT 라이선스 | Node.js 22 이상. HTTP 하위 경로는 브라우저 번들로 사용 가능 |
| `@l2s1/runtime-linux-x64` | Rust 실행 파일과 대응 공유 라이브러리 | CPU, glibc Linux x64 |
| `@l2s1/runtime-linux-arm64` | Rust 실행 파일과 대응 공유 라이브러리 | CPU, glibc Linux arm64 |
| `@l2s1/runtime-darwin-x64` | Rust 실행 파일과 대응 공유 라이브러리 | CPU, macOS x64 |
| `@l2s1/runtime-darwin-arm64` | Rust 실행 파일과 대응 공유 라이브러리 | CPU·Metal, macOS arm64 |
| `@l2s1/runtime-win32-x64` | Rust 실행 파일과 대응 DLL | CPU, Windows x64 |

래퍼는 정확히 일치하는 런타임 버전을 참조합니다. `os`, `cpu`, Linux `libc` 메타데이터는 호환되지 않는 선택 패키지의 설치를 막습니다. 런타임 선택은 패키지 식별, 버전, 요청한 장치를 검증합니다. 런타임이 없으면 명시적인 오류를 반환합니다. HTTP·사용자 정의 백엔드는 런타임을 설치하지 않아도 동작합니다. CUDA는 별도로 빌드한 실행 파일을 `binaryPath`로 제공해 사용할 수 있습니다.

래퍼와 생성된 런타임 모두 `publishConfig.access = public`과 npm registry URL을 사용합니다. npm 문서에 따르면 scoped 패키지는 기본적으로 비공개이므로 공개 배포에는 public 접근을 명시해야 합니다. [npm scoped 패키지 문서](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages/)

<a id="release-conditions"></a>
## 릴리스 조건

1. npm `@l2s1` 사용자·조직 scope의 소유권 또는 배포 권한을 확인합니다. GitHub 저장소 소유자가 npm scope의 소유자인 것은 아닙니다. 인증 없는 registry 404는 scope가 비어 있다는 증거가 아닙니다.
2. 릴리스 커밋에 `v*` 버전 태그를 붙여 푸시하거나 해당 커밋의 워크플로를 수동 실행해 검사를 시작합니다. 워크플로 네 개는 버전 태그 푸시·수동 실행에서만 시작하며 PR·브랜치 푸시는 CI를 시작하지 않습니다. `typescript-runtimes.yml`의 네이티브 빌드·설치 작업 5개를 통과합니다. 로컬 Linux 빌드는 macOS·Windows·arm64 아티팩트의 검증이 아닙니다.
3. 모든 tarball을 `npm publish --dry-run --access public --ignore-scripts`로 확인합니다. 버전, 파일, 라이선스, 모델 가중치 부재, 설치 스크립트 부재를 확인합니다.
4. 같은 버전의 런타임 tarball 5개를 먼저 공개하고 래퍼를 공개합니다. 래퍼가 먼저 공개되면 npm의 선택 의존성 처리로 인해 로컬 추론이 불가능한 설치가 생길 수 있습니다.
5. 깨끗한 프로젝트에서 registry 이름으로 래퍼를 설치하고 자동 런타임 선택과 실제 모델 판단을 검증합니다. 최종 registry 검증에는 실제 공개가 필요하며 로컬 tarball 설치는 준비 근거입니다.

공개 scoped 배포에는 npm 계정, scope 권한, 지원되는 배포 인증이 필요합니다. 2026-09-26 확인 기준 로컬 npm CLI는 인증되지 않았고(`npm whoami`는 `ENEEDAUTH`), 인증 없는 `npm view @l2s1/node`는 404를 반환합니다. 이 검토에서 패키지를 공개하지 않았습니다.

CI 배포에서는 npm이 OIDC 기반 GitHub Actions trusted publishing을 지원합니다. npm 11.5.1 이상, Node.js 22.14.0 이상, 해당 패키지의 trusted-publisher 설정, 워크플로의 `id-token: write`가 필요합니다. 이 PR은 검토 가능한 아티팩트만 빌드하며 자동 배포 트리거를 추가하거나 npm 계정을 설정하지 않습니다. [npm trusted publishing 문서](https://docs.npmjs.com/trusted-publishers/)

<a id="binary-and-license-boundaries"></a>
## 실행 파일과 라이선스 범위

네이티브 패키지는 프로젝트 MIT 라이선스, 제삼자 고지, SHA-256 파일 목록을 포함합니다. 포함된 엔진·헤더·브리지·공유 라이브러리는 같은 고정 llama.cpp 소스로 빌드합니다. 모델 가중치는 포함하지 않으며 각 약관에 따라 따로 준비합니다. HTTP 전용 래퍼는 Node·브라우저 플랫폼 API를 사용하며 제삼자 JavaScript 런타임 라이브러리를 추가하지 않습니다.

Linux 빌드 작업은 Ubuntu 22.04를 사용합니다. 대응하는 glibc·C++ 런타임 요구사항을 유지하고 릴리스 전에 지원한다고 명시한 가장 오래된 배포판을 테스트하세요. Alpine·musl은 포함된 대상이 아닙니다. Windows는 Microsoft Visual C++ x64 런타임이 필요합니다. macOS 배포·Metal 지원은 해당 네이티브 작업으로 입증해야 합니다. 일반 CPU 커널은 빌드 호스트의 ISA에 의존하지 않으며, 호스트 최적화 빌드와 성능이 다를 수 있습니다.

기본 main 브랜치의 HTTP 계약은 상태·판단·미디어를 지원합니다. 선택적 요청 reasoning·policy 확장은 앱이 선택하고 서버가 지원할 때만 전달합니다. 오래된 서버는 미지원 필드를 거부합니다. 시작 정책은 `L2S1.load({ policy })`로 계속 사용할 수 있습니다.

<a id="verification-scope"></a>
## 검증 범위

릴리스 CI는 TypeScript와 예제 타입, HTTP 전송, 사용자 정의 백엔드 라우팅, 시작 실패·취소·정리, 실제 Rust HTTP 검증·점수 계산, 압축한 래퍼·런타임 tarball의 오프라인 설치를 확인합니다. 네이티브 패키지 검증은 파일 해시, 실행 파일 버전, 번들 내부 Linux 공유 라이브러리 탐색을 확인합니다. 플랫폼 CI는 잘못된 모델로 시작을 테스트하며 모델을 다운로드하지 않습니다. 이는 실행 파일 로딩 확인이며 추론 품질 확인이 아닙니다.

실제 GGUF 스모크 테스트에는 별도 가중치가 필요합니다. 판단 보류를 포함한 타입 결과와 점수 구조를 검증하며 작업 정답률이나 GPU 동작을 입증하지 않습니다.
