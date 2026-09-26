<a id="l2s1-for-typescript"></a>
# TypeScript에 대한 L2S1

[English](../../en/typescript/README.md) · [한국어](README.md) · [日本語](../../ja/typescript/README.md)

[English index](../../en/README.md) · [한국어 색인](../README.md) · [日本語索引](../../ja/README.md)

`@l2s1/node`는 Node.js의 기존 Rust 추론 엔진을 사용합니다. 하나의 모델 프로세스가 호출 전반에 걸쳐 상주합니다. 바이너리, 선택, 서열, 이미지, 정책, 추론 및 증거 필드는 Rust HTTP v1 스키마를 사용합니다. 검증, 토큰화, 채점, 판단 보류 및 GPU 실행은 Rust에 유지됩니다.

Node.js 22+가 필요합니다. 래퍼는 현재 OS 및 CPU 아키텍처에 대해 사전 구축된 선택적 런타임 패키지를 선택합니다. 각 런타임 패키지에는 Rust 실행 파일과 일치하는 llama.cpp/GGML 공유 라이브러리가 포함되어 있습니다. 설치 시 컴파일이나 다운로드 스크립트가 실행되지 않습니다. 모델 중량은 별도로 제공됩니다. 이 패키지는 배포용으로 준비되었지만 npm에 게시되지 않았습니다.

Scope 접근, 패키지 순서, 인증, 플랫폼 릴리스 조건은 [npm 공개 검토](PUBLISHING.md)를 참고하세요.

<a id="install-from-this-repository"></a>
## 이 저장소에서 설치

최종 사용자의 경우 빌드 워크플로에서 생성된 래퍼와 플랫폼 런타임 tarball을 설치합니다. 예를 들어 Linux x64에서는 다음과 같습니다.

```sh
npm install ./l2s1-node-0.1.1.tgz ./l2s1-runtime-linux-x64-0.1.1.tgz
```

패키지가 게시된 후 `npm install @l2s1/node`는 선택적 종속성을 통해 런타임을 선택합니다. 선택적 종속성을 활성화된 상태로 유지하세요. 런타임 및 래퍼 버전이 일치해야 합니다.

사용자 정의 네이티브 빌드의 경우 저장소 루트에서 Rust 실행 파일을 빌드합니다.

```sh
cargo build --release --locked --features llama --bin l2s1
```

NVIDIA CUDA의 경우 `--features llama-cuda`를 사용하세요. macOS Metal의 경우 `--features llama-metal`를 사용하세요. 툴체인 요구 사항 및 공유 라이브러리는 [네이티브 빌드 가이드](../GUIDE.md#build)를 참조하세요. 배포 시 실행 파일과 필수 네이티브 라이브러리를 함께 보관하세요.

TypeScript 패키지를 빌드하고 패키징합니다.

```sh
cd sdks/typescript
npm ci
npm run build
npm pack
# In your application:
npm install /path/to/L2S1/sdks/typescript/l2s1-node-0.1.1.tgz
```

<a id="load-a-local-model"></a>
## 로컬 모델 로드

```ts
import { L2S1 } from '@l2s1/node';

const engine = await L2S1.load({
  model: '/path/to/chat-model.gguf',
  device: 'cpu',
});
try {
  const response = await engine.decide({
    state: { temperature_c: 6 },
    decisions: [{
      id: 'cold',
      instruction: 'Is temperature_c below 10?',
      kind: {
        type: 'binary',
        false_label: 'Temperature is at least 10.',
        true_label: 'Temperature is below 10.',
      },
    }],
  });
  for (const result of response.results) {
    console.log(result.id, result.value, result.status);
    if (result.evidence.type === 'model_scored') {
      console.log(result.evidence.estimate.p_true, result.evidence.candidate_mass);
    }
  }
} finally {
  await engine.close();
}
```

`load()`는 설치된 OS/CPU 런타임이나 `binaryPath`를 사용합니다. 기본 `transport: 'stdio'`는
빌드된 Rust 실행 파일과 stdin/stdout JSON으로 통신하며 포트를 열지 않습니다.
같은 프로세스의 N-API 바인딩은 아닙니다. `transport: 'http'`를 명시하면 loopback
서버를 엽니다. 모델 프로세스는 재사용하며 `close()`로 종료하고 기다립니다.
단일 호출의 timeout·취소는 native 추론 중단을 보장하지 않습니다. stdio는 최대
16개 미완료 호출을 허용하고 timeout 후에도 native 응답까지 슬롯을 유지합니다.

빌드 워크플로에는 Linux x64/arm64(glibc), macOS x64/arm64 및 Windows x64가 포함됩니다. Linux 및 Windows 패키지는 CPU를 노출합니다. macOS arm64는 CPU 및 Metal를 노출합니다. CUDA 및 기타 사용자 정의 빌드는 `binaryPath`를 사용합니다. Linux 패키지에는 시스템 glibc/C++ 런타임이 필요합니다. Windows 패키지에는 Microsoft Visual C++ x64 런타임이 필요합니다. 지원되지 않는 플랫폼에서는 명확한 오류가 발생합니다. 이는 워크플로 대상입니다. 한 플랫폼의 로컬 검증에서는 다른 플랫폼 아티팩트가 CI를 통과했는지 확인하지 않습니다.

명시적 리소스 관리를 지원하는 TypeScript 애플리케이션은 `await using engine = await L2S1.load(...)`를 작성할 수 있습니다. 이는 범위 종료 시 `close()`를 호출합니다. 패키지는 ESM입니다.

`LoadOptions`는 CPU/CUDA/Metal, 컨텍스트/배치/스레드 수, 비전 프로젝터(`mmproj`), LoRA, 실행 모드, 병렬 너비, 프롬프트 레이아웃/세부 정보를 노출합니다. 그리고 시작 정책. 덜 일반적인 Rust 플래그는 `extraArgs`에서 전달될 수 있습니다. `--stdio` / `--listen`는 예약되어 있습니다. `startupTimeoutMs`의 기본값은 120,000이고 `timeoutMs`는 180,000입니다. 시작 `signal`가 로드를 취소합니다. `onStderr`는 네이티브 로그 청크를 수신합니다. 호출에서는 `{ signal, timeoutMs }`를 두 번째 인수로 허용합니다. 실패한 추론 요청은 자동으로 재시도되지 않습니다.

빌드 후 Node.js 24의 TypeScript 지원을 사용하여 [warehouse example](../../../sdks/typescript/examples/warehouse.ts)를 실행합니다.

```sh
L2S1_BINARY=/absolute/path/to/l2s1 node examples/warehouse.ts /path/to/chat-model.gguf
```

<a id="connect-to-a-server"></a>
## 서버에 연결

애플리케이션 코드는 백엔드를 전환하는 동안 하나의 API를 유지할 수 있습니다.

```ts
import { L2S1, type DecisionBackend } from '@l2s1/node';

const local = await L2S1.load({ model: '/path/to/model.gguf' });
const remote = L2S1.connect({ baseUrl: 'https://inference.example.com/l2s1' });
// Both expose decide(request), capabilities(), close() and async disposal.
// const response = await remote.decide(request);

function useCustomBackend(backend: DecisionBackend) {
  return L2S1.fromBackend(backend);
}
```

`DecisionBackend`에는 내보낸 응답 유형을 반환하는 비동기 `decide(request, options)` 및 `capabilities(options)` 메서드가 필요합니다. 선택적 `close()` 후크는 소유한 리소스를 해제합니다. 이는 애플리케이션 판단 코드를 변경하지 않고도 사용자 정의 IPC, RPC 또는 기타 런타임 어댑터를 허용합니다. `fromBackend()`는 라이프사이클 소유권을 Facade로 이전합니다. HTTP 연결을 닫으면 이 클라이언트의 요청이 취소되고 공유 원격 서버는 계속 실행됩니다.

<a id="repeat-fixed-decisions-with-new-state"></a>
## 데이터만 바꿔 고정 질문 반복 호출

```ts
const batchEngine = await L2S1.load({
  model: '/path/to/model.gguf', executionMode: 'parallel', parallelWidth: 4,
});
type Temperature = { temperature_c: number };
const plan = batchEngine.prepare<Temperature>([{
  id: 'cold', instruction: 'Is temperature_c below 10?',
  kind: { type: 'binary', false_label: 'At least 10.', true_label: 'Below 10.' },
}]);
const first = await plan.decide({ temperature_c: 6 });
const second = await plan.decide({ temperature_c: 15 });
const responses = await plan.decideBatch([{ temperature_c: 2 }, { temperature_c: 20 }]);
// Different definitions per item: await batchEngine.decideBatch(requests).
await batchEngine.close();
```

`prepare()`는 고정 질문과 state 타입을 재사용합니다. 토큰 컴파일이나 영속 KV 재사용은
아닙니다. `decideBatch()`는 stdio 또는 HTTP `/v1/decision-batches`로 배열 전체를 한 번
전달하고 native parallel을 실행합니다. `executionMode: 'parallel'`과 `parallelWidth`를
지정하세요. state·ID·media·정책은 독립적이며 결과는 입력 순서입니다. timeout은 배치
전체에 적용합니다. 직렬 fallback·자동 재시도 없이 `batch_unsupported` 또는
`batch_not_enabled`를 명시합니다. 사용자 backend는 선택적 `decideBatch()`를 구현합니다.

최대 128개 요청·총 128개 판단, direct reasoning·판단별 최대 26개 선택지를 지원합니다.
모두 text이거나 모든 판단에 이미지가 하나씩 있는 배치와 일치하는 projector를 사용합니다.
text/image 혼합은 거부합니다. 실행 전에 모든 wire 입력을 검증하며 실행 오류는 전체
배치의 실패입니다. 완료된 wave는 되돌리거나 재실행하지 않습니다. `capabilities().batch`를
확인하세요. `Promise.all(decide(...))`는 자동 배치가 아닙니다.
[배치 API 검토](../BATCHING_API_REVIEW.md)와 [Python SDK](../python/README.md)를 참고하세요.



기존 Rust 서버에는 `@l2s1/node/http`를 사용합니다. 이 하위 경로에는 Node 내장 가져오기가 없으며 브라우저용으로 번들로 제공될 수도 있습니다. 브라우저 호출에는 CORS를 제공하는 동일 출처 프록시 또는 역방향 프록시가 필요합니다. Rust 서버는 CORS 헤더를 추가하지 않습니다. 이는 브라우저 내부에서 네이티브 추론을 실행하지 않습니다.

```ts
import { L2S1Client, L2S1Error } from '@l2s1/node/http';

const client = new L2S1Client({
  baseUrl: 'http://127.0.0.1:8080',
  // headers: { Authorization: 'Bearer proxy-token' },
});
await client.health();
const capabilities = await client.capabilities();
// await client.decide(request, { signal: abortController.signal });
```

`L2S1Error`는 Rust 실패로부터 `code`, `status`, `requestId` 및 `userReason`를 유지합니다. 전송 취소 및 네트워크 오류는 원래 가져오기 오류를 유지합니다. 클라이언트는 버전이 지정된 응답 봉투, 판단 주문/ID/종류, 결과 상태 및 증거 종류를 확인합니다. 모델 점수와 제공자 선택에는 뚜렷한 TypeScript 증거 유형이 있습니다. `selection_only`에는 확률이 없습니다. 클라이언트는 Rust wgpu 및 OpenRouter HTTP 서버도 지원합니다.

<a id="policies-reasoning-and-images"></a>
## 정책, 추론 및 이미지

`policy` 요청은 `{ min_top_probability, min_candidate_mass }`를 사용합니다. 둘 다 공급되어야 합니다. `target_error_rate`는 `min_top_probability = 1 - rate`에 매핑됩니다. 이는 정확성을 보장하지 않습니다. `failure_reasons`는 알려진 판단 보류/오류 코드에 대한 메시지를 제공합니다. `reasoning: { mode: 'thinking', max_tokens: 128 }`에는 백엔드 및 모델 광고 지원이 필요합니다. 지원되지 않는 요청은 Rust에서 실패합니다. 옵션 기능을 선택하기 전에 `capabilities()`를 읽어보세요.

동봉된 v1 엔진은 요청별 정책, 오류 예산, 실패 문구를 지원합니다. reasoning은 direct만 지원하며 thinking 요청은 명시적으로 거절합니다. 오래된 서버는 지원하지 않는 필드에 HTTP 400을 반환합니다. `L2S1.load({ policy })`로 시작 시 기본 정책도 설정할 수 있습니다. 클라이언트는 요청한 제어를 조용히 무시하지 않습니다.

이미지는 데이터 URL 접두사 없이 표준 base64를 사용합니다.

```ts
const request = {
  state: { task: 'classify the image' },
  media: [{ type: 'image' as const, id: 'photo', data_base64: imageBytes.toString('base64') }],
  decisions: [{
    id: 'cat', instruction: 'Is there a cat in the image?', media_ids: ['photo'],
    kind: { type: 'binary' as const, false_label: 'No cat.', true_label: 'A cat is present.' },
  }],
};
// Load with { model: visionModel, mmproj: matchingProjector } before deciding.
```

생략됨 `media_ids`는 모든 요청 미디어를 사용합니다. `[]`는 텍스트만 선택합니다. Rust는 미디어, 판단, 본문 및 모델 제한을 적용합니다. [HTTP 계약](../GUIDE.md#direct-image-input-and-http-api)를 참조하세요.

<a id="verification-and-portability"></a>
## 검증 및 이식성

```sh
npm run check
npm test
npm run test:rust
npm pack --dry-run
```

`test:rust`는 작은 결정론적 Rust 백엔드를 구축하고 관리형 노드 호출을 통해 실제 Rust 채점, 검증 및 HTTP 엔벨로프를 실행합니다. Rust 툴체인이 필요하지만 모델이나 C++ 툴체인은 필요하지 않습니다. 모델 품질 증거가 아닌 픽스처 증거입니다.

선택적 실제 모델 스모크(저장소 루트에서 먼저 네이티브 바이너리를 빌드):

```sh
cd sdks/typescript
L2S1_BINARY=/absolute/path/to/l2s1 L2S1_MODEL=/path/to/chat-model.gguf node --test test/model.integration.mjs
```

WASM은 별도의 런타임 포트입니다. WASM용 Rust 래퍼를 컴파일해도 C++ 추론 엔진이 패키지되지 않거나 CUDA/Metal 실행이 유지되지 않습니다. WASM 배포판에는 별도로 구축된 추론 엔진, JS/WASM 경계 및 CPU/WebGPU 경로가 필요합니다. 이 패키지는 대신 사전 구축된 네이티브 런타임 패키지를 사용합니다.

<a id="build-distribution-artifacts"></a>
## 배포 아티팩트 빌드

워크플로는 `v*` 버전 태그 푸시 또는 수동 실행으로 시작합니다. [runtime 워크플로](../../../.github/workflows/typescript-runtimes.yml)는 5개 플랫폼을 모두 빌드하고 래퍼/런타임 `.tgz` 파일을 워크플로 아티팩트로 업로드합니다. 패키지를 게시하지 않습니다. 래퍼를 게시하기 전에 일치하는 버전으로 모든 런타임 패키지를 게시하세요. 워크플로는 체크섬, 실행 가능한 시작, 새로운 npm 프로젝트에 대한 설치 및 자동 런타임 해결을 확인합니다. CI의 잘못된 모델 시작은 모델 추론 검증이 아닙니다.

현재 플랫폼을 로컬로 빌드하려면 저장소 루트에서 실행하십시오.

```sh
# On macOS arm64, use llama-metal and --devices cpu,metal to include Metal.
L2S1_PORTABLE_BUILD=1 cargo build --release --locked --features llama \
  --bin l2s1 --message-format=json-render-diagnostics > native-build.jsonl
node sdks/typescript/scripts/bundle-runtime.mjs --cargo-log native-build.jsonl
node sdks/typescript/scripts/verify-runtime.mjs sdks/typescript/runtime-packages/linux-x64
cd sdks/typescript/runtime-packages/linux-x64
npm pack --pack-destination ../../ --ignore-scripts
cd ../..
npm pack
npm run test:package -- l2s1-node-0.1.1.tgz l2s1-runtime-linux-x64-0.1.1.tgz
```

런타임을 다시 빌드할 때 새로운 출력 디렉터리를 사용하세요. `L2S1_PORTABLE_BUILD=1`는 빌드 호스트 CPU 지침 및 OpenMP 종속성을 비활성화합니다. 일반 CPU 커널은 호스트 최적화 사용자 정의 빌드보다 느릴 수 있습니다. 각 아티팩트에는 라이선스 알림과 SHA-256 매니페스트가 포함되어 있습니다. 검증자는 Linux llama.cpp/GGML 종속성이 번들 디렉터리에서 해결되는지 확인합니다.

[배포 파이프라인](../RELEASE_PIPELINE.md)에서 SDK와 native runtime의 자동 게시·인증 설정을 확인하세요.
