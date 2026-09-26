<a id="l2s1--llm-to-system-1-website"></a>
# L2S1 — LLM to System 1 웹사이트

[English](../../en/web/README.md) · [한국어](README.md) · [日本語](../../ja/web/README.md)

[English index](../../en/README.md) · [한국어 색인](../README.md) · [日本語索引](../../ja/README.md)

영어 SvelteKit 소개 페이지는 `adapter-static`으로 사전 렌더링합니다. 계약 설명 예시는 합성 확률임을 표시하며 성능 표는 기록된 모델 테스트 요약을 담습니다. `/webgpu`는 영어·한국어·일본어로 Qwen3 0.6B 브라우저 실제 추론과 명시적인 큰 GGUF 로컬 서버 선택을 제공합니다. 브라우저 모드 입력은 worker에 남고 로컬 모드 입력은 분석할 때만 전송됩니다. 모델은 사용자가 요청할 때 내려받으며 사이트에 번들하지 않습니다. [실행 안내](../WEBGPU_DEMO.md)를 참고하세요.

```sh
npm ci
npm run check
npm run lint
npm run build
npm run dev
```

생성된 `build/` 디렉터리를 정적 호스트에 배포합니다. 이 설정에서는 배포가 수행되지 않았습니다.

브라우저 확인의 경우:

```sh
npx playwright install chromium
npm run build
npm test
```

`PLAYWRIGHT_CHROMIUM_EXECUTABLE`는 이미 설치된 호환 Chromium을 선택할 수 있습니다. 브라우저 테스트에서는 JavaScript 없는 사전 렌더링, 키보드 제어 판단 보류, 모든 판단 유형, 클립보드 피드백, 문서 다운로드 및 데스크톱/모바일 오버플로를 다룹니다. 스크린샷은 `test-results/` 아래에 저장됩니다.

`scripts/sync-content.mjs`는 루트의 영어·한국어·일본어 README, 라이선스, `docs/` 최상위 Markdown, 전체 `docs/en/`, `docs/ko/`, `docs/ja/` 언어 디렉토리, 공개 패키지·웹사이트 README, 벤치마크 README·보고서, 창고 JSON 예제를 `static/docs/`에 복사합니다. 저장소 상대 문서 링크는 경로를 유지합니다. 각 언어 디렉토리는 색인과 전체 공개 문서 본문을 포함하며, 같은 문서의 언어 전환 링크와 원문 제목의 앵커를 유지합니다. 모델 디렉토리나 원래 모델 출력은 복사하지 않습니다. 명시된 파일 다섯 개만 공개하도록 한 목록이 typed-decisions 요약, 채점 기록 4,000개, 이동 가능한 manifest를 `/benchmarks/typed-decisions-20260926/`에 공개합니다. 요약과 채점 JSONL은 저장소 바이트를 보존합니다. Manifest는 두 모델 실행 기록과 절대 경로 정규화 설명을 포함합니다. 기록에는 원래 상태·생성된 사고 텍스트가 없습니다. `src/lib/benchmarks.json`은 실제 측정 중 일부를 경로 없이 요약한 것이며 새로 완료한 성능 실행에서만 갱신하세요.

사이트 소스는 저장소의 MIT 라이센스를 따릅니다. 프레임워크 및 종속성 알림은 `THIRD_PARTY_LICENSES.txt`에 별도로 기록되고 정적 빌드에 복사됩니다. 빌드 도구와 테스트는 제공된 클라이언트 런타임의 일부가 아닙니다.
