<a id="cloudflare-pages-deployment"></a>
# Cloudflare 페이지 배포

[English](../en/PAGES_DEPLOYMENT.md) · [한국어](PAGES_DEPLOYMENT.md) · [日本語](../ja/PAGES_DEPLOYMENT.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

SvelteKit 웹 앱은 `adapter-static`를 사용하고 `web/build`를 Cloudflare Pages 프로젝트 `l2s1`에 게시합니다. 요청된 운영 환경 호스트 이름은 `https://l2s1.luticalab.net`입니다.

공개 데모에는 기록된 이미지와 텍스트 판단이 포함됩니다. 이는 페이지 내부의 모델 추론이 아닌 네이티브 L2S1 HTTP 서버의 저장된 결과입니다. 실시간 추론을 위해서는 동일한 원본 프록시(비전의 경우 `/inference`, 텍스트의 경우 `/text-inference`) 뒤에서 별도로 실행되는 네이티브 L2S1 서버가 필요합니다. Vite 개발 서버는 해당 프록시를 로컬 호스트 포트 8080에 제공하고
8081. 정적 페이지는 Vite 프록시를 제공하지 않습니다. 라이브 모드가 작동하려면 먼저 운영 환경 페이지 함수 또는 동등한 프록시가 도달 가능한 HTTPS 네이티브 백엔드에 연결되어야 합니다. 현재 UI는 임의의 원격 API URL을 허용하지 않습니다. 페이지 자산 배포를 GGUF 런타임 배포로 설명하지 마세요.

`/webgpu` 페이지는 사용자 브라우저에서 텍스트 모델을 실행합니다. 네이티브 추론 프록시는 필요하지 않습니다. 사용자가 고정 ONNX 모델과 런타임을 명시적으로 다운로드하면 Web Worker가 WebGPU 추론을 실행합니다. 이는 Pages에 이미지 추론을 추가하거나 네이티브 GGUF 서버를 배포하는 것이 아닙니다. [WebGPU 실행](WEBGPU_DEMO.md)을 참고하세요.

<a id="account-and-authentication"></a>
## 계정 및 인증

2026-09-26에서 Cloudflare의 API는 `luticalab.net`가 `cb2875b9abef08939d6a42e711a3600d` 계정의 활성 영역임을 확인했습니다. 해당 영역 ID는 `f3f8abbf6f629072dc472e9afef8309a`입니다. 이러한 ID는 자격 증명이 아닌 공개 구성 식별자입니다.

해당 계정으로 범위가 지정된 **Account → Cloudflare Pages → Edit** 및 `luticalab.net`로 범위가 지정된 **Zone → DNS → Edit**와 함께 API 토큰을 사용하세요. 영역 읽기 액세스는 변경 전에 영역을 확인하는 데에도 유용합니다. 환경이나 Wrangler 자격 증명 저장소에 자격 증명을 보관하세요. 이 파일이나 `wrangler.jsonc`에 추가하지 마십시오.

2026-09-26에서 사용자는 페이지 읽기/쓰기 및 DNS 읽기/편집 권한을 사용하여 공식 `cf` OAuth 로그인을 완료했습니다. 로그인 후 페이지와 요청된 DNS 레코드가 모두 성공적으로 쿼리되었습니다. 이전 Wrangler 로그인에는 페이지/DNS 권한이 부족했습니다. 성공적인 `whoami`는 배포 액세스를 입증하지 못했습니다. 공식 `cf` CLI는 다음 범위의 공유 페이지/DNS OAuth 로그인을 지원합니다.

```bash
cf auth login --force --no-browser --scopes account:read user:read zone:read pages:read pages:write dns_records:read dns_records:edit
cf auth whoami
CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d cf pages projects list
cf --zone f3f8abbf6f629072dc472e9afef8309a dns records list --name l2s1.luticalab.net
```

포트 8877에서 이 시스템의 로컬 호스트 콜백에 연결할 수 있는 브라우저에서 인쇄된 OAuth URL을 엽니다. `cf` 및 Wrangler CLI 자격 증명 저장소는 별개입니다. 한 곳에 로그인해도 다른 OAuth 로그인이 자동으로 대체되지는 않습니다.

<a id="build-and-upload"></a>
## 빌드 및 업로드

Wrangler 4.141.0 또는 검토된 최신 버전을 사용하세요. 저장소 루트에서:

```bash
npm --prefix web ci
npm --prefix web run check
npm --prefix web run lint
npm --prefix web run build
npm --prefix web test
cd web
export CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d
npx wrangler@4.141.0 whoami
npx wrangler@4.141.0 pages project list --json
```

`l2s1` 프로젝트가 이미 존재합니다. 존재하지 않는 새로운 동등한 페이지 프로젝트를 다시 생성하려면 다음과 같이 한 번 생성하세요.

```bash
npx wrangler@4.141.0 pages project create l2s1 --production-branch main --force
```

Wrangler 4.141.0는 기본적으로 새 프로젝트 생성을 작업자에게 위임합니다. 생성 전용 `--force` 옵션은 명시적으로 요청된 페이지 배포를 유지합니다. 기존 페이지 프로젝트의 후속 명령에 `--force`를 전달하지 마십시오.

확인된 빌드를 운영 환경 분기에 업로드합니다.

```bash
npx wrangler@4.141.0 pages deploy ./build --project-name l2s1 --branch main --commit-dirty=true
npx wrangler@4.141.0 pages deployment list --project-name l2s1
```

`--commit-dirty=true`는 커밋되지 않은 로컬 빌드가 업로드되었음을 기록합니다. 변경 사항이 커밋된 후에는 생략하거나 실제 Git 상태에 따라 설정하세요. `l2s1.pages.dev`를 사용할 수 없는 경우 프로젝트는 접미사 `pages.dev` 호스트 이름을 받을 수 있습니다. 가정하는 대신 Cloudflare에서 반환한 호스트 이름을 사용하세요.

이 워크플로는 직접 업로드를 사용합니다. Cloudflare는 나중에 직접 업로드 프로젝트를 Git 통합으로 변환하는 것을 지원하지 않습니다. CI에서 Wrangler를 실행하면 자동 업로드를 계속 구현할 수 있습니다.

<a id="associate-the-production-hostname"></a>
## 운영 환경 호스트 이름 연결

2026-09-26에 운영 사이트를 프로젝트 `l2s1`과 주소 `https://l2s1.luticalab.net`으로 정정했습니다. 프로젝트 ID는 `d15c7797-4208-4388-9a7c-44a3df22e496`입니다. 프록시 CNAME은 `l2s1.pages.dev`를 가리키며 DNS 레코드 ID는 `58376692674167fd35459ab8d5519f5a`입니다.

이전 `n2s1` 프로젝트는 기존 링크를 올바른 주소로 리디렉션하는 용도로만 유지합니다. 해당 프로젝트에는 `web/legacy/`를 별도로 업로드하며, 본 사이트의 빌드를 업로드하지 않습니다. 경로와 쿼리 문자열을 유지합니다.

```bash
npx wrangler@4.141.0 pages deploy ./legacy --project-name n2s1 --branch main
```

먼저 Pages 프로젝트의 **Custom domains**에 `l2s1.luticalab.net`를 추가합니다. 페이지 연결이 존재한 후에만 실제 운영 환경 `pages.dev` 호스트 이름을 가리키는 `l2s1.luticalab.net`에 대한 DNS CNAME 레코드를 생성하거나 확인하세요. Cloudflare에 이미 호스팅된 영역의 경우 대시보드는 사용자 지정 도메인 흐름의 일부로 이 CNAME을 생성할 수 있습니다. 기존 레코드를 먼저 확인하고, 검토하지 않고 다른 서비스를 가리키는 레코드를 덮어쓰지 않도록 하세요.

자동화된 도메인 연결의 경우 Cloudflare의 API는 `{"name":"l2s1.luticalab.net"}`와 함께 `POST /accounts/{account_id}/pages/projects/{project_name}/domains`를 허용합니다. DNS 레코드는 영역 DNS API를 통해 별도로 관리됩니다. Wrangler에는 현재 Pages 사용자 정의 도메인 하위 명령이 없습니다.

`cf` CLI를 인증한 후 해당 도메인 명령도 사용할 수 있습니다.

```bash
CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d cf pages projects domains create l2s1 --name l2s1.luticalab.net
```

페이지 사용자 정의 도메인 연결 없이 DNS CNAME만 추가하는 것만으로는 충분하지 않으며 522 오류가 발생할 수 있습니다.

<a id="verify-after-upload"></a>
## 업로드 후 확인

1. 페이지 배포 목록에서 운영 환경 배포가 성공했는지 확인합니다.
2. 페이지 사용자 정의 도메인과 해당 인증서가 활성화되어 있는지 확인하세요.
3. `l2s1.luticalab.net`를 확인하고 해당 HTTPS URL을 요청합니다.
4. 브라우저에서 운영 환경 URL을 열고, 기록된 이미지 데모를 실행하고, 판단 모드를 전환하고, 벤치마크 데이터와 다운로드 가능한 아티팩트를 확인하세요.
5. 공개 API가 연결되고 배포된 브라우저에서 이미지 요청이 성공한 후에만 라이브 네이티브 추론을 별도로 검증된 것으로 처리합니다.

<a id="official-references"></a>
## 공식 참고자료

- [Pages 랭글러 구성](https://developers.cloudflare.com/pages/functions/wrangler-configuration/)
- [직접 업로드](https://developers.cloudflare.com/pages/get-started/direct-upload/)
- [사용자 정의 도메인](https://developers.cloudflare.com/pages/configuration/custom-domains/)
- [Pages 도메인 추가 API](https://developers.cloudflare.com/api/resources/pages/subresources/projects/subresources/domains/methods/create/)
