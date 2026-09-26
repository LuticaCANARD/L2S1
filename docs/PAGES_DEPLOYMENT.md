# Cloudflare Pages deployment

[English](en/PAGES_DEPLOYMENT.md) · [한국어](ko/PAGES_DEPLOYMENT.md) · [日本語](ja/PAGES_DEPLOYMENT.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

The SvelteKit web app uses `adapter-static` and publishes `web/build` to the
Cloudflare Pages project `n2s1`. The requested production hostname is
`https://n2s1.luticalab.net`.

The public demo includes recorded image and text decisions. These are stored
results from the native L2S1 HTTP server, not model inference inside Pages.
Live inference requires a separately running native L2S1 server behind a
same-origin proxy: `/inference` for vision and `/text-inference` for text. The
Vite development server supplies those proxies to localhost ports 8080 and
8081. Static Pages does not provide the Vite proxy; a production Pages Function
or equivalent proxy must be connected to a reachable HTTPS native backend
before live mode can work. The current UI does not accept arbitrary remote API
URLs. Do not describe a Pages asset deployment as deployment of the GGUF
runtime.

The `/webgpu` page runs its text model in the user's browser. It needs no native
inference proxy: the user explicitly downloads the pinned ONNX model and runtime,
then their Web Worker executes WebGPU inference. This does not add image inference
to Pages or deploy a native GGUF server. See [WebGPU execution](WEBGPU_DEMO.md).

## Account and authentication

On 2026-09-26, Cloudflare's API confirmed that `luticalab.net` is an active zone
in account `cb2875b9abef08939d6a42e711a3600d`. Its zone ID is
`f3f8abbf6f629072dc472e9afef8309a`. These IDs are public configuration identifiers,
not credentials.

Use an API token with **Account → Cloudflare Pages → Edit**, scoped to that
account, and **Zone → DNS → Edit**, scoped to `luticalab.net`. Zone Read access
is also useful for checking the zone before changes. Keep credentials in the
environment or the Wrangler credential store; never add them to this file or
`wrangler.jsonc`.

On 2026-09-26, the user completed the official `cf` OAuth login with Pages
read/write and DNS read/edit permissions. Both Pages and the requested DNS
record were successfully queried after login. The earlier Wrangler login lacked
Pages/DNS permissions; its successful `whoami` did not prove deployment access.
The official `cf` CLI supports a shared Pages/DNS OAuth login with these scopes:

```bash
cf auth login --force --no-browser --scopes account:read user:read zone:read pages:read pages:write dns_records:read dns_records:edit
cf auth whoami
CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d cf pages projects list
cf --zone f3f8abbf6f629072dc472e9afef8309a dns records list --name n2s1.luticalab.net
```

Open the printed OAuth URL in a browser that can reach this machine's localhost
callback at port 8877. The `cf` and Wrangler CLI credential stores are separate;
logging into one does not automatically replace the other's OAuth login.

## Build and upload

Use Wrangler 4.141.0 or a reviewed newer version. From the repository root:

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

The `n2s1` project already exists. To recreate a new equivalent Pages project
when it does not exist, create it once:

```bash
npx wrangler@4.141.0 pages project create n2s1 --production-branch main --force
```

Wrangler 4.141.0 delegates new project creation to Workers by default. The
creation-only `--force` option keeps the explicitly requested Pages deployment.
Do not pass `--force` on subsequent commands for an existing Pages project.

Upload the verified build to the production branch:

```bash
npx wrangler@4.141.0 pages deploy ./build --project-name n2s1 --branch main --commit-dirty=true
npx wrangler@4.141.0 pages deployment list --project-name n2s1
```

`--commit-dirty=true` records that a local uncommitted build was uploaded. After
the changes are committed, omit it or set it according to the actual Git state.
The project may receive a suffixed `pages.dev` hostname if `n2s1.pages.dev` is
unavailable; use the hostname returned by Cloudflare rather than assuming it.

This workflow uses Direct Upload. Cloudflare does not support converting a
Direct Upload project to Git integration later; automatic uploads can still
be implemented by running Wrangler in CI.

## Associate the production hostname

On 2026-09-26, project `n2s1` was created with production branch `main` and the
confirmed hostname `n2s1.pages.dev`. Its project ID is
`4f6ca94b-0c26-47e7-993b-018c828ff31a`. The custom domain
`n2s1.luticalab.net` was associated, then a proxied CNAME pointing to
`n2s1.pages.dev` was created. The DNS record ID is
`d4d260b26ad1ee0eeb0fd27d44d7a196`.

The first production deployment succeeded on 2026-09-26 at 09:50:57 UTC:
`6350d743-8bc2-4d9d-b211-6a00553999d0`, available at
`https://6350d743.n2s1.pages.dev`. Cloudflare reports the custom domain,
ownership verification, and certificate validation as `active`.
HTTPS requests returned 200 for the home page, `/demo`, image and text recorded
results, and both typed-decisions summary artifacts. Their response bodies were
byte-for-byte equal to the verified local build. This verifies publication of
recorded results and assets; no production inference proxy was deployed.
A Chromium browser opened the deployed `/demo`, displayed all three image
decision cards and the material error disclosure, and switched to the text
thinking recording with 88 generated reasoning tokens. No browser page errors
were observed in that check.

The WebGPU production deployment also succeeded on 2026-09-26:
`6c8056b9-b301-4f24-952b-372d774f9e96`, available at
`https://6c8056b9.n2s1.pages.dev`. The custom-domain `/webgpu`, worker JavaScript,
and its runtime chunk returned HTTPS 200 with bodies equal to the local build.
Chromium verified the load controls and adapter-unavailable explanation, with no
page errors and no Hugging Face or jsDelivr requests before a download click.
That publication check is separate from actual model execution and does not
establish hardware GPU performance.

First add `n2s1.luticalab.net` to the Pages project's **Custom domains**. Only
after the Pages association exists, create or verify a DNS CNAME record for
`n2s1.luticalab.net` pointing to the actual production `pages.dev` hostname.
For a zone already hosted on Cloudflare, the dashboard can create this CNAME
as part of the custom domain flow. Check the existing record first and avoid
overwriting a record that points to another service without reviewing it.

For an automated domain association, Cloudflare's API accepts
`POST /accounts/{account_id}/pages/projects/{project_name}/domains` with
`{"name":"n2s1.luticalab.net"}`. DNS records are managed separately through
the zone DNS API. Wrangler currently has no Pages custom-domain subcommand.

After authenticating the `cf` CLI, its domain command is also available:

```bash
CLOUDFLARE_ACCOUNT_ID=cb2875b9abef08939d6a42e711a3600d cf pages projects domains create n2s1 --name n2s1.luticalab.net
```

Adding only a DNS CNAME, without the Pages custom-domain association, is
insufficient and can produce a 522 error.

## Verify after upload

1. Confirm the production deployment is successful in Pages deployment listing.
2. Confirm the Pages custom domain and its certificate are active.
3. Resolve `n2s1.luticalab.net` and request its HTTPS URL.
4. Open the production URL in a browser, run the recorded image demo, switch the
   decision mode, and check the benchmark data and downloadable artifacts.
5. Treat live native inference as separately verified only after the public API
   has been connected and an image request succeeds from the deployed browser.

## Official references

- [Pages Wrangler configuration](https://developers.cloudflare.com/pages/functions/wrangler-configuration/)
- [Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/)
- [Custom domains](https://developers.cloudflare.com/pages/configuration/custom-domains/)
- [Pages add-domain API](https://developers.cloudflare.com/api/resources/pages/subresources/projects/subresources/domains/methods/create/)
