<a id="이미지-분석과-생각-후-판단-데모"></a>
# Image analysis and decisions after thinking

[English](IMAGE_DEMO.md) · [한국어](../ko/IMAGE_DEMO.md) · [日本語](../ja/IMAGE_DEMO.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)

At `/demo`, select a photo or text state and inspect structured decisions. Recorded examples are actual model runs. Opening the page or clicking a recording does not run the model again. The public static site has no inference server; live analysis of new input runs in the local environment below.

The photo example evaluates material (`choice`), whether there is one main object (`binary`), and transparency (`ordinal`). It does not generate free-form descriptions. Text input offers `direct` or `thinking`. `thinking` generates actual thought tokens before scoring candidates. The UI shows generated-token counts and completion. Currently, the single-token candidate path for Qwen3 dense text models is supported; thinking for photo models is unsupported.

<a id="로컬-실행"></a>
## Local execution

You need an image-capable GGUF, the matching model projector, and a built L2S1 executable. The CUDA example follows. For CPU, select `--device cpu`.

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/mmproj.gguf --device cuda \
  --listen 127.0.0.1:8080
```

To also use text thinking, start a Qwen3 text server in another terminal.

```sh
./target/release/l2s1 --model /path/to/Qwen3-0.6B-Q8_0.gguf \
  --device cuda --listen 127.0.0.1:8081
```

Start the web development server.

```sh
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/demo`, upload an image, and click **Analyze with my local model**. You can edit state JSON and question/candidate JSON. Changing input or policy clears previous results. JPEG, PNG, and WebP up to 8 MiB are supported. If the context limit is exceeded, the server returns an error; the UI does not present a cropped photo as a successful analysis. Reduce photo size, questions, or state, or increase server context and retry.

Vite's same-origin development proxy routes photo `/inference` to `127.0.0.1:8080` and text `/text-inference` to `127.0.0.1:8081`. The browser does not offer an arbitrary external inference URL field. Static builds and preview have no inference proxy: recordings are available, but live execution needs a separate same-origin proxy.

<a id="수락-기준과-실패-이유"></a>
## Acceptance criteria and failure reasons

The allowed within-candidate uncertainty ratio `epsilon` becomes `min_top_probability = 1 - epsilon`. The two UI numbers express the same acceptance criterion. **This setting does not guarantee an actual correctness error rate.** Relative candidate probabilities are not probabilities of correctness; validate your real allowed error rate against separately labeled data.

You can independently set `min_candidate_mass`. Even with high relative candidate probability, a decision can abstain if the model assigns almost no probability to the entire candidate-code set. Ties abstain. The server retains standard `abstention_reasons` codes and returns custom explanations separately in `reason_messages`. Custom `reasoning_limit` and `native_failure` messages are displayed alongside the original errors. Editing failure reasons does not hide errors or turn failures into successes.

Output shows `option_probability`, `candidate_mass`, expected level value, abstention reasons, and full JSON. Scores are not correctness probabilities or semantic-loss rates. An abstained selection is `null`; candidate scores remain inspectable.

<a id="요청과-기록"></a>
## Requests and recordings

[examples/image-analysis.json](../../examples/image-analysis.json) is the default three-question photo request. For photos, add `media: [{"id":"photo","type":"image","data_base64":"..."}]` and send it to `POST /v1/decisions`. Each question's `media_ids` references `photo`. Text questions use `media_ids: []`.

Web recordings are stored in `web/static/demo/recorded.json`, `text-direct.json`, and `text-thinking.json`. They preserve requests, responses, models, execution times, and photo sources, and display `sample.jpg`. Model weights are not bundled into the website. The photo is the first glass entry, `glass192.jpg`, from the dataset ZIP, used unchanged. Its source label is **glass**, but the actual Qwen3-VL-2B run selected **metal**, an incorrect prediction. The UI makes this discrepancy prominent. An accepted high score does not guarantee correctness. Object count and transparency are not ground-truth labels in this dataset, so no accuracy claim is made for them. One example photo cannot replace an accuracy evaluation. See the [image notice](../../web/static/demo/THIRD_PARTY_NOTICE.txt) for its source and license.

The original web contract playground uses synthetic scores. It is labeled separately from `/demo`'s **actual model recordings** and **just-run local inference**.
