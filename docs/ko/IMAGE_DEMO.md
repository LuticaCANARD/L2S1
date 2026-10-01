<a id="이미지-분석과-생각-후-판단-데모"></a>
# 이미지 분석과 생각 후 판단 데모

[English](../en/IMAGE_DEMO.md) · [한국어](IMAGE_DEMO.md) · [日本語](../ja/IMAGE_DEMO.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

웹의 `/demo`에서 사진이나 텍스트 상태를 선택하고 구조화된 판단을 확인할 수 있습니다. 기록 예시는 실제 모델 실행 결과입니다. 페이지를 열거나 기록 버튼을 누를 때 모델을 새로 실행하지 않습니다. 공개 정적 사이트에는 추론 서버가 없으므로, 새 입력의 실시간 분석은 아래 로컬 환경에서 실행합니다.

사진 예시는 재질(choice), 주요 물체가 하나인지(binary), 투명도(ordinal)를 판단합니다. 자유 형식 설명문을 생성하는 데모가 아닙니다. 텍스트 입력에서는 `direct` 또는 `thinking`을 선택할 수 있습니다. `thinking`은 실제 생각 토큰을 생성한 뒤 후보를 평가합니다. 화면에는 생성 토큰 수와 완료 여부를 표시합니다. 현재 Qwen3 dense 텍스트 모델의 단일 토큰 후보 경로를 지원하며 사진 모델의 thinking은 지원하지 않습니다.

<a id="로컬-실행"></a>
## 로컬 실행

이미지를 지원하는 GGUF와 동일 모델의 projector, 빌드된 L2S1 실행 파일이 필요합니다. CUDA 예시는 다음과 같습니다. CPU에서는 `--device cpu`를 선택하세요.

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/mmproj.gguf --device cuda \
  --listen 127.0.0.1:8080
```

텍스트 thinking도 실행하려면 별도 터미널에서 Qwen3 텍스트 서버를 실행합니다.

```sh
./target/release/l2s1 --model /path/to/Qwen3-0.6B-Q8_0.gguf \
  --device cuda --listen 127.0.0.1:8081
```

웹 개발 서버를 실행합니다.

```sh
cd web
npm ci
npm run dev
```

`http://127.0.0.1:5173/demo`를 열고 이미지를 업로드한 뒤 **내 로컬 모델로 분석**을 누릅니다. 상태 JSON과 질문·후보 JSON을 편집할 수 있습니다. 입력이나 정책을 수정하면 이전 결과는 지워집니다. JPEG, PNG, WebP 최대 8 MiB를 지원합니다. 컨텍스트 한도 초과 시 서버가 오류를 반환하며, 사진을 잘라 분석한 것처럼 표시하지 않습니다. 사진 크기·질문 수·상태를 줄이거나 서버 컨텍스트를 늘려 다시 실행하세요.

Vite 개발 서버의 같은 사이트 프록시가 사진의 `/inference`를 `127.0.0.1:8080`으로, 텍스트의 `/text-inference`를 `127.0.0.1:8081`로 연결합니다. 브라우저에서 임의 외부 추론 URL을 입력하는 흐름은 없습니다. 정적 빌드·preview에는 추론 프록시가 없으므로 실제 실행 기록은 볼 수 있지만 실시간 실행에는 별도의 같은 사이트 프록시가 필요합니다.

<a id="수락-기준과-실패-이유"></a>
## 수락 기준과 실패 이유

후보 내 허용 불확실성 비율 `epsilon`을 입력하면 `min_top_probability = 1 - epsilon`으로 변환합니다. UI의 두 숫자는 같은 수락 기준을 표현합니다. **이 설정은 실제 정답 오류율을 보장하지 않습니다.** 후보 상대 확률은 정답 확률이 아니며, 업무의 실제 수락된 오답율은 별도 정답 데이터에서 검증해야 합니다.

`min_candidate_mass`도 독립적으로 지정할 수 있습니다. 후보 간 확률이 높아도 모델이 후보 코드 전체에 거의 확률을 주지 않으면 보류할 수 있습니다. 동률은 보류합니다. 서버는 표준 `abstention_reasons` 코드를 유지하고 사용자 지정 설명을 `reason_messages`에 별도로 반환합니다. `reasoning_limit`, `native_failure`의 사용자 지정 문구도 원래 오류와 함께 표시합니다. 실패 이유 편집은 오류를 숨기거나 성공으로 바꾸지 않습니다.

`option_probability`, `candidate_mass`, 기대 단계값, 보류 이유, 전체 JSON을 출력에 표시합니다. 점수는 정답일 확률·의미 손실률이 아닙니다. 보류한 선택값은 `null`이며 후보 점수는 확인할 수 있습니다.

<a id="요청과-기록"></a>
## 요청과 기록

[examples/image-analysis.json](../../examples/image-analysis.json)은 사진의 세 질문을 담은 기본 요청입니다. 사진 입력에는 `media: [{"id":"photo","type":"image","data_base64":"..."}]`를 추가해 `POST /v1/decisions`로 보냅니다. 각 질문의 `media_ids`는 `photo`를 참조합니다. 텍스트 질문은 `media_ids: []`를 사용합니다.

웹 기록은 `web/static/demo/recorded.json`, `text-direct.json`, `text-thinking.json`에 저장합니다. 요청·응답·모델·실행시각·사진 출처를 함께 보존하고 `sample.jpg`를 표시합니다. 모델 가중치는 웹에 포함하지 않습니다. 사진은 데이터셋 ZIP의 첫 glass 항목 `glass192.jpg`를 그대로 사용했습니다. 출처 라벨은 **glass**인데 실제 Qwen3-VL-2B 실행은 **metal**을 선택하여 오판했습니다. UI에서 이 차이를 눈에 띄게 표시합니다. 수락된 높은 점수도 정답을 보장하지 않습니다. 물체 수·투명도는 이 데이터셋의 정답 라벨이 아니므로 정확성을 주장하지 않습니다. 예시 한 장은 정확도 평가를 대신하지 않습니다. 예시 이미지의 출처와 라이선스는 [이미지 고지](../../web/static/demo/THIRD_PARTY_NOTICE.txt)에 있습니다.

웹의 원래 계약 playground는 합성 점수 예시입니다. `/demo`의 **실제 모델 실행 기록** 및 **방금 실행한 로컬 추론**과 구분해 표시합니다.
