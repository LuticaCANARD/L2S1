<a id="license"></a>
# 라이선스

[English](../en/LICENSING.md) · [한국어](LICENSING.md) · [日本語](../ja/LICENSING.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

L2S1 소스 코드는 `Cargo.toml`에 명시된 [MIT 라이선스](../../LICENSE)를 따릅니다.

[THIRD_PARTY_LICENSES.txt](../../THIRD_PARTY_LICENSES.txt)에는 의존성의 라이선스 전문과 저작자 표시가 포함되어 있습니다. 해당 구성 요소를 배포할 때 적용되는 고지를 유지하세요.

`l2s1-llama-sys` 소스 패키지는 빌드 중 고정된 llama.cpp 아카이브를 가져옵니다. 아카이브에는 원본 라이선스가 포함되어 있으며, crate에도 [제삼자 고지](../../crates/l2s1-llama-sys/THIRD_PARTY_LICENSES.txt)가 있습니다. 네이티브 의존성을 배포할 때 이 고지를 유지하세요.

저장소 전용 `l2s1-tools` 패키지는 고정된 공개 벤치마크 채점 방식을 따르며, 동결된 소규모 Laya 프로브 렌더링을 포함합니다. [고지](crates/l2s1-tools/NOTICE.md)에는 JevBench MIT 저작자 표시와 출처가 포함됩니다. 전체 데이터셋과 체크포인트는 함께 배포하지 않습니다.

<a id="model-weights"></a>
## 모델 가중치

이 저장소와 Cargo 패키지는 모델 가중치를 포함하지 않습니다. 사용자가 로컬 모델 파일을 준비합니다. `models/`, `*.gguf`, `*.safetensors`는 `.gitignore`와 Cargo 패키지 설정에서 제외됩니다.

프로젝트의 MIT 라이선스는 소스 코드에 적용됩니다. 별도로 구한 모델은 각 모델의 라이선스를 유지합니다.
