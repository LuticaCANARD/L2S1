<a id="l2s1-llamacpp-native-dependency"></a>
# L2S1 llama.cpp 네이티브 종속성

[English](../../../en/crates/l2s1-llama-sys/README.md) · [한국어](README.md) · [日本語](../../../ja/crates/l2s1-llama-sys/README.md)

[English index](../../../en/README.md) · [한국어 색인](../../README.md) · [日本語索引](../../../ja/README.md)

이 상자는 동일한 소스 개정판에서 llama.cpp 및 L2S1의 C++ 브리지를 빌드합니다. CMake FetchContent는 ggml-org/llama.cpp 커밋 `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`를 다운로드하고 빌드하기 전에 소스 아카이브의 SHA-256를 확인합니다. `cmake/CMakeLists.txt` 및 `UPSTREAM_COMMIT`를 참조하세요. 네이티브 소스는 이 저장소에 체크인되지 않았습니다. 업스트림 소스 아카이브에는 라이선스가 포함되어 있습니다. 이 상자에는 `THIRD_PARTY_LICENSES.txt`가 포함되어 있습니다.

기본 빌드는 CPU 전용입니다. macOS에서 CUDA에 대해 `l2s1/llama-cuda`를 활성화하거나 macOS에서 Metal에 대해 `l2s1/llama-metal`를 활성화합니다. CMake 3.24+ 및 C++17 컴파일러가 필요합니다. CUDA에는 CUDA 툴킷이 추가로 필요합니다. CUDA 및 Metal는 하나의 빌드에서 함께 활성화할 수 없습니다. macOS 및 Linux에서 `l2s1` 빌드 스크립트는 llama.cpp 라이브러리 디렉터리를 실행 가능한 rpath로 포함합니다. 설치된 네이티브 라이브러리는 종속 llama.cpp 및 GGML 라이브러리에 대한 자체 디렉터리도 검색합니다. 다운스트림 실행 파일은 빌드 스크립트에서 `DEP_L2S1_LIBDIR`를 읽고 자체 rpath를 추가할 수 있습니다. 네이티브는 기본값을 경고로 기록합니다. 레벨을 변경하려면 프로세스 시작 전에 `L2S1_LOG=error|warn|info|debug|off`를 설정하십시오. 모델 로드 실패에는 마지막 llama.cpp 오류 메시지가 포함됩니다. CUDA 빌드는 배포 가능한 아티팩트에 대한 호스트별 아키텍처 선택을 비활성화합니다. 알려진 대상에 대해 빌드할 때 CUDA 아키텍처를 제한하려면 `L2S1_CUDA_ARCHITECTURES`(예: `86`)를 설정하세요. `L2S1_NATIVE_COMPILER_LAUNCHER`를 `ccache`에 대한 전체 경로와 같은 컴파일러 시작 프로그램으로 설정하여 llama.cpp의 C, C++ 및 CUDA 컴파일을 래핑합니다. `scripts/build_cuda_arch.sh` 진입점은 설치 시 `ccache`를 자동으로 감지합니다. Rust 캐싱을 시도하려면 `RUSTC_WRAPPER=sccache`를 명시적으로 설정하세요. `cc` 크레이트는 해당 Rust 래퍼가 선택된 경우 L2S1 C++ 브리지에 대해 `sccache`도 사용합니다.

첫 번째 기본 빌드에서는 고정된 소스를 다운로드하기 위해 네트워크 액세스가 필요합니다. 오프라인 빌드 또는 다른 llama.cpp 소스 체크아웃의 경우 `L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp`를 설정합니다. 새 변수가 설정 해제되면 레거시 `LLAMA_CPP_DIR`가 허용됩니다. 선택한 체크아웃에는 `include/llama.h`, `src/llama-ext.h`, `tools/mtmd/mtmd.h`, `common/jinja` 및 해당 종속성이 포함되어야 합니다. 이 크레이트는 해당 소스에서 자체 네이티브 라이브러리를 빌드합니다. 별도로 빌드된 라이브러리는 헤더와 절대 혼합되지 않습니다. 대체 업스트림 개정판은 호환성이 보장되지 않으며 사용하기 전에 네이티브 계약 테스트를 통과해야 합니다.

빌드에는 직접 스틸 이미지 입력을 위한 `libmtmd`가 포함되어 있으며 이를 동일한 `libllama` 개정판과 연결합니다. 비디오 지원이 비활성화되었습니다. 소스 및 바이너리 패키지에는 일치하는 `libmtmd` 및 GGML 공유 라이브러리가 포함되어야 합니다.

버전이 지정된 릴리스에 따라 `l2s1` 패키지보다 먼저 이 네이티브 패키지를 게시합니다. 사전 빌드된 실행 파일에는 적절한 로더 경로와 함께 패키지된 네이티브 공유 라이브러리도 필요합니다. 소스 패키지 빌드는 재배치 가능한 바이너리 아카이브를 자체적으로 생성하지 않습니다.
