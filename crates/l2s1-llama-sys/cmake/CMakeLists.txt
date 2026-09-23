cmake_minimum_required(VERSION 3.24)
project(l2s1_llama_source LANGUAGES C CXX)

include(FetchContent)

# The URL names an immutable upstream commit. URL_HASH also catches a changed
# archive or an incomplete download before any native code is compiled.
FetchContent_Declare(llama_cpp
    URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/3d82ef62d47fd74e18f36c5eccbdcf965b617b17
    URL_HASH SHA256=3b85f212ab8512b81ef39c9b7ccea323bb126ec596067c3fb06d7c4faf1a60e2
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(llama_cpp)

# build.rs compiles the L2S1 bridge against these exact headers and Jinja files.
file(WRITE "${CMAKE_BINARY_DIR}/l2s1-llama-source.txt" "${llama_cpp_SOURCE_DIR}")
