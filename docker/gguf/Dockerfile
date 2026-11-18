# Gurdy 2.0 — GGUF quantiser image (CPU only): llama-quantize built from the PINNED llama.cpp release tag.
# Used by scripts/build_gguf.sh (merge + convert run in gurdy2/brain-train:cuda on the CPU; this image only quantises).
# Build context: the repo root is fine; nothing is copied from it.
FROM ubuntu:24.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_TAG=b12494
ARG LLAMA_SHA256=b40d84985b98330ca2d51c6eef3ead6e7ec1aae2af97f7452af708e5900e2fcb
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o llama.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_TAG}.tar.gz \
    && echo "${LLAMA_SHA256}  llama.tar.gz" | sha256sum -c - \
    && tar xzf llama.tar.gz --strip-components=1 && rm llama.tar.gz
# only the quantiser, CPU backend, static llama libs (no CUDA, no curl, no tests/examples)
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
          -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --target llama-quantize -j2

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/llama-quantize /usr/local/bin/llama-quantize
LABEL llama_cpp_tag=b12494
USER ubuntu
ENTRYPOINT ["llama-quantize"]
