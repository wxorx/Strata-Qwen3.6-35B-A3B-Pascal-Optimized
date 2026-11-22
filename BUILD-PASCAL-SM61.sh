#!/usr/bin/env bash
# Strata Qwen3.6 Pascal-optimized engine: Linux build for sm_61 (mirror of BUILD-PASCAL-SM61.bat).
# Tested on Ubuntu 24.04: gcc 13, CMake 3.28, CUDA Toolkit 12.6, Quadro P1000 (sm_61).
#
# The engine's own CMake already carries GCC/POSIX branches; nothing in the C++/CUDA sources
# needed changing for Linux. Two decisions differ from the Windows script:
#  * -DSQ_ENABLE_MMQ_KQUANTS=ON also compiles llama.cpp's Q4_K/Q5_K/Q5_1/Q6_K MMQ kernels,
#    which the layer-major prefill requires for Unsloth UD-Q4_K_M-style GGUFs. It stays
#    default-OFF in CMake, so the published Windows artifact is byte-for-byte unchanged.
#  * the plain `strata-qwen35moe` target is built (the `-optimized` target only adds a name tag).
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_DIR="$PWD/third_party/llama.cpp"
BUILD_DIR="$PWD/build-sm61"
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
NVCC=${NVCC:-$(command -v nvcc || true)}
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
[ -z "$NVCC" ] && { echo "ERROR: nvcc (CUDA Toolkit 12.x) is required."; exit 1; }
CUDA_ROOT=$(dirname "$(dirname "$NVCC")")

command -v git   >/dev/null || { echo "ERROR: git is required."; exit 1; }
command -v cmake >/dev/null || { echo "ERROR: cmake (>= 3.24) is required."; exit 1; }

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo "Fetching pinned llama.cpp source..."
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" config core.longpaths true
git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_COMMIT"
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_COMMIT"

cmake -S engines/qwen35moe -B "$BUILD_DIR" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DSQ_ENABLE_PREFILL_MMQ=ON \
  -DSQ_ENABLE_MMQ_KQUANTS=ON \
  -DSQ_LLAMA_DIR="$LLAMA_DIR"

cmake --build "$BUILD_DIR" --target strata-qwen35moe -j"$(nproc)"

echo
echo "Built: $BUILD_DIR/strata-qwen35moe"
