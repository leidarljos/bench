#!/usr/bin/env bash
#SBATCH --job-name=llama-build
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
export PATH=/opt/cuda/bin:$PATH
nvcc --version | tail -1; gcc --version | head -1; ls /usr/bin/gcc-* 2>/dev/null
cd "$HOME"
if [ ! -d llama.cpp ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp.git; fi
cd llama.cpp
HOST=""
for c in /usr/bin/gcc-14 /usr/bin/gcc-13; do [ -x "$c" ] && HOST="-DCMAKE_CUDA_HOST_COMPILER=$c" && break; done
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF $HOST 2>&1 | tail -5
cmake --build build --config Release --target llama-server -j 8 2>&1 | grep -E "error|Error|warning: #|Built target" | tail -8
ls -la build/bin/llama-server
echo ONE-DONE
