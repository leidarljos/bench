#!/usr/bin/env bash
#SBATCH --job-name=model-get
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
mkdir -p "$HOME/models" && cd "$HOME/models"
f=Qwen2.5-7B-Instruct-Q5_K_M.gguf
[ -s "$f" ] || curl -L --retry 5 -o "$f" "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/$f"
ls -la "$f"; sha256sum "$f"
echo ONE-DONE
