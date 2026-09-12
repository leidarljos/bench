#!/usr/bin/env bash
#SBATCH --job-name=lme-rerank
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "$HOME/Git/Github/packset"
cargo build --release -p packset-daemon --example longmemeval -p packset-embed 2>&1 | grep -E "^error" -A12 | head -30
export PACKSET_EMBED="$PWD/target/release/packset-embed"
export PACKSET_EMBED_CACHE="$HOME/.cache/packset-embed"
export PACKSET_EMBED_MODEL="${PACKSET_EMBED_MODEL:-e5-large-v2}"
export PACKSET_LME_CACHE="$HOME/.cache/packset-lme"
export PACKSET_LME_QUESTIONS="${PACKSET_LME_QUESTIONS:-100}"
export PACKSET_LME_DUMP="$HOME/lme-dump-100-rerank.jsonl"
export PACKSET_LME_RERANK=1
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8
./target/release/examples/longmemeval "$HOME/data/longmemeval_s.json" | tail -12
wc -l "$PACKSET_LME_DUMP"
echo ONE-DONE
