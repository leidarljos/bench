#!/usr/bin/env bash
#SBATCH --job-name=lme-dump
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
export PACKSET_LME_DUMP="$HOME/lme-dump-${PACKSET_LME_QUESTIONS}.jsonl"
export RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4
./target/release/examples/longmemeval "$HOME/data/longmemeval_s.json" | tail -12
wc -l "$PACKSET_LME_DUMP"
echo ONE-DONE
