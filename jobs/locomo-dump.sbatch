#!/usr/bin/env bash
#SBATCH --job-name=locomo-dump
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "$HOME/Git/Github/packset"
cargo build --release -p packset-daemon --example locomo_dump -p packset-embed 2>&1 | grep -E "^error" -A12 | head -30
export PACKSET_EMBED="$PWD/target/release/packset-embed"
export PACKSET_EMBED_CACHE="$HOME/.cache/packset-embed"
export PACKSET_EMBED_MODEL="${PACKSET_EMBED_MODEL:-e5-large-v2}"
export PACKSET_LME_CACHE="$HOME/.cache/packset-lme"
export PACKSET_LOCOMO_DUMP="$HOME/locomo-dump.jsonl"
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8
./target/release/examples/locomo_dump "$HOME/locomo10.json"
wc -l "$PACKSET_LOCOMO_DUMP"
echo ONE-DONE
