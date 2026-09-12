#!/usr/bin/env bash
#SBATCH --job-name=lme-full
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --partition=long
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "$HOME/Git/Github/packset"
cargo build --release -p packset-daemon --example longmemeval -p packset-embed 2>&1 | grep -E "^error" -A12 | head -30
export PACKSET_EMBED="$PWD/target/release/packset-embed"
export PACKSET_EMBED_CACHE="$HOME/.cache/packset-embed"
export PACKSET_EMBED_MODEL="${PACKSET_EMBED_MODEL:-e5-large-v2}"
export PACKSET_LME_CACHE="$HOME/.cache/packset-lme"
unset PACKSET_LME_QUESTIONS
export RAYON_NUM_THREADS=12 OMP_NUM_THREADS=12
./target/release/examples/longmemeval "$HOME/data/longmemeval_s.json"
echo ONE-DONE
