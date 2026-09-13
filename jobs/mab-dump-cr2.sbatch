#!/usr/bin/env bash
#SBATCH --job-name=mab-dump-cr2
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
cd "$HOME/Git/Github/packset"
cargo build --release -p packset-daemon --example memoryagentbench -p packset-embed 2>&1 | grep -E "^error" -A12 | head -30
export PACKSET_EMBED="$PWD/target/release/packset-embed"
export PACKSET_EMBED_CACHE="$HOME/.cache/packset-embed"
export PACKSET_EMBED_MODEL="${PACKSET_EMBED_MODEL:-e5-large-v2}"
export PACKSET_MAB_CACHE="$HOME/.cache/packset-mab"
export PACKSET_MAB_SPLITS="Conflict_Resolution"
export PACKSET_MAB_DUMP="$HOME/mab-dump-cr2.jsonl"
export PACKSET_MAB_SURVEY="$HOME/mab-survey"
export PACKSET_MAB_GNN="$HOME/mab-gnn"
export PACKSET_MAB_CHUNKS="$HOME/mab-chunks"
export RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4
./target/release/examples/memoryagentbench "$HOME/data/mab" | tail -60
wc -l "$PACKSET_MAB_DUMP"
echo ONE-DONE
