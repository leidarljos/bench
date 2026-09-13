#!/usr/bin/env bash
#SBATCH --job-name=mab-survey
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
mkdir -p "$HOME/mab-survey"
export RAYON_NUM_THREADS=8
for n in 0 1 2 3 4 5 6 7; do
  bin="$HOME/.cache/packset-mab/e5-large-v2-Conflict_Resolution-$n.bin"
  texts="$HOME/mab-chunks/Conflict_Resolution-$n.json"
  [ -s "$bin" ] || { echo "no cache for row $n"; continue; }
  echo "=== row $n"
  "$HOME/bin-out/landscape" --bin "$bin" --texts "$texts" --k 5 --max-steps 600 --starts 1500 --pairs 60 --sigma-scale 0.35 > "$HOME/mab-survey/Conflict_Resolution-$n.json"
done
ls -la "$HOME/mab-survey"
echo ONE-DONE
