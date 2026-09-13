#!/usr/bin/env bash
#SBATCH --job-name=saddle-conflicts
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
export RAYON_NUM_THREADS=8
mkdir -p "$HOME/saddle-conflicts"
for n in 0 4; do
  bin="$HOME/.cache/packset-mab/e5-large-v2-Conflict_Resolution-$n.bin"
  texts="$HOME/mab-chunks/Conflict_Resolution-$n.json"
  for scale in 0.2 0.15; do
    echo "=== row $n scale $scale"
    "$HOME/bin-out/landscape" --bin "$bin" --texts "$texts" --sigma-scale "$scale" --k 5 --max-steps 300 --named > "$HOME/saddle-conflicts/row$n-scale$scale.json"
    "$HOME/.venvs/mab/bin/uv" run --script "$HOME/Git/Github/landscape/analysis/saddle_conflicts.py" "$HOME/saddle-conflicts/row$n-scale$scale.json" "$texts"
  done
done
echo ONE-DONE
