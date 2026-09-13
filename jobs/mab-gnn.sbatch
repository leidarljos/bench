#!/usr/bin/env bash
#SBATCH --job-name=mab-gnn
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
set -uo pipefail
mkdir -p "$HOME/mab-gnn"
for n in 0 1 2 3 4 5 6 7; do
  bin="$HOME/.cache/packset-mab/e5-large-v2-Conflict_Resolution-$n.bin"
  [ -s "$bin" ] || { echo "no cache for row $n"; continue; }
  echo "=== row $n"
  "$HOME/.venvs/mab/bin/uv" run --script "$HOME/Git/Github/landscape/gnn/links.py" --bin "$bin" --k 8 --candidates 30 --epochs 200 --out "$HOME/mab-gnn/Conflict_Resolution-$n.json"
done
ls -la "$HOME/mab-gnn"
echo ONE-DONE
