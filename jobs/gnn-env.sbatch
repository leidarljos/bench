#!/usr/bin/env bash
#SBATCH --job-name=gnn-env
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
UV="$HOME/.venvs/mab/bin/uv"
PY="$HOME/.venvs/gnn/bin/python"
[ -x "$PY" ] || "$UV" venv "$HOME/.venvs/gnn" --python 3.12
"$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2
"$UV" pip install --python "$PY" torch_geometric numpy scikit-learn mlflow 2>&1 | tail -2
"$PY" - <<'PY'
import torch, torch_geometric
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("pyg", torch_geometric.__version__)
PY
echo ONE-DONE
