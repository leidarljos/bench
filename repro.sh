#!/usr/bin/env bash
# Reproduce the package from nothing, on a Slurm node with one GPU.
# Each step is one job script under jobs/; each checks its inputs by hash.
# Tables never need a model: `python3 scripts/tables.py` regenerates them
# from results/ alone, which is the reproduction most readers want.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

check() { # file expected-sha256
  echo "$2  $1" | sha256sum -c - >/dev/null && echo "ok  $1" || { echo "sha256 differs: $1" >&2; exit 1; }
}

case "${1:-tables}" in
  tables)
    cd "$here"
    python3 scripts/tables.py results/lme-dump-all.jsonl
    python3 scripts/tables.py results/lme-dump-all-window.jsonl
    python3 scripts/tables.py results/lme-dump-100-rerank.jsonl
    python3 scripts/tables.py results/locomo-dump.jsonl --gold evidence --type category
    python3 scripts/tables.py --verdicts results/lme-qa-oracle-11663.jsonl \
      --verdicts results/lme-qa-sessions-fused-11663.jsonl --verdicts results/lme-qa-sessions-11663.jsonl
    python3 scripts/tables.py --verdicts results/locomo-qa-oracle-top10-11672.jsonl \
      --verdicts results/locomo-qa-turns-fused-top20-11672.jsonl \
      --verdicts results/locomo-qa-turns-fused-top10-11672.jsonl --verdicts results/locomo-qa-turns-top10-11672.jsonl
    ;;
  data)
    mkdir -p "$HOME/data" "$HOME/models"
    [ -s "$HOME/data/longmemeval_s.json" ] || curl -sL -o "$HOME/data/longmemeval_s.json" \
      https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
    check "$HOME/data/longmemeval_s.json" 08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894
    [ -s "$HOME/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf" ] || curl -L --retry 5 -o "$HOME/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf" \
      https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q5_K_M.gguf
    check "$HOME/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf" 2e998d7e181c8756c5ffc55231b9ee1cdc9d3acec4245d6e27d32bd8e738c474
    echo "locomo10.json comes from the LoCoMo repository; expected sha256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    ;;
  jobs)
    # The source: the packset repository at the commits manifest.toml names,
    # checked out under ~/Git/Github/packset, is what the job scripts build.
    cd "$here/jobs"
    b=$(sbatch --parsable llama-build.sbatch); m=$(sbatch --parsable model-get.sbatch)
    r=$(sbatch --parsable lme-full.sbatch)
    d=$(sbatch --parsable --dependency=afterok:$r lme-dump-all.sbatch)
    sbatch --dependency=afterok:$b:$m:$d lme-qa-all.sbatch
    l=$(sbatch --parsable --dependency=afterok:$r locomo-dump.sbatch)
    sbatch --dependency=afterok:$b:$m:$l locomo-qa.sbatch
    ;;
  *) echo "usage: repro.sh [tables|data|jobs]" >&2; exit 2 ;;
esac
