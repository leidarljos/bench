#!/usr/bin/env bash
#SBATCH --job-name=mab-qa
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
set -uo pipefail
MODEL="$HOME/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
DUMP="${DUMP:-$HOME/mab-dump.jsonl}"
export PATH=/opt/cuda/bin:$PATH
"$HOME/llama.cpp/build/bin/llama-server" -m "$MODEL" -c 65536 -np 2 -ngl 99 -fa on --port 8089 --host 127.0.0.1 \
  > "$HOME/llama-server-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8089/health | grep -q '"ok"' && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8089/health
export QA_BASE_URL=http://127.0.0.1:8089/v1 QA_API_KEY=none QA_MODEL=Qwen2.5-7B-Instruct-Q5_K_M
cd "$HOME/Git/Github/packset"
for spec in "fused live 10" "fused latest 10" "fused 10" "lexical 10" "fused 5"; do
  set -- $spec; top="${!#}"; arm="${spec% *}"
  echo "=== arm: $arm top $top"
  python3 scripts/longmemeval_qa.py "$HOME/data/mab" "$DUMP" --bench mab --chunks "$HOME/mab-chunks" --arm "$arm" --top "$top" --workers 2 \
    --out "$HOME/mab-qa-$(echo "$arm" | tr ' ' '-')-top$top-$SLURM_JOB_ID.jsonl" 2>&1 | tail -30
done
kill $SERVER
echo ONE-DONE
