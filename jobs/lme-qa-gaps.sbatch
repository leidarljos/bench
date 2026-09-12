#!/usr/bin/env bash
#SBATCH --job-name=lme-qa-gaps
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
set -uo pipefail
MODEL="$HOME/models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"
DUMP="${DUMP:-$HOME/lme-dump-all.jsonl}"
export PATH=/opt/cuda/bin:$PATH
"$HOME/llama.cpp/build/bin/llama-server" -m "$MODEL" -c 65536 -np 2 -ngl 99 -fa on --port 8089 --host 127.0.0.1 \
  > "$HOME/llama-server-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8089/health | grep -q '"ok"' && break; sleep 5; done
curl -s -m 5 http://127.0.0.1:8089/health
export QA_BASE_URL=http://127.0.0.1:8089/v1 QA_API_KEY=none QA_MODEL=Qwen2.5-7B-Instruct-Q5_K_M
cd "$HOME/Git/Github/packset"
TYPES="temporal-reasoning,knowledge-update,multi-session,single-session-preference"
echo "=== arm: sessions fused top 5 gaps"
python3 scripts/longmemeval_qa.py "$HOME/data/longmemeval_s.json" "$DUMP" --arm "sessions fused" --top 5 --types "$TYPES" --workers 2 --gaps \
  --out "$HOME/lme-qa-gaps-sessions-fused-top5-$SLURM_JOB_ID.jsonl" 2>&1 | tail -12
kill $SERVER
echo ONE-DONE
