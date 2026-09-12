#!/usr/bin/env bash
#SBATCH --job-name=mab-get
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --output=%x-%j.out
set -uo pipefail
mkdir -p "$HOME/data/mab" && cd "$HOME/data/mab"
[ -x "$HOME/.venvs/mab/bin/python" ] || python3 -m venv "$HOME/.venvs/mab"
"$HOME/.venvs/mab/bin/pip" -q install pyarrow 2>&1 | tail -1
for split in Accurate_Retrieval Conflict_Resolution Test_Time_Learning Long_Range_Understanding; do
  [ -s "$split.parquet" ] || curl -sL --retry 5 -o "$split.parquet" "https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/parquet/default/$split/0.parquet"
done
ls -la *.parquet; sha256sum *.parquet
"$HOME/.venvs/mab/bin/python" - <<'PY'
import pyarrow.parquet as pq, json
for split in ["Accurate_Retrieval","Conflict_Resolution","Test_Time_Learning","Long_Range_Understanding"]:
    t = pq.read_table(f"{split}.parquet")
    rows = t.to_pylist()
    with open(f"{split}.jsonl","w") as f:
        for r in rows:
            f.write(json.dumps(r)+"\n")
    r = rows[0]
    print(split, len(rows), "rows; fields:", list(r.keys()))
    print("  context chars", len(r["context"]), "questions", len(r["questions"]), "answers", len(r["answers"]))
    print("  metadata keys", list(r["metadata"].keys()) if isinstance(r["metadata"], dict) else type(r["metadata"]))
    print("  q0:", str(r["questions"][0])[:200])
    print("  a0:", str(r["answers"][0])[:200])
    print("  ctx head:", r["context"][:300].replace("\n"," | "))
    md = r["metadata"]
    if isinstance(md, dict):
        for k,v in md.items():
            print("  md", k, str(v)[:200])
PY
echo ONE-DONE
