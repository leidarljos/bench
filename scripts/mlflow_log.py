#!/usr/bin/env python3
"""Log the package's runs to an MLflow tracking store.

One MLflow run per (job, arm): the parameters as they ran (benchmark, arm,
top-k, reading mode, reader, job id), the metrics the tables print
(accuracy overall and by type from a verdict file; hit@k and recall@k
from a dump), and the raw file as the run's artifact. Nothing here calls a
model; it reads the same files `tables.py` reads, so the tracker and the
tables cannot disagree.

    export MLFLOW_TRACKING_URI=sqlite:///$HOME/mlruns/mlflow.db
    python3 scripts/mlflow_log.py --verdicts ~/lme-qa-*.jsonl --dumps ~/lme-dump-*.jsonl

The experiment is the benchmark (`longmemeval`, `locomo`,
`memoryagentbench`); the run name is the file's stem. A file already
logged (same name under the same experiment) is skipped unless `--again`.
"""
import argparse
import collections
import json
import os
import re
import sys

import mlflow

sys.path.insert(0, os.path.dirname(__file__))
from tables import CUTOFFS, rows_of  # noqa: E402

READER = os.environ.get("QA_MODEL", "Qwen2.5-7B-Instruct-Q5_K_M")

# Filename shapes the jobs write. Groups: arm, top, mode, job.
VERDICT_SHAPES = [
    ("longmemeval", re.compile(r"lme-qa-time-(?P<arm>.+)-top(?P<top>\d+)-(?P<mode>timeline|raw)-(?P<job>\d+)\.jsonl$")),
    ("longmemeval", re.compile(r"lme-qa-gaps-(?P<arm>.+)-top(?P<top>\d+)-(?P<job>\d+)\.jsonl$")),
    ("longmemeval", re.compile(r"lme-qa-(?P<arm>.+)-top(?P<top>\d+)-(?P<job>\d+)\.jsonl$")),
    ("longmemeval", re.compile(r"lme-qa-(?P<arm>.+)-(?P<job>\d+)\.jsonl$")),
    ("locomo", re.compile(r"locomo-learn-(?P<mode>none|fsrs|oracle)-top(?P<top>\d+)-(?P<job>\d+)\.jsonl$")),
    ("locomo", re.compile(r"locomo-qa-(?P<arm>.+)-top(?P<top>\d+)-(?P<job>\d+)\.jsonl$")),
    ("memoryagentbench", re.compile(r"mab-qa-(?P<arm>.+)-top(?P<top>\d+)-(?P<job>\d+)\.jsonl$")),
]
DUMP_SHAPES = [
    ("longmemeval", re.compile(r"lme-dump.*\.jsonl$"), "answer_session_ids", "question_type"),
    ("locomo", re.compile(r"locomo-dump.*\.jsonl$"), "evidence", "category"),
]


def match(path, shapes):
    name = os.path.basename(path)
    for shape in shapes:
        m = shape[1].search(name)
        if m:
            return shape, m
    return None, None


def verdict_metrics(rows):
    by = collections.OrderedDict()
    for r in rows:
        t = by.setdefault(str(r["question_type"]), [0, 0])
        t[0] += bool(r["correct"])
        t[1] += 1
    out = {"accuracy": sum(bool(r["correct"]) for r in rows) / max(1, len(rows)), "asked": len(rows)}
    for kind, (c, n) in by.items():
        key = re.sub(r"[^A-Za-z0-9_.\- /]", "_", kind)
        out[f"accuracy/{key}"] = c / n
        out[f"asked/{key}"] = n
    return out


def dump_metrics(rows, gold):
    out = {"asked": len(rows)}
    arms = list(rows[0]["retrieved"].keys())
    for arm in arms:
        for k in CUTOFFS:
            hit = 0
            recall = 0.0
            for r in rows:
                top = r["retrieved"][arm][:k]
                if not isinstance(top, list) or (top and isinstance(top[0], list)):
                    continue
                truth = r[gold]
                found = [t for t in truth if t in top]
                hit += 1 if found else 0
                recall += len(found) / max(1, len(truth))
            a = re.sub(r"[^A-Za-z0-9_.\- /]", "_", arm)
            out[f"hit{k}/{a}"] = hit / len(rows)
            out[f"recall{k}/{a}"] = recall / len(rows)
    return out


def already(experiment_id, name):
    found = mlflow.search_runs([experiment_id], filter_string=f"tags.mlflow.runName = '{name}'", max_results=1)
    return len(found) > 0


def log_verdicts(path, again):
    shape, m = match(path, VERDICT_SHAPES)
    if not shape:
        print(f"skip {path}: no known shape", file=sys.stderr)
        return
    bench = shape[0]
    g = m.groupdict()
    name = os.path.basename(path)[: -len(".jsonl")]
    exp = mlflow.set_experiment(bench)
    if not again and already(exp.experiment_id, name):
        print(f"have {name}")
        return
    rows = rows_of(path)
    arm = g.get("arm") or ("turns fused, learn " + g.get("mode", ""))
    mode = g.get("mode") or ("timeline" if "gaps" not in name and bench == "longmemeval" and "time" in name else "")
    if "gaps" in name:
        mode = "timeline with gaps"
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "benchmark": bench, "kind": "reading", "arm": arm, "top": g.get("top", ""),
            "mode": mode, "reader": READER, "judge": READER, "job": g.get("job", ""), "file": name,
        })
        mlflow.log_metrics(verdict_metrics(rows))
        mlflow.log_artifact(path)
    print(f"logged {name}")


def log_dump(path, again):
    shape, m = match(path, DUMP_SHAPES)
    if not shape:
        print(f"skip {path}: no known shape", file=sys.stderr)
        return
    bench, _, gold, kind_key = shape
    name = os.path.basename(path)[: -len(".jsonl")]
    exp = mlflow.set_experiment(bench)
    if not again and already(exp.experiment_id, name):
        print(f"have {name}")
        return
    rows = rows_of(path)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"benchmark": bench, "kind": "retrieval", "file": name,
                           "arms": ", ".join(rows[0]["retrieved"].keys())[:500]})
        mlflow.log_metrics(dump_metrics(rows, gold))
        mlflow.log_artifact(path)
    print(f"logged {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", nargs="*", default=[])
    ap.add_argument("--dumps", nargs="*", default=[])
    ap.add_argument("--again", action="store_true", help="log a file even when a run of its name exists")
    a = ap.parse_args()
    for p in a.verdicts:
        log_verdicts(p, a.again)
    for p in a.dumps:
        log_dump(p, a.again)


if __name__ == "__main__":
    main()
