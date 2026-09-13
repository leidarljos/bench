#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlflow>=3",
#     "cyclopts>=3",
# ]
# ///
"""Log the package's runs to an MLflow tracking store.

One MLflow run per (job, arm): the parameters as they ran (benchmark, arm,
top-k, reading mode, reader, job id), the metrics the tables print
(accuracy overall and by type from a verdict file; hit@k and recall@k
from a dump), and the raw file as the run's artifact. Nothing here calls a
model; it reads the same files `tables.py` reads, so the tracker and the
tables cannot disagree.

    export MLFLOW_TRACKING_URI=sqlite:///$HOME/mlruns/mlflow.db
    scripts/mlflow_log.py --verdicts ~/lme-qa-*.jsonl --dumps ~/lme-dump-*.jsonl

The experiment is the benchmark (`longmemeval`, `locomo`,
`memoryagentbench`); the run name is the file's stem. A file already
logged (same name under the same experiment) is skipped unless `--again`.
"""

from __future__ import annotations

import collections
import os
import re
import sys
from pathlib import Path

import cyclopts
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
from tables import CUTOFFS, rows_of  # noqa: E402

app = cyclopts.App(help=__doc__)

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
    ("longmemeval", re.compile(r"lme-dump.*\.jsonl$"), "answer_session_ids"),
    ("locomo", re.compile(r"locomo-dump.*\.jsonl$"), "evidence"),
]


def clean(name: str) -> str:
    """An MLflow metric name: letters, digits, and a few marks."""
    return re.sub(r"[^A-Za-z0-9_.\- /]", "_", name)


def match(path: Path, shapes):
    for shape in shapes:
        if m := shape[1].search(path.name):
            return shape, m
    return None, None


def verdict_metrics(rows) -> dict[str, float]:
    by: dict[str, list[int]] = collections.OrderedDict()
    for r in rows:
        t = by.setdefault(str(r["question_type"]), [0, 0])
        t[0] += bool(r["correct"])
        t[1] += 1
    out = {"accuracy": sum(bool(r["correct"]) for r in rows) / max(1, len(rows)), "asked": len(rows)}
    for kind, (c, n) in by.items():
        out[f"accuracy/{clean(kind)}"] = c / n
        out[f"asked/{clean(kind)}"] = n
    return out


def dump_metrics(rows, gold: str) -> dict[str, float]:
    out = {"asked": len(rows)}
    for arm in rows[0]["retrieved"]:
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
            out[f"hit{k}/{clean(arm)}"] = hit / len(rows)
            out[f"recall{k}/{clean(arm)}"] = recall / len(rows)
    return out


def already(experiment_id: str, name: str) -> bool:
    found = mlflow.search_runs([experiment_id], filter_string=f"tags.mlflow.runName = '{name}'", max_results=1)
    return len(found) > 0


def log_verdicts(path: Path, again: bool):
    shape, m = match(path, VERDICT_SHAPES)
    if not shape:
        print(f"skip {path}: no known shape", file=sys.stderr)
        return
    bench = shape[0]
    g = m.groupdict()
    name = path.stem
    exp = mlflow.set_experiment(bench)
    if not again and already(exp.experiment_id, name):
        print(f"have {name}")
        return
    rows = rows_of(path)
    arm = g.get("arm") or ("turns fused, learn " + g.get("mode", ""))
    mode = g.get("mode") or ""
    if "gaps" in name:
        mode = "timeline with gaps"
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "benchmark": bench, "kind": "reading", "arm": arm, "top": g.get("top", ""),
            "mode": mode, "reader": READER, "judge": READER, "job": g.get("job", ""), "file": name,
        })
        mlflow.log_metrics(verdict_metrics(rows))
        mlflow.log_artifact(str(path))
    print(f"logged {name}")


def log_dump(path: Path, again: bool):
    shape, m = match(path, DUMP_SHAPES)
    if not shape:
        print(f"skip {path}: no known shape", file=sys.stderr)
        return
    bench, _, gold = shape
    name = path.stem
    exp = mlflow.set_experiment(bench)
    if not again and already(exp.experiment_id, name):
        print(f"have {name}")
        return
    rows = rows_of(path)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"benchmark": bench, "kind": "retrieval", "file": name,
                           "arms": ", ".join(rows[0]["retrieved"])[:500]})
        mlflow.log_metrics(dump_metrics(rows, gold))
        mlflow.log_artifact(str(path))
    print(f"logged {name}")


@app.default
def main(*, verdicts: list[Path] = [], dumps: list[Path] = [], again: bool = False):
    """Log verdict files and retrieval dumps as MLflow runs.

    Parameters
    ----------
    verdicts
        Verdict files, one JSON line a question with the judge's yes or no.
    dumps
        Retrieval dumps, one JSON line a question with the ids each arm ranked.
    again
        Log a file even when a run of its name exists.
    """
    for p in verdicts:
        log_verdicts(p, again)
    for p in dumps:
        log_dump(p, again)


if __name__ == "__main__":
    app()
