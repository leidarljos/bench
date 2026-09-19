#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["cyclopts>=3"]
# ///
"""Recompute every table in this package from the raw files beside it.

Retrieval tables come from the dumps: one JSON line a question with the
ids each arm ranked, so hit@k and recall@k are a deterministic count.
Answer-accuracy tables come from the verdict files: one JSON line a
question with the judge's yes or no. Nothing here calls a model; a reader
who has only this directory can regenerate every number in the README.

    scripts/tables.py results/lme-dump-all.jsonl --gold answer_session_ids
    scripts/tables.py results/locomo-dump.jsonl --gold evidence --type category
    scripts/tables.py --verdicts results/lme-qa-oracle-11663.jsonl

Accuracy rows carry a 95% bootstrap interval, and two or more verdict
files given together print the paired difference of each against the one
before it, over the questions both answered, with its interval.
"""
import collections
import json
import sys
from pathlib import Path

import cyclopts

app = cyclopts.App(help=__doc__)

CUTOFFS = (1, 5, 10)


def rows_of(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval(rows, gold, kind_key):
    arms = list(rows[0]["retrieved"].keys())
    by = collections.OrderedDict()
    by["all"] = rows
    for r in rows:
        by.setdefault(str(r.get(kind_key, "?")), []).append(r)
    for label, rs in by.items():
        if label != "all":
            rs = by[label]
        print(f"\n{label} (n={len(rs)})\n")
        print("| arm | " + " | ".join(f"hit@{k} | recall@{k}" for k in CUTOFFS) + " |")
        print("|---|" + "---|---|" * len(CUTOFFS))
        for arm in arms:
            cells = []
            for k in CUTOFFS:
                hit = 0
                recall = 0.0
                for r in rs:
                    top = r["retrieved"][arm][:k]
                    truth = r[gold]
                    found = [t for t in truth if t in top]
                    hit += 1 if found else 0
                    recall += len(found) / max(1, len(truth))
                cells.append(f"{hit / len(rs):.3f} | {recall / len(rs):.3f}")
            print(f"| {arm} | " + " | ".join(cells) + " |")


# Bootstrap resamples for the intervals; the seed makes the tables
# regenerate byte for byte.
BOOT = 2000
SEED = 20260912


def interval(values, seed=SEED):
    """A 95% percentile bootstrap interval on the mean of 0/1 values."""
    import random

    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(BOOT):
        s = 0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return (means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1])


def verdicts_of(path):
    rows = rows_of(path)
    by = collections.OrderedDict()
    for r in rows:
        t = by.setdefault(r["question_type"], [0, 0])
        t[0] += bool(r["correct"])
        t[1] += 1
    print(f"\n{path} (n={len(rows)})\n\n| type | asked | accuracy | 95% interval |\n|---|---|---|---|")
    for kind, (c, n) in sorted(by.items()):
        lo, hi = interval([int(bool(r["correct"])) for r in rows if r["question_type"] == kind])
        print(f"| {kind} | {n} | {c / n:.3f} | {lo:.3f} to {hi:.3f} |")
    total = sum(bool(r["correct"]) for r in rows)
    lo, hi = interval([int(bool(r["correct"])) for r in rows])
    print(f"| all | {len(rows)} | {total / max(1, len(rows)):.3f} | {lo:.3f} to {hi:.3f} |")
    return rows


def paired(a_path, a_rows, b_path, b_rows):
    """The difference between two verdict files on the questions both
    answered: the mean of (b - a) with a paired bootstrap interval. The
    interval says whether an arm's gain is more than the reader's noise."""
    a = {r["question_id"]: int(bool(r["correct"])) for r in a_rows}
    b = {r["question_id"]: int(bool(r["correct"])) for r in b_rows}
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return
    diffs = [b[q] - a[q] for q in shared]
    mean = sum(diffs) / len(diffs)
    import random

    rng = random.Random(SEED)
    means = []
    for _ in range(BOOT):
        s = 0
        for _ in range(len(diffs)):
            s += diffs[rng.randrange(len(diffs))]
        means.append(s / len(diffs))
    means.sort()
    lo, hi = means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]
    verdict = "the interval excludes zero" if lo > 0 or hi < 0 else "the interval includes zero"
    print(f"\npaired: {b_path} minus {a_path} over {len(shared)} shared questions: "
          f"{mean:+.3f} ({lo:+.3f} to {hi:+.3f}); {verdict}")
    # The same by type, where the arms are meant to differ.
    kinds = {r["question_id"]: str(r["question_type"]) for r in b_rows}
    for kind in sorted(set(kinds[q] for q in shared)):
        qs = [q for q in shared if kinds[q] == kind]
        d = [b[q] - a[q] for q in qs]
        if len(d) < 2:
            continue
        rng = random.Random(SEED)
        ms = []
        for _ in range(BOOT):
            s = 0
            for _ in range(len(d)):
                s += d[rng.randrange(len(d))]
            ms.append(s / len(d))
        ms.sort()
        klo, khi = ms[int(0.025 * BOOT)], ms[int(0.975 * BOOT) - 1]
        mark = "*" if klo > 0 or khi < 0 else ""
        print(f"  {kind}: {sum(d) / len(d):+.3f} ({klo:+.3f} to {khi:+.3f}) over {len(d)}{mark}")


@app.default
def main(
    dump: Path | None = None,
    *,
    gold: str = "answer_session_ids",
    type: str = "question_type",
    verdicts: list[Path] = [],
):
    """Print the tables a dump and verdict files carry.

    Parameters
    ----------
    dump
        A retrieval dump, one JSON line a question with the ids each arm ranked.
    gold
        The field that holds the gold ids in the dump.
    type
        The field that groups questions into rows.
    verdicts
        Verdict files, each a JSON line a question with the judge's yes or no;
        two or more print the paired difference of each against the one before.
    """
    if dump is not None:
        retrieval(rows_of(dump), gold, type)
    previous = None
    for v in verdicts:
        rows = verdicts_of(v)
        if previous is not None:
            paired(previous[0], previous[1], v, rows)
        previous = (v, rows)
    if dump is None and not verdicts:
        print("nothing to print; give a dump or --verdicts", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    app()
