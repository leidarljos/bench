#!/usr/bin/env python3
"""Recompute every table in this package from the raw files beside it.

Retrieval tables come from the dumps: one JSON line a question with the
ids each arm ranked, so hit@k and recall@k are a deterministic count.
Answer-accuracy tables come from the verdict files: one JSON line a
question with the judge's yes or no. Nothing here calls a model; a reader
who has only this directory can regenerate every number in the README.

    python3 scripts/tables.py results/lme-dump-all.jsonl --gold answer_session_ids
    python3 scripts/tables.py results/locomo-dump.jsonl --gold evidence --type category
    python3 scripts/tables.py --verdicts results/lme-qa-oracle-11663.jsonl
"""
import argparse
import collections
import json
import sys

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


def verdicts(path):
    rows = rows_of(path)
    by = collections.OrderedDict()
    for r in rows:
        t = by.setdefault(r["question_type"], [0, 0])
        t[0] += bool(r["correct"])
        t[1] += 1
    print(f"\n{path} (n={len(rows)})\n\n| type | asked | accuracy |\n|---|---|---|")
    for kind, (c, n) in sorted(by.items()):
        print(f"| {kind} | {n} | {c / n:.3f} |")
    total = sum(bool(r["correct"]) for r in rows)
    print(f"| all | {len(rows)} | {total / max(1, len(rows)):.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?")
    ap.add_argument("--gold", default="answer_session_ids")
    ap.add_argument("--type", default="question_type")
    ap.add_argument("--verdicts", action="append", default=[])
    a = ap.parse_args()
    if a.dump:
        retrieval(rows_of(a.dump), a.gold, a.type)
    for v in a.verdicts:
        verdicts(v)
    if not a.dump and not a.verdicts:
        ap.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
