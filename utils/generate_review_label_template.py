"""
Generate/update a CSV template for expert ground-truth judgement of ReviewDataOK / ReviewModelOK.

Neither node has an automatic outcome label the way DataOK/ModelOK/ExplainOK do (see
docs/uncertainty_operational_profile.md §4.2): "was the review agent's own judgment
trustworthy on this run" has no independent check anywhere in the workflow itself. Instead,
an actuary/expert fills in the `outcome` column by hand:

    1 = the review's decision on this run was the right call
    0 = it wasn't (e.g. it approved something that shouldn't have proceeded, or rejected
        something that was actually fine)

Start with a handful of hand-judged examples; growing into "stats from many runs" later uses
this exact same file -- just more filled-in rows, mined by utils/mine_hip_labels.py exactly
like the other nodes.

Usage:
    python -m utils.generate_review_label_template
      [--memory data/memory/central_memory.json]
      [--out data/audit/expert_review_labels.csv]

Safe to re-run: only appends rows for review_history timestamps not already present in --out,
so any judgements already filled in are never overwritten.
"""

import argparse
import json
import os

import pandas as pd

DEFAULT_MEMORY_PATH = "data/memory/central_memory.json"
DEFAULT_OUT_PATH = "data/audit/expert_review_labels.csv"

# "decision"/"action"/"metadata_file" are context only (so the expert doesn't have to open
# the raw JSON to remember what happened) -- mining only reads "phase_reviewed"/"outcome"/
# "stratum" (see utils/mine_hip_labels.py).
COLUMNS = ["timestamp", "phase_reviewed", "decision", "action", "metadata_file",
           "outcome", "stratum", "notes"]


def _load_review_history(memory_path):
    with open(memory_path, "r") as f:
        memory = json.load(f)
    return memory.get("review_history", [])


def generate_template(memory_path=DEFAULT_MEMORY_PATH, out_path=DEFAULT_OUT_PATH):
    """
    Append any not-yet-templated review_history rows to `out_path`, leaving `outcome`/
    `stratum`/`notes` blank for an expert to fill in. Never overwrites existing rows.

    Returns:
        str: `out_path`.
    """
    review_history = _load_review_history(memory_path)

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    else:
        existing = pd.DataFrame(columns=COLUMNS)
    already_templated = set(existing["timestamp"].astype(str)) if not existing.empty else set()

    new_rows = []
    for rec in review_history:
        ts = str(rec.get("timestamp"))
        if ts in already_templated:
            continue
        new_rows.append({
            "timestamp": ts,
            "phase_reviewed": rec.get("phase_reviewed"),
            "decision": rec.get("decision"),
            "action": rec.get("action"),
            "metadata_file": rec.get("metadata_file"),
            "outcome": "",
            "stratum": "",
            "notes": "",
        })

    if not new_rows:
        print(f"[generate_review_label_template] No new review_history rows to template; "
              f"{out_path} already covers all {len(already_templated)}.")
        return out_path

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"[generate_review_label_template] Added {len(new_rows)} new row(s) to {out_path}. "
          f"Fill in 'outcome' (1 = review's judgment was correct/trustworthy, 0 = it wasn't) "
          f"for as many rows as you're able to judge; 'stratum'/'notes' are optional.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()
    generate_template(memory_path=args.memory, out_path=args.out)


if __name__ == "__main__":
    main()
