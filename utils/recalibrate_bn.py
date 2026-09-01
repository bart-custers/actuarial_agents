"""
Offline/periodic recalibration entrypoint for UncertaintyGraphBN.

Mines historical run outcomes (utils/mine_hip_labels.py) and fits HIP-LLM's
OperationalFailureProb per calibratable node -- DataOK/ModelOK/ExplainOK from automatic
decision-routing labels, ReviewDataOK/ReviewModelOK from the hand-filled expert-judgement CSV
(see utils/generate_review_label_template.py and docs/uncertainty_operational_profile.md
§4.2) -- then saves the result to data/audit/bn_calibration.json for agents/central_hub.py to
load at workflow-construction time.

Recalibration cadence is a manual, human-triggered decision for now (still an open question
in docs/uncertainty_operational_profile.md §6) -- run this script by hand after enough new
labeled runs (or expert judgements) have accumulated, then re-run the workflow to pick up the
new calibration.

Usage:
    python -m utils.recalibrate_bn [--memory data/memory/central_memory.json]
                                    [--expert-labels data/audit/expert_review_labels.csv]
                                    [--out data/audit/bn_calibration.json]
                                    [--min-rows 10] [--review-min-rows 3]
"""

import argparse

from utils.mine_hip_labels import DEFAULT_EXPERT_LABELS_PATH, mine_labels, to_hip_inputs
from utils.audit import UncertaintyGraphBN

# ReviewDataOK/ReviewModelOK start from a handful of hand-judged examples, not hundreds of
# automatic runs -- give them their own, lower default sample-size floor.
REVIEW_NODES = {"ReviewDataOK", "ReviewModelOK"}


def recalibrate(memory_path="data/memory/central_memory.json",
                 expert_labels_path=DEFAULT_EXPERT_LABELS_PATH,
                 out_path="data/audit/bn_calibration.json",
                 min_rows=10,
                 review_min_rows=3):
    """
    Mine labels, calibrate every node with enough data, and save the result.

    Args:
        memory_path (str): path to central_memory.json.
        expert_labels_path (str): path to the hand-filled ReviewDataOK/ReviewModelOK
            judgement CSV (see utils/generate_review_label_template.py).
        out_path (str): where to write the calibration JSON.
        min_rows (int): skip calibrating DataOK/ModelOK/ExplainOK if fewer than this many
            labeled rows were mined -- HIP-LLM's posterior bounds are only meaningful with a
            real sample (see docs/uncertainty_operational_profile.md §6, open question 3).
        review_min_rows (int): same threshold, but for ReviewDataOK/ReviewModelOK -- lower by
            default since expert judgements start scarce.

    Returns:
        dict: {node: {"failure_probability_lower": float, "failure_probability_upper": float}}
        for every node that was actually calibrated.
    """
    tables = mine_labels(memory_path, expert_labels_path=expert_labels_path)
    bn = UncertaintyGraphBN()

    calibrated = {}
    for node, df in tables.items():
        threshold = review_min_rows if node in REVIEW_NODES else min_rows
        if len(df) < threshold:
            print(f"[recalibrate_bn] Skipping {node}: only {len(df)} labeled rows "
                  f"(need >= {threshold}).")
            continue
        outcomes, strata, profile = to_hip_inputs(df)
        result = bn.calibrate_node(node, outcomes=outcomes, strata=strata, profile=profile)
        calibrated[node] = result
        print(f"[recalibrate_bn] {node}: n={len(df)} profile={profile} "
              f"failure_probability=[{result['failure_probability_lower']:.3f}, "
              f"{result['failure_probability_upper']:.3f}]")

    bn.save_calibration(out_path)
    print(f"[recalibrate_bn] Saved calibration for {list(calibrated)} to {out_path}")
    return calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", default="data/memory/central_memory.json")
    parser.add_argument("--expert-labels", default=DEFAULT_EXPERT_LABELS_PATH)
    parser.add_argument("--out", default="data/audit/bn_calibration.json")
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--review-min-rows", type=int, default=3)
    args = parser.parse_args()
    recalibrate(memory_path=args.memory, expert_labels_path=args.expert_labels,
                out_path=args.out, min_rows=args.min_rows,
                review_min_rows=args.review_min_rows)


if __name__ == "__main__":
    main()
