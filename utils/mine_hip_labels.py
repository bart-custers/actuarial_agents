"""
Mine data/memory/central_memory.json into per-BN-node (outcome, stratum) label tables for
HIP-LLM's OperationalFailureProb calibration.

See docs/uncertainty_operational_profile.md (esp. §4.1-4.4) for the design this implements:
- DataOK / ModelOK outcome labels come from the paired Review decision's "action" field
  (agents/review_agent.py, utils/decision_mapping.py::ROUTING_MAP_REVIEW).
- ExplainOK's outcome label comes from the explanation phase's own "action" field
  (agents/explanation_agent.py, utils/decision_mapping.py::ROUTING_MAP_EXPLANATION).
- ReviewDataOK / ReviewModelOK have no automatic ground truth either (see
  docs/uncertainty_operational_profile.md §4.2) -- their labels instead come from a
  hand-filled expert-judgement CSV (see utils/generate_review_label_template.py),
  starting with a handful of examples and growing into "many runs" via the same file.
- Strata: DataOK <- used_pipeline ("adaptive"/"deterministic"), ModelOK <- model_type_used
  ("glm"/"gbm"), ExplainOK <- a single "all" stratum (no comparable field exists yet),
  ReviewDataOK/ReviewModelOK <- whatever the expert enters, defaulting to "all".

Each metadata dict in central_memory.json carries its own "timestamp" field, formatted
"%Y%m%d_%H%M%S" by every agent (agents/*.py, `timestamp = datetime.now().strftime(...)`).
A phase run and its review happen strictly sequentially within one workflow iteration
(agents/central_hub.py), so each review/explanation record is matched to the most recent
preceding phase record via `pd.merge_asof(..., direction="backward")` on that timestamp.

Usage (prints row counts/empirical profile per node -- does not fit or write anything):
    python -m utils.mine_hip_labels
"""

import json
import os

import pandas as pd

DEFAULT_MEMORY_PATH = "data/memory/central_memory.json"
DEFAULT_EXPERT_LABELS_PATH = "data/audit/expert_review_labels.csv"

# Mirrors how "approve_with_notes" already maps to "proceed" (treated as success) in
# utils/decision_mapping.py::ROUTING_MAP_REVIEW / ROUTING_MAP_EXPLANATION.
REVIEW_ACTION_LABEL = {
    "proceed": 1,
    "reclean_data": 0,
    "retrain_model": 0,
    "abort_workflow": 0,
}
EXPLANATION_ACTION_LABEL = {
    "finalize": 1,
    "consult_actuary": 1,  # escalation, not a verdict that the explanation itself was wrong
    "retrain_model": 0,
    "abort_workflow": 0,
}


def _load_memory(memory_path=DEFAULT_MEMORY_PATH):
    with open(memory_path, "r") as f:
        return json.load(f)


def _records_df(records, extra_fields):
    """Build a DataFrame with a parsed `ts` column plus the requested top-level fields."""
    rows = []
    for rec in records:
        if "timestamp" not in rec:
            continue
        row = {"ts": pd.to_datetime(rec["timestamp"], format="%Y%m%d_%H%M%S")}
        for field in extra_fields:
            row[field] = rec.get(field)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def _dataprep_logs(memory):
    logs = memory.get("logs", [])
    records = [
        entry["content"]
        for entry in logs
        if entry.get("event_type") == "data_preparation" and "content" in entry
    ]
    return _records_df(records, extra_fields=["used_pipeline"])


def _model_logs(memory):
    return _records_df(memory.get("model_history", []), extra_fields=["model_type_used"])


def _review_history(memory, phase_reviewed):
    df = _records_df(memory.get("review_history", []), extra_fields=["phase_reviewed", "action"])
    if df.empty:
        return df
    return df[df["phase_reviewed"] == phase_reviewed].reset_index(drop=True)


def _explanation_history(memory):
    return _records_df(memory.get("explanation_history", []), extra_fields=["action"])


def _label_and_join(review_df, source_df, source_stratum_field):
    """Attach an outcome (from `action`) and a stratum (from the paired phase run) per row."""
    if review_df.empty or source_df.empty:
        return pd.DataFrame(columns=["timestamp", "outcome", "stratum"])

    review_df = review_df.copy()
    review_df["outcome"] = review_df["action"].map(REVIEW_ACTION_LABEL)
    review_df = review_df.dropna(subset=["outcome"])

    merged = pd.merge_asof(
        review_df.sort_values("ts"),
        source_df.sort_values("ts"),
        on="ts",
        direction="backward",
        suffixes=("", "_source"),
    )
    merged["stratum"] = merged[source_stratum_field]
    merged = merged.dropna(subset=["stratum"])
    merged["timestamp"] = merged["ts"].dt.strftime("%Y%m%d_%H%M%S")
    return merged[["timestamp", "outcome", "stratum"]].reset_index(drop=True)


def mine_dataok_labels(memory):
    review_df = _review_history(memory, phase_reviewed="dataprep")
    source_df = _dataprep_logs(memory)
    return _label_and_join(review_df, source_df, "used_pipeline")


def mine_modelok_labels(memory):
    review_df = _review_history(memory, phase_reviewed="modelling")
    source_df = _model_logs(memory)
    return _label_and_join(review_df, source_df, "model_type_used")


def mine_explainok_labels(memory):
    df = _explanation_history(memory)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "outcome", "stratum"])
    df = df.copy()
    df["outcome"] = df["action"].map(EXPLANATION_ACTION_LABEL)
    df = df.dropna(subset=["outcome"])
    df["stratum"] = "all"
    df["timestamp"] = df["ts"].dt.strftime("%Y%m%d_%H%M%S")
    return df[["timestamp", "outcome", "stratum"]].reset_index(drop=True)


def _expert_review_labels(phase_reviewed, expert_labels_path=DEFAULT_EXPERT_LABELS_PATH):
    """
    Ground-truth labels for ReviewDataOK/ReviewModelOK, hand-filled by an expert into the
    template produced by utils/generate_review_label_template.py. No independent automatic
    outcome signal exists for these two nodes (docs/uncertainty_operational_profile.md §4.2),
    so unlike the other mine_*_labels functions this reads a human-curated CSV, not
    central_memory.json.
    """
    empty = pd.DataFrame(columns=["timestamp", "outcome", "stratum"])
    if not os.path.exists(expert_labels_path):
        return empty

    df = pd.read_csv(expert_labels_path, dtype=str, keep_default_na=False)
    df = df[df["phase_reviewed"] == phase_reviewed]
    df = df[df["outcome"].str.strip() != ""]
    if df.empty:
        return empty

    df = df.copy()
    df["outcome"] = df["outcome"].astype(int)
    df["stratum"] = df["stratum"].str.strip().replace("", "all")
    return df[["timestamp", "outcome", "stratum"]].reset_index(drop=True)


def mine_reviewdataok_labels(expert_labels_path=DEFAULT_EXPERT_LABELS_PATH):
    return _expert_review_labels("dataprep", expert_labels_path)


def mine_reviewmodelok_labels(expert_labels_path=DEFAULT_EXPERT_LABELS_PATH):
    return _expert_review_labels("modelling", expert_labels_path)


def mine_labels(memory_path=DEFAULT_MEMORY_PATH, expert_labels_path=DEFAULT_EXPERT_LABELS_PATH):
    """Return one DataFrame per BN node (DataOK, ModelOK, ExplainOK, ReviewDataOK,
    ReviewModelOK), each with columns [timestamp, outcome (0/1), stratum]."""
    memory = _load_memory(memory_path)
    return {
        "DataOK": mine_dataok_labels(memory),
        "ModelOK": mine_modelok_labels(memory),
        "ExplainOK": mine_explainok_labels(memory),
        "ReviewDataOK": mine_reviewdataok_labels(expert_labels_path),
        "ReviewModelOK": mine_reviewmodelok_labels(expert_labels_path),
    }


def empirical_profile(df):
    """Stratum weights observed in the mined sample itself -- the default operational
    profile until an actuarial/product-specified real-world traffic mix exists
    (see docs/uncertainty_operational_profile.md, "Operational profile weights")."""
    if df.empty:
        return {}
    counts = df["stratum"].value_counts(normalize=True)
    return counts.to_dict()


def to_hip_inputs(df):
    """(outcomes, strata, profile) ready to pass into
    HIPLLM.OperationalFailureProb(...).fit(outcomes=..., strata=...)."""
    outcomes = df["outcome"].astype(int).tolist()
    strata = df["stratum"].astype(str).tolist()
    profile = empirical_profile(df)
    return outcomes, strata, profile


def main():
    tables = mine_labels()
    for node, df in tables.items():
        print(f"\n=== {node} ===")
        if df.empty:
            print("No labeled rows found.")
            continue
        print(f"rows: {len(df)}")
        print(df.groupby("stratum")["outcome"].agg(["count", "mean"]).round(3).to_string())
        print(f"empirical profile: {empirical_profile(df)}")


if __name__ == "__main__":
    main()
