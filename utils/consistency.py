import numpy as np
import pandas as pd
import hashlib
from typing import List, Dict, Any

# Create consistency snapshots
def dataprep_consistency_snapshot(df: pd.DataFrame, target: str = None) -> Dict[str, Any]:
    """Create a dataset 'consistency snapshot' capturing structure, distributions,
    missingness, variable types, etc., for long-term drift monitoring.
    """
    snapshot = {}

    # Basic items
    snapshot["n_rows"] = int(df.shape[0])
    snapshot["n_columns"] = int(df.shape[1])
    snapshot["memory_MB"] = round(df.memory_usage(deep=True).sum() / 1e6, 4)

    # Variables
    snapshot["dtypes"] = df.dtypes.astype(str).to_dict()

    # Missing values
    snapshot["missing_counts"] = df.isna().sum().astype(int).to_dict()
    snapshot["missing_pct"] = df.isna().mean().round(6).to_dict()

    # Distribution of numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        snapshot["numeric_summary"] = (
            df[numeric_cols].describe().round(4).to_dict()
        )
    else:
        snapshot["numeric_summary"] = {}

    # Distribution of categorical variables
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    snapshot["categorical_summary"] = {
        col: df[col].value_counts(normalize=True).round(4).to_dict()
        for col in cat_cols
    }

    # Target variable distribution
    if target and target in df.columns:
        snapshot["target_distribution"] = (
            df[target].value_counts(normalize=True).round(4).to_dict()
        )
        snapshot["target_mean"] = float(df[target].mean())
        snapshot["target_variance"] = float(df[target].var())

    return snapshot


# Compare snapshots
def compare_dataprep_consistency_snapshots(current: Dict[str, Any],history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compares current snapshot to all historical snapshots."""
    if not history:
        return {
            "status": "no_history",
            "summary": "No historical dataprep snapshots available."
        }

    prev = history[-1]  # most recent snapshot

    def collect_hist(key):
        return [snapshot[key] for snapshot in history if key in snapshot]

    comparison = {}

    # Row/column count changes
    comparison["row_count_change"] = current["n_rows"] - prev["n_rows"]
    comparison["col_count_change"] = current["n_columns"] - prev["n_columns"]

    # Row count trends over history
    all_rows = collect_hist("n_rows")
    if len(all_rows) > 1:
        comparison["row_count_trend"] = {"mean": float(np.mean(all_rows))}

    # Missing value drift
    missing_drift = {}
    for col, curr_pct in current["missing_pct"].items():
        prev_pcts = [snap["missing_pct"].get(col, 0) for snap in history]
        avg_prev = np.mean(prev_pcts)
        diff = curr_pct - avg_prev
        if abs(diff) > 0.05:
            missing_drift[col] = round(diff, 4)

    comparison["missing_drift"] = missing_drift

    # Type changes
    comparison["type_changes"] = {
        col: (prev["dtypes"].get(col), curr_dtype)
        for col, curr_dtype in current["dtypes"].items()
        if prev["dtypes"].get(col) != curr_dtype
    }

    # Target variable check
    if "target_distribution" in current and "target_distribution" in prev:
        target_drift = {}
        historical = [
            snapshot["target_distribution"]
            for snapshot in history
            if "target_distribution" in snapshot
        ]

        class_means = {}
        for snapshot in historical:
            for cls, pct in snapshot.items():
                class_means.setdefault(cls, []).append(pct)

        class_means = {cls: np.mean(vals) for cls, vals in class_means.items()}

        for cls, curr_val in current["target_distribution"].items():
            avg_prev = class_means.get(cls, 0)
            diff = curr_val - avg_prev
            if abs(diff) > 0.05:
                target_drift[cls] = round(diff, 4)

        comparison["target_distribution_drift"] = target_drift

    return comparison


# Summarize comparison for LLM input
def summarize_dataprep_snapshot_comparison(comp: Dict[str, Any]) -> str:
    """Turn comparison dictionary into a human-readable LLM-friendly summary."""
    if comp.get("status") == "no_history":
        return "No historical dataprep snapshots exist."

    lines = ["### Dataprep Consistency Summary"]

    if comp["row_count_change"] != 0:
        lines.append(f"- Row count changed by {comp['row_count_change']} since previous dataprep.")

    if "row_count_trend" in comp:
        t = comp["row_count_trend"]
        lines.append(
            f"- Historical row counts ranged {t['min']}–{t['max']} (mean {t['mean']:.1f}, std {t['std']:.1f})."
        )

    if comp["missing_drift"]:
        lines.append("- Missing value drift detected vs historical mean:")
        for col, diff in comp["missing_drift"].items():
            lines.append(f"  • {col}: {diff:+.3f}")

    if comp["type_changes"]:
        lines.append("- Variable type changes:")
        for col, (prev, curr) in comp["type_changes"].items():
            lines.append(f"  • {col}: {prev} → {curr}")

    if comp["column_order_changed"]:
        lines.append("- Column order changed (structural drift).")

    if comp.get("target_distribution_drift"):
        if comp["target_distribution_drift"]:
            lines.append("- Target distribution drift:")
            for cls, diff in comp["target_distribution_drift"].items():
                lines.append(f"  • Class {cls}: {diff:+.3f}")

    if len(lines) == 1:
        lines.append("No meaningful inconsistencies detected.")

    return "\n".join(lines)
