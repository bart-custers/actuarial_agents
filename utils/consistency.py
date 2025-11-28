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


# Compare dataprep snapshots
def compare_dataprep_consistency_snapshots(current: Dict[str, Any],history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compares current snapshot to all historical snapshots."""
    if not history:
        return {
            "status": "no_history",
            "summary": "No historical dataprep snapshots available."
        }

    comparison = {}
    prev = history[-1]  # most recent snapshot

    def collect_hist(key):
        return [snapshot[key] for snapshot in history if key in snapshot]

    # Row/column count changes
    comparison["row_count_change"] = current["n_rows"] - prev["n_rows"]
    comparison["col_count_change"] = current["n_columns"] - prev["n_columns"]

    # Row count trends over history
    all_rows = collect_hist("n_rows")
    if len(all_rows) > 1:
        comparison["row_count_trend"] = {
            "min": float(np.min(all_rows)),
            "max": float(np.max(all_rows)),
            "mean": float(np.mean(all_rows)),
            "std": float(np.std(all_rows))
        }

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

    if comp.get("target_distribution_drift"):
        if comp["target_distribution_drift"]:
            lines.append("- Target distribution drift:")
            for cls, diff in comp["target_distribution_drift"].items():
                lines.append(f"  • Class {cls}: {diff:+.3f}")

    if len(lines) == 1:
        lines.append("No meaningful inconsistencies detected.")

    return "\n".join(lines)


# Compare modelling snapshots
def compare_modelling_consistency_snapshots(current: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare current model metrics snapshot to historical records.
    Returns a structured drift analysis.
    """
    if not history:
        return {
            "status": "no_history",
            "summary": "No historical model metric snapshots available."
        }

    comparison = {}
    prev = history[-1]

    # Metric analysis
    scalar_metrics = ["RMSE", "MAE", "R2", "Poisson_Deviance", "Gini_score"]
    scalar_drift = {}

    for m in scalar_metrics:
        curr_val = current.get(m)
        hist_vals = [snap.get(m) for snap in history if snap.get(m) is not None]

        if curr_val is None or len(hist_vals) < 1:
            continue

        mean_prev = np.mean(hist_vals)
        diff = curr_val - mean_prev

        if mean_prev != 0:
            pct_diff = diff / mean_prev
        else:
            pct_diff = 0

        if abs(pct_diff) > 0.05:
            scalar_drift[m] = {
                "current": curr_val,
                "avg_history": float(mean_prev),
                "pct_change": round(pct_diff, 4)
            }

    comparison["scalar_metric_drift"] = scalar_drift

    # Feature analysis
    curr_imp = current.get("Feature_Importance", {})
    prev_imp = prev.get("Feature_Importance", {})

    if isinstance(curr_imp, list):
        curr_imp = dict(curr_imp)
    if isinstance(prev_imp, list):
        prev_imp = dict(prev_imp)

    feature_drift = {}

    # Features that appear/disappear
    added = set(curr_imp.keys()) - set(prev_imp.keys())
    removed = set(prev_imp.keys()) - set(curr_imp.keys())

    if added:
        feature_drift["new_features"] = list(added)
    if removed:
        feature_drift["removed_features"] = list(removed)

    comparison["feature_importance_drift"] = feature_drift

    return comparison


# Summarize comparison for LLM input
def summarize_modelling_snapshot_comparison(comp: Dict[str, Any]) -> str:
    if comp.get("status") == "no_history":
        return "No historical model metric snapshots exist."

    lines = ["### Model Metrics Consistency Summary"]

    # scalar drifts
    if comp["scalar_metric_drift"]:
        lines.append("- Analysis of scalar metrics:")
        for m, d in comp["scalar_metric_drift"].items():
            lines.append(
                f"  • {m}: {d['current']:.4f} vs hist avg {d['avg_history']:.4f} "
                f"(change {d['pct_change']:+.3f})"
            )

    # feature importance drift
    feat = comp["feature_importance_drift"]

    if "new_features" in feat:
        lines.append(f"- New features introduced: {', '.join(feat['new_features'])}")

    if "removed_features" in feat:
        lines.append(f"- Features removed since last model: {', '.join(feat['removed_features'])}")

    if "importance_changes" in feat:
        lines.append("- Feature importance magnitude changes:")
        for f, info in feat["importance_changes"].items():
            lines.append(
                f"  • {f}: {info['previous']:.3f} → {info['current']:.3f} "
                f"(Δ {info['change']:+.4f})"
            )

    return "\n".join(lines)
