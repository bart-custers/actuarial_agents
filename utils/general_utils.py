import json
import re
import numpy as np
import pandas as pd

def make_json_compatible(obj):
    """Recursively convert objects to JSON-serializable formats."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, (np.generic, np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_compatible(v) for v in obj]
    else:
        return obj

def save_json_safe(data, path):
    """Convenience wrapper that saves a JSON-safe version of any Python object."""
    safe_data = make_json_compatible(data)
    with open(path, "w") as f:
        json.dump(safe_data, f, indent=2)
    return path

def generate_review_report_txt(report_path, 
                               phase, 
                               model_metrics, 
                               analysis, 
                               consistency_summary, 
                               consistency_check,
                               impact_analysis_output,
                               review_output,
                               final_report):
    """
    Creates a txt report combining all review agent outputs.
    """

    lines = []
    lines.append("==== MODEL REVIEW REPORT ====")
    lines.append(f"Phase: {phase}")
    lines.append("")
    lines.append("=== ANALYSIS OF RESULTS ===")
    if model_metrics is None:
        lines.append("No model metrics available.")
    else:
        for k, v in model_metrics.items():
            if k == "Feature_Importance":
                continue  # skip large frames in text
            lines.append(f"- {k}: {v}")
    lines.append(analysis)

    lines.append("\n=== CONSISTENCY CHECK ===")
    lines.append(consistency_summary)
    lines.append(consistency_check)

    lines.append("\n=== IMPACT ANALYSIS ===")
    lines.append(impact_analysis_output)

    lines.append("\n=== REVIEW FEEDBACK ===")
    lines.append(review_output)

    lines.append("\n=== FINAL SUMMARY ===")
    lines.append(final_report)

    # Write to disk
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path

def extract_analysis(text: str) -> str:
    """
    Extract the ANALYSIS: section from a larger string.
    Returns the extracted text, or the original text if no match is found.
    """
    match = re.search(
        r"ANALYSIS:(.*?)(?:\n[A-Z]+:|\Z)",
        text,
        flags=re.DOTALL
    )
    return match.group(1).strip() if match else text
