import json
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
