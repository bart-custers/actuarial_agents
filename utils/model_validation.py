def evaluate_model_quality(metrics):
    """
    Evaluate basic model quality based on numeric metrics.
    Returns: "minor", "moderate", or "critical"
    """
    rmse = metrics.get("RMSE", 0)
    dev = metrics.get("Poisson Deviance", 0)
    mae = metrics.get("MAE", 0)

    # Conservative thresholds for actuarial frequency models
    if rmse > 0.3 or dev > 0.3 or mae > 0.8:
        return "critical"
    elif rmse > 0.2 or dev > 0.2 or mae > 0.5:
        return "moderate"
    else:
        return "minor"