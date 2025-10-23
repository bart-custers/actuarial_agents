PROMPTS = {
    "data_prep": """
    You are an AI assistant summarizing a data preprocessing pipeline for an actuarial audience.
    Write a clear explanation of the following cleaning steps and why they are important:
    {summary_text}
    """,

    "modelling": """
    You are an actuarial data scientist reviewing model results.
    Explain these evaluation metrics for a GLM claim frequency model in plain terms:

    {metrics}

    Highlight whether model fit is reasonable, any bias patterns, and next steps for improvement.
    """,

    "review_model": """
    You are an actuarial model reviewer.

    Metrics:
    {metrics}

    Numeric severity: {severity}
    Review notes: {review_notes}

    Historical context:
    {memory_context}

    Reproducibility drift (if available):
    {coef_drift}

    At the end of your response, include:
    Status: <APPROVED | NEEDS_REVISION | RETRAIN_REQUESTED>
    """,

    "explain_model": """
    You are an explanation specialist.

    The model performance metrics are:
    {metrics}

    Review notes from the auditor:
    {review_notes}

    Based on historical logs:
    {memory_context}

    Provide:
    1. Stability and consistency analysis
    2. Belief revision summary
    """
}