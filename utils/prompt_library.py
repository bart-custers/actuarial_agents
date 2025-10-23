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
    Evaluate the following model results and provide an explicit decision line

    At the end of your response, include a line exactly in this format:
    Status: <APPROVED | NEEDS_REVISION | RETRAIN_REQUESTED>

    Metrics: {metrics}
    Numeric severity: {severity}
    Review notes: {review_notes}

    If previous memory of dataprep, modelling and reviews exist, ensure consistency with them.
    Historical memory summary:
    {memory_for_prompt}

    Consistency check for model coefficients:
    {consistency_info}

    Provide:
    1. One line starting with "Status:" (e.g., "Status: APPROVED")
    2. A short professional justification.
    """,

    "consistency_prompt": """
    You are an actuarial explanation specialist.

    Compare current model results to the previous run stored in memory.

    Current Metrics:
    {model_metrics}

    Current Review Notes:
    {review_notes}

    Previous Review Outcome:
    Status: {last_review.get('status', 'N/A')}
    Notes: {last_review.get('review_notes', [])}

    Identify:
    - Whether the model is consistent with prior iterations (metrics, direction of coefficients)
    - Any drift or unexplained changes
    - Any contradictory findings or explanations.

    Provide a concise stability summary in plain English.
    """,

    "belief_revision_prompt": """
    You are a reasoning assistant performing belief revision for model interpretation.
    Given the current explanation, past review notes, and current metrics,
    update the overall understanding of the model performance and rationale.

    Ensure that your belief update resolves contradictions and forms a coherent explanation.

    Use this structure:
    1. Consistent beliefs (what remains stable)
    2. Revised beliefs (what changed and why)
    3. Remaining uncertainties

    Current LLM review:
    {llm_review}

    Review notes:
    {review_notes}

    Last Review Notes:
    {last_review_for_prompt}
    """
}