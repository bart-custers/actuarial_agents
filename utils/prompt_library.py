PROMPTS = {
    # --------------------
    # Data prep agent prompts
    # --------------------
    "dataprep_layer1": """
    You are an expert data preparation agent for actuarial datasets on insurance claims.
    Dataset summary:
    {info_dict}

    Think step-by-step and list:
    1) Briefly restate the assignment in one sentence.
    2) List the top 6 actions you think are most important to prepare this dataset for claim frequency modelling (short bullet list).
    3) Provide any immediate warnings (e.g., very skewed numeric columns, too many missing values).
    Respond concisely.
    """,

    "dataprep_layer2": """
    You are advising on preprocessing transformations for actuarial modeling.
    Given this dataset summary and the existing pipeline code, suggest
    any improvements or additional transformations.

    Context (summary of your earlier recommendations):
    {summary1}

    Dataset summary:
    {info_dict}

    Existing pipeline:
    {pipeline_code}

    You will now propose OPTIONAL adaptive preprocessing code.

    ### Instructions
    - Think step-by-step, using the dataset summary and existing pipeline as context.
    - ONLY output Python code inside a ```python``` code block.
    - No explanations or comments outside the code block.
    - Do NOT import any modules.
    - Do NOT read/write files.
    - Always output a dataframe named `df_out` at the end.
    
    - Wrap the code in triple backticks like this:

    ```python code here```

    At the end of your answer, output:

    CONFIDENCE: <a number between 0 and 1>

    This line MUST be present.
    """,

    "dataprep_layer3": """
    Compare deterministic vs adaptive pipelines:
    {comparison}
    The model gave confidence={confidence}.
    Should the adaptive pipeline be used? Justify clearly and decide: USE_ADAPTIVE or KEEP_BASELINE.
    """,

    "dataprep_layer4": """
    Summarize the verified data preparation and reasoning.
    Include stability, differences, and final rationale.
    Verification feedback: {verification}.
    """,
    # --------------------
    # Modelling agent prompts
    # --------------------
    "modelling_layer1": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims.

    A dataset has been preprocessed and is now ready for model training.
    Here is the dataset description:

    {dataset_desc}

    Your tasks, thinking step-by-step:

    1. Restate the modelling goal in one short sentence.
    2. Confirm that this modelling task is about regression, with claim_count/exposure as target variable.
    3. Propose the most appropriate modelling approach for this problem:
    - Choose **exactly one**: GLM or GBM.
    4. Justify your choice in 3–5 bullet points (actuarial + ML reasoning).
    5. State any risks or pitfalls you anticipate for this model type.

    Respond concisely. Decide: USE_GLM or USE_GBM.
    """,

    "modelling_layer2":""" 
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims.

    ### Instructions (please read them all carefully)
    - You will now propose python code to train this model on the dataset: {model_choice}
    - You MUST fill in the code inside the functions below. Think step-by-step.
    - Do not change the structure. 
    - Do not rename variables. 
    - Do not move the final `result` definition.
    - As context you can use the existing pipeline code for a GLM: {current_model_code}

    Your code will be executed with the following variables already defined:
    - X_train : pandas DataFrame
    - y_train : numpy array
    - X_test  : pandas DataFrame

    You MUST produce:
    - `model` : the trained model object
    - `preds` : predictions for X_test as a 1D numpy array
    - At the end of your message, output: CONFIDENCE: <number between 0 and 1>

    The FINAL line of your code must be:

    result = {{"model": model, "preds": preds}}

    Here is the template you must complete:

    ```python
    # ----- YOU MUST ONLY FILL IN THE TWO FUNCTIONS BELOW -----

    def train_model(X_train, y_train):
        # Train your model here.
        # Must return a trained model object named `model`.

        model = None  # REPLACE THIS BASED ON model_choice
        return model

    def generate_predictions(model, X_test):
        # Must return predictions for X_test as a 1D array named `preds`.

        preds = None  # REPLACE THIS
        return preds

    # ----- DO NOT MODIFY ANYTHING BELOW THIS LINE -----

    model = train_model(X_train, y_train)
    preds = generate_predictions(model, X_test)

    result = {{"model": model, "preds": preds}}```
    """
    # You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims.

    # For this task you proposed to use the following model:
    # {model_choice}

    # You will now propose python code to train this model on the dataset. As context you can use the existing pipeline code for a GLM:
    # {current_model_code}

    # ### Instructions
    # - Think step-by-step, using the model_choice.
    # - ONLY output Python code inside a ```python``` code block.
    # - No explanations or comments outside the code block.
    # - Do NOT read/write files.
    # - You may assume `X_train`, `y_train`, and `X_test` are pandas DataFrames.
    # - The final line of your code must define: result = {{'preds': preds, 'model': model}}.
    # - Where preds are an array-like of predictions on X_test.
    
    # - Wrap the code in triple backticks like this:

    # ```python code here```

    # At the end of your answer, output:

    # CONFIDENCE: <a number between 0 and 1>
    # """
    ,

    "modelling_layer3": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Your task is to review the model performance.

    The following model ({model_type}) was trained for claim frequency prediction. 
    This is the model object: {model_obj}

    Here are the evaluation results:

    {metrics}

    Think step-by-step and provide a concise summary that includes:
    1. Summarize the model’s goodness-of-fit and calibration quality. 
    2. Highlight whether the model seems overfitted or underfitted.
    3. Mention which variables appear most influential and why.
    4. Review the metrics and provide additional evaluation techniques to consider, given the model used.
    """,
    # --------------------
    # Reviewing agent prompts
    # --------------------
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
    # --------------------
    # Explanation agent prompts
    # --------------------
    "consistency_prompt": """
    You are an actuarial explanation specialist.

    Compare current model results to the previous run stored in memory.

    Current Metrics:
    {model_metrics}

    Current Review Notes:
    {review_notes}

    Previous Review Outcome:
    Status: {last_review_status}
    Notes: {last_review_for_prompt}

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