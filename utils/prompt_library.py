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

    Think step-by-step.
    1. Was the model confident about the adaptive pipeline?
    2. How does the adaptive pipeline compare to the deterministic pipeline?
    
    Task: Decide whether the adaptive pipeline should be used. Justify in a short bullet list
    
    The final line of your answer should contain: Decision: USE_ADAPTIVE or KEEP_BASELINE.
    """,

    "dataprep_layer4": """
    Summarize the verified data preparation and reasoning.
    Think step-by-step.
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

    Respond concisely. The final line of your answer should contain: Decision: USE_GLM or USE_GBM.
    """,

    "modelling_layer2":""" 
    You are an expert actuarial modelling assistant helping to build claim frequency models.

    Your task is to produce ONLY Python code that trains the model type chosen earlier: {model_choice}

    YOU MUST OUTPUT *ONLY ONE* PYTHON CODE BLOCK AND NOTHING ELSE.
    - NO explanations. 
    - NO text before or after. 
    - NO commentary. 
    - NO markdown except ```python.
    - If you produce anything outside the code block, the system will CRASH.

    Your code will be executed with the following variables already defined:
    - X_train : pandas DataFrame
    - y_train : numpy array
    - exposure_train : numpy array
    - X_test  : pandas DataFrame

    You MUST produce:
    - `model` : the trained model object
    - `preds` : predictions for X_test as a 1D numpy array

    The FINAL line of your code must be:

    result = {{"model": model, "preds": preds}}

    Here is the template you must complete:

    ```python
    # ----- YOU MUST ONLY FILL IN THE TWO FUNCTIONS BELOW -----
    import # Import necessary libraries

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

    model = train_model(X_train, y_train, exposure_train)
    preds = generate_predictions(model, X_test)

    result = {{"model": model, "preds": preds}}
    ```
    """
    ,

    "modelling_layer3": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Your task is to review the model performance.

    The model ({model_type}) was trained for claim frequency prediction. Use the following context:

    - Evaluation results: {metrics}
    - Comparison of actual and predicted values: {act_vs_exp}

    Think step-by-step and provide a concise summary that includes:
    1. Summarize the model’s goodness-of-fit and calibration quality. 
    2. Highlight whether the model seems overfitted or underfitted.
    3. Mention which variables appear most influential and why.
    4. Summarize the comparison of actual vs predicted values over features.

    Output a section called: ANALYSIS: <your reasoning here>, be concise and use 300 words maximum in bullet points.
    """,

    "modelling_layer4": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Your task is to review the model performance.

    Think step-by-step and complete the following task:
    For the trained model, compare the predictions to the predictions of the previous model. 
    Use the table provided in: {impact_analysis_tables}

    Output a section called: ANALYSIS: <your reasoning here>, be concise and use 300 words maximum in bullet points.
    """,

    # --------------------
    # Reviewing agent prompts
    # --------------------
    "review_layer1": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Your task is to evaluate the {phase} of a frequency prediction model and provide a critical review.

    Possible phases:
    - DataPrepAgent (cleaning / preprocessing)
    - ModellingAgent (model training / predictive performance)

    Think step-by-step and list:
    1) Briefly restate the assignment and your role in one sentence.
    2) List the most relevant actions for evaluation you think are most important for this task.
    
    Respond concisely. Use maximum 100 words in bullet points.
    """,

    "review_layer2_dataprep": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Your task is to evaluate the {phase} of a frequency prediction model and provide a critical review.

    Think step-by-step, using the following context: 
    - Summary of your earlier thinking: {layer1_out}
    - The used preprocessing pipeline: {used_pipeline}
    - The confidence score for the preprocessing: {confidence}
    - The verification feedback received: {verification}

    If previous memory of dataprep, modelling and reviews exist, ensure consistency with them.
    Historical memory summary:
    {review_memory}

    Your task:
    - Evaluate plausibility.
    - Identify data/model quality issues.
    - Output a section called: ANALYSIS: <your reasoning here>, be concise and use 200 words maximum in bullet points.
    """,

    "review_layer2_modelling": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Your task is to evaluate the {phase} of a frequency prediction model and provide a critical review.

    Think step-by-step, using the following context: 
    - Summary of your earlier thinking: {layer1_out}
    - The model type used: {model_type_used}
    - The model evaluation from the modelling agent: {evaluation}

    If previous memory of dataprep, modelling and reviews exist, ensure consistency with them.
    Historical memory summary:
    {review_memory}

    Your task:
    - Evaluate plausibility.
    - Identify model training issues.
    - Output a section called: ANALYSIS: <your reasoning here>, be concise and use 200 words maximum in bullet points.
    """,

    "review_layer3": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    In addition to the previous analyses, now assess the consistency of the outcome of phase: {phase}. 
    
    A summary on the consistency is already provided in: {consistency_summary}.
    
    - Output a section called: ANALYSIS: <your reasoning here>, be concise and use 200 words maximum in bullet points. Think step-by-step. Be concise.
    """,

    "review_layer4": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    In addition to the previous analyses, now assess the impact analysis that compares the current predictions to the previous predictions. 
    
    A summary on the impact analysis is already provided in: {impact_analysis_input}.
    
    - Output a section called: ANALYSIS: <your reasoning here>, be concise and use 200 words maximum in bullet points. Think step-by-step. Be concise.
    """,

    "review_layer5": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims. 
    Based on the analysis: {analysis} and {consistency_check} and {impact_analysis_output}, choose the correct next action. Think step-by-step.

    Valid actions:
    - APPROVE → proceed to next agent
    - REQUEST_RECLEAN → redo data cleaning
    - REQUEST_RETRAIN → redo model training
    - ABORT → stop workflow entirely

    Do not provide explanations. The final line of your answer should contain: Decision: APPROVE or REQUEST_RECLEAN or REQUEST_RETRAIN or ABORT.
    """,

    "review_layer6": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims.

    Based on the whole review process you just performed, create a short summary report of max 500 words.
    Output a section called: REPORT: <your reasoning here>, be concise and use 500 words max.

    """,

    "review_revision": """
    You are an expert in actuarial modelling, assisting in claim frequency prediction for insurance claims.

    Your task is to improve the following prompt so that the agent performs better in the next iteration.

    Context:
    - Phase under review: {phase}
    - Reviewer detected issues: {analysis}
    - Reviewer decision: {decision}

    Here is the ORIGINAL prompt used by the agent: 
    <<< ORIGINAL_PROMPT >>>
    {base_prompt}
    <<< END >>>

    Rewrite this prompt to address the issues above.

    Guidelines:
    - Keep the structure of the original prompt.
    - Highlight specific improvements needed.
    - Do NOT change the agent identity or role.
    - Do NOT remove required output fields.

    Return only the revised prompt. No explanations.

    """,

    # --------------------
    # Explanation agent prompts
    # --------------------
    "summary_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.

    Your goal is to:
    1. Extract the main ideas from each item.
    2. Summarize the belief of the agent.

    Items:
    Item 1: {item1}
    Item 2: {item2}
    Item 3: {item3}

    Summarize the belief of the agent in 200 words.
    """,

    "belief_revision_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.
    Your goal is to assess the clarity, validity, confidence level, and potential issues in the beliefs.

    You are given a belief summary for data preparation, modelling, and reviewing:

    [BELIEF SUMMARY]
    {belief_summary}

    Your tasks:

    1. **Identify Beliefs**
    Review all explicit or implicit beliefs expressed in the summary.

    2. **Assess Validity & Stability**
    - Determine whether the beliefs are well-supported.
    - Highlight any inconsistencies or unclear reasoning.
    - Flag any beliefs that require verification or caution.
    - Check whether beliefs are conflicting with good actuarial practice.

    3. **Indicate needed actions**
    Indicate whether the workflow can be approved, or whether additional actions are needed.

    - Be concise and neutral.
    - Do not speculate beyond the provided summary.
    - Maintain high precision and avoid hallucination.
    - Use maximum 400 words.

    At the end of your analysis, classify whether there are belief contradictions. The final line of your answer should contain: Decision: NONE or MINOR or SEVERE.
    """,

    "tcav_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.
    Your goal is to assess the fairness of the model predictions.

    <PLACEHOLDER>

    At the end of your analysis, classify whether there are fairness biases. The final line of your answer should contain: Decision: NONE or MINOR or SEVERE.
    """,

    "fairness_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.
    Your goal is to assess the fairness of the model predictions.

    You are given two tables that show Mean difference (actual vs predicted) for various groups, over premium bins. 
    Values far from zero indicate miscalibration. Systematic patterns across bins indicate structural bias.

    Age fairness table: {table_age}
    Population density fairness table: {table_density}

    Your tasks:

    1. Analyse the tables, think step-by-step.

    2. Write a short report with your critical analysis. 
    
    - Be concise and neutral.
    - Do not speculate beyond the provided summary.
    - Maintain high precision and avoid hallucination.
    - Use maximum 400 words.

    At the end of your analysis, classify whether there are fairness biases. The final line of your answer should contain: Decision: NONE or MINOR or SEVERE.
    """,

    "decision_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.
    
    Your tasks: Think step-by-step. Based on the analysis: {belief_assessment} and {tcav_assessment} and {fairness_assessment}, choose the correct next action. Think step-by-step.

    Valid actions:
    - APPROVE → the workflow can be finalized
    - MINOR_ISSUES → minor issues detected, the workflow can be finalized but an actuary should be consulted
    - REQUEST_RECLEAN → redo data cleaning
    - REQUEST_RETRAIN → redo model training
    - ABORT → stop workflow entirely

    Requested output:
    - Output a section called: ANALYSIS: <your reasoning here>, be concise and use 400 words maximum in bullet points. Think step-by-step.
    - The final line of your answer should contain: Decision: APPROVE or MINOR_ISSUES or REQUEST_RECLEAN or REQUEST_RETRAIN or ABORT.
    """,

    "recommendation_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.
    
    Your tasks: 
    Based on the final evaluation: {final_evaluation}, you proposed the decision to: {decision}. Provide recommendations to the agents for the next iteration. 

    Requested output:
    - Output a section called: ANALYIS: <your recommendation here>, be concise and use 400 words maximum in bullet points. Think step-by-step.
    """,

    "report_prompt": """
    You are an expert in actuarial modelling, assisting in explaining an agent workflow for claim frequency prediction.
    
    Your tasks: 
    Think step-by-step. Based on the final evaluation: {final_evaluation}, you proposed the decision to: {decision}, create a final explanation report.

    Requested output:
    - Output a section called: ANALYIS: <your report summary here>, be concise and use 500 words maximum in bullet points. 
    """
}