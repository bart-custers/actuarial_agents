import os
import joblib
import pandas as pd
from datetime import datetime
from agents.base_agent import BaseAgent
from agents.data_pipeline import DataPipeline
from utils.message_types import Message
from llms.wrappers import LLMWrapper

class DataPrepAgent(BaseAgent):
    def __init__(self, name="dataprep", llm_backend="llama7b", system_prompt=None):
        super().__init__(name)
        self.llm = LLMWrapper(backend=llm_backend, system_prompt=system_prompt)

    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting deterministic data pipeline...")

        dataset_path = message.metadata.get("dataset_path", "data/raw/freMTPL2freq.csv")
        processed_dir = "data/processed"
        artifacts_dir = "data/artifacts"
        logs_dir = "data/logs"

        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # --- Load dataset ---
        try:
            data = pd.read_csv(dataset_path)
        except Exception as e:
            return Message(
                sender=self.name,
                recipient=message.sender,
                type="error",
                content=f"Failed to load dataset: {e}",
            )

        # --- Run deterministic cleaning ---
        pipeline = DataPipeline()
        results = pipeline.clean(data)
        summary_text = pipeline.summary()

        # --- Save processed datasets ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]

        X_train_path = os.path.join(processed_dir, f"{base_name}_X_train_{timestamp}.csv")
        X_test_path = os.path.join(processed_dir, f"{base_name}_X_test_{timestamp}.csv")
        y_train_path = os.path.join(processed_dir, f"{base_name}_y_train_{timestamp}.csv")
        y_test_path = os.path.join(processed_dir, f"{base_name}_y_test_{timestamp}.csv")

        # Save the split sets
        pd.DataFrame(results["X_train"], columns=results["feature_names"]).to_csv(X_train_path, index=False)
        pd.DataFrame(results["X_test"], columns=results["feature_names"]).to_csv(X_test_path, index=False)
        results["y_train"].to_csv(y_train_path, index=False)
        results["y_test"].to_csv(y_test_path, index=False)

        # --- Save preprocessing artifacts ---
        preproc_path = os.path.join(artifacts_dir, f"preprocessor_{timestamp}.pkl")
        features_path = os.path.join(artifacts_dir, f"feature_names_{timestamp}.pkl")

        joblib.dump(results["feature_names"], features_path)
        joblib.dump(pipeline.preprocessor, preproc_path)

        # --- Log summary ---
        log_path = os.path.join(logs_dir, f"dataprep_summary_{timestamp}.txt")
        with open(log_path, "w") as f:
            f.write("=== DATA PREPARATION SUMMARY ===\n")
            f.write(summary_text + "\n\n")
            f.write(f"Processed data saved to: {processed_dir}\n")
            f.write(f"Artifacts saved to: {artifacts_dir}\n")

        # --- LLM-generated explanation ---
        explain_prompt = f"""
        You are an AI assistant summarizing a data preprocessing pipeline for an actuarial audience.
        Write a clear explanation of the following cleaning steps and why they are important:
        {summary_text}
        """
        explanation = self.llm(explain_prompt)

        # --- Return structured output message ---
        metadata = {
            "status": "success",
            "summary_log": log_path,
            "llm_explanation": explanation,
            "processed_paths": {
                "X_train": X_train_path,
                "X_test": X_test_path,
                "y_train": y_train_path,
                "y_test": y_test_path,
            },
            "artifacts": {
                "preprocessor": preproc_path,
                "feature_names": features_path,
            },
        }

        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content="Data cleaning and preprocessing completed successfully.",
            metadata=metadata,
        )

# import os
# import json
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from .base_agent import LLMBaseAgent

# class DataPrepAgent(LLMBaseAgent):
#     """
#     Combined deterministic + LLM-reflective data preparation agent.
#     """

#     def receive(self, content, sender):
#         task = content.get("task", "")
#         if task == "clean_and_refine":
#             data_path = content.get("data_path", "data/raw/freMTPL2freq.csv")
#             return self.clean_and_refine(data_path)
#         else:
#             return {"error": f"Unknown task: {task}"}

#     # ------------------------------------------------------------
#     # Step 1: Deterministic baseline cleaning (from your provided code)
#     # ------------------------------------------------------------
#     def deterministic_clean(self, data_path):
#         print(f"Attempting to load: {data_path}")
#         df = pd.read_csv(data_path)
#         print(f"{self.name}: Loaded dataset with shape {df.shape}")

#         # --- Basic cleaning ---
#         df = df.dropna(subset=["ClaimNb", "Exposure"])
#         df = df[df["Exposure"] > 0]

#         # --- Feature selection ---
#         features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
#         target = "ClaimNb"

#         X = df[features]
#         y = df[target]

#         # --- Split ---
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # --- Save ---
#         os.makedirs("data/processed", exist_ok=True)
#         X_train_path = "data/processed/X_train.csv"
#         X_test_path = "data/processed/X_test.csv"
#         y_train_path = "data/processed/y_train.csv"
#         y_test_path = "data/processed/y_test.csv"

#         X_train.to_csv(X_train_path, index=False)
#         X_test.to_csv(X_test_path, index=False)
#         y_train.to_csv(y_train_path, index=False)
#         y_test.to_csv(y_test_path, index=False)

#         # --- Preprocessing pipeline ---
#         numeric_features = ["VehPower", "VehAge", "DrivAge", "Density"]
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", Pipeline([("scaler", StandardScaler())]), numeric_features)
#             ],
#             remainder="drop"
#         )

#         # --- Store metadata ---
#         data_bundle = {
#             "preprocessor": preprocessor,
#             "train_path": X_train_path,
#             "test_path": X_test_path,
#             "y_train_path": y_train_path,
#             "y_test_path": y_test_path
#         }

#         self.hub.update_memory("cleaned_data", data_bundle)
#         self.hub.update_memory("dataprep_summary", {
#             "rows_before": len(df),
#             "rows_after": len(X_train) + len(X_test),
#             "features_used": features,
#             "target": target,
#             "comment": "Baseline preprocessing applied."
#         })

#         return data_bundle, df

#     # ------------------------------------------------------------
#     # Step 2: LLM reflection + refinement
#     # ------------------------------------------------------------
#     def clean_and_refine(self, data_path):
#         # Run deterministic cleaning first
#         data_bundle, df = self.deterministic_clean(data_path)

#         # Build reasoning prompt
#         prompt = self.create_prompt(data_bundle, df)

#         # Get reflection/refinement suggestions from LLM
#         llm_output = self.llm(prompt)
#         try:
#             result = json.loads(llm_output)
#         except json.JSONDecodeError:
#             result = {"raw_response": llm_output, "error": "Invalid JSON from LLM"}

#         # Store results
#         self.log_memory(result)
#         self.hub.update_memory("dataprep_refinement", result)
#         return result

#     # ------------------------------------------------------------
#     # Step 3: Build LLM reasoning prompt
#     # ------------------------------------------------------------
#     def create_prompt(self, data_bundle, df):
#         n_rows, n_cols = df.shape
#         summary = self.hub.memory.get("dataprep_summary", {})
#         pipeline_description = f"""
#         # Baseline deterministic pipeline:
#         1. Drop NA rows in ClaimNb and Exposure.
#         2. Filter out rows with Exposure <= 0.
#         3. Select features {summary.get("features_used", [])}.
#         4. Split data into train/test (80/20).
#         5. Standardize numeric features {['VehPower', 'VehAge', 'DrivAge', 'Density']}.
#         """

#         prompt = f"""
#         You are an expert data scientist for motor insurance pricing.

#         Below is the summary of a baseline preprocessing pipeline and dataset information.
#         Dataset: {n_rows} rows × {n_cols} columns.

#         {pipeline_description}

#         Your tasks:
#         1. Identify potential issues or limitations with this preprocessing.
#         2. Suggest concrete improvements (e.g. transformations, encoding, feature engineering).
#         3. Output a JSON object with:
#         {{
#             "issues_found": [...],
#             "suggested_changes": [...],
#             "revised_pipeline_pseudocode": "..."
#         }}

#         Return **only valid JSON**.
#         """
#         return prompt

# from .base_agent import BaseAgent
# from .base_agent import LLMBaseAgent
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import os

# class DataPrepAgent(BaseAgent):
#     def receive(self, content, sender):
#         task = content.get("task", "")
#         if task == "clean_data":
#             data_path = content.get("data_path", "data/raw/freMTPL2freq.csv")
#             print(f"Attempting to load: {data_path}")
#             df = pd.read_csv(data_path)
#             print(f"{self.name}: Loaded dataset with shape {df.shape}")

#             # --- Basic cleaning ---
#             df = df.dropna(subset=["ClaimNb", "Exposure"])
#             df = df[df["Exposure"] > 0]

#             # --- Feature selection ---
#             features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
#             target = "ClaimNb"

#             X = df[features]
#             y = df[target]

#             # --- Split into train/test ---
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )

#             # --- Save to processed folder ---
#             os.makedirs("data/processed", exist_ok=True)
#             X_train.to_csv("data/processed/X_train.csv", index=False)
#             X_test.to_csv("data/processed/X_test.csv", index=False)
#             y_train.to_csv("data/processed/y_train.csv", index=False)
#             y_test.to_csv("data/processed/y_test.csv", index=False)

#             # --- Define preprocessing pipeline ---
#             numeric_features = ["VehPower", "VehAge", "DrivAge", "Density"]
#             preprocessor = ColumnTransformer(
#                 transformers=[("num", Pipeline([("scaler", StandardScaler())]), numeric_features)],
#                 remainder="drop"
#             )

#             # --- Store metadata in memory ---
#             data_bundle = {
#                 "preprocessor": preprocessor,
#                 "train_path": "data/processed/X_train.csv",
#                 "test_path": "data/processed/X_test.csv"
#             }

#             self.hub.update_memory("cleaned_data", data_bundle)

#             return {"status": "success", "message": "Data cleaned and saved to processed/"}
#         else:
#             return {"status": "unknown_task"}

# class DataPrepAgent(LLMBaseAgent):
#     def create_prompt(self, message):
#         return f"""
#         You are a Data Prep Agent.
#         Task: {message['task']}
#         Data Path: {message.get('data_path', '')}
#         Respond with a JSON containing:
#         {{
#             "cleaned_data_summary": "...",
#             "confidence": 0.0
#         }}
#         Only respond in JSON.
#         """