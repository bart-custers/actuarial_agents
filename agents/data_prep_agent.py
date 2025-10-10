import os
import joblib
import pandas as pd
from datetime import datetime
from agents.base_agent import BaseAgent
from agents.data_pipeline import DataPipeline
from utils.message_types import Message
from llms.wrappers import LLMWrapper

class DataPrepAgent(BaseAgent):
    def __init__(self, name="dataprep", shared_llm=None, system_prompt=None):
        super().__init__(name)
        # Use shared LLM if provided, else create own
        self.llm = shared_llm or LLMWrapper(backend="mock", system_prompt=system_prompt)
        self.system_prompt = system_prompt

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
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]

        X_train_path = os.path.join(processed_dir, f"{base_name}_X_train.csv")
        X_test_path = os.path.join(processed_dir, f"{base_name}_X_test.csv")
        y_train_path = os.path.join(processed_dir, f"{base_name}_y_train.csv")
        y_test_path = os.path.join(processed_dir, f"{base_name}_y_test.csv")

        # Save the split sets
        pd.DataFrame(results["X_train"], columns=results["feature_names"]).to_csv(X_train_path, index=False)
        pd.DataFrame(results["X_test"], columns=results["feature_names"]).to_csv(X_test_path, index=False)
        results["y_train"].to_csv(y_train_path, index=False)
        results["y_test"].to_csv(y_test_path, index=False)

        # --- Save preprocessing artifacts ---
        preproc_path = os.path.join(artifacts_dir, f"preprocessor.pkl")
        features_path = os.path.join(artifacts_dir, f"feature_names.pkl")

        joblib.dump(results["feature_names"], features_path)
        joblib.dump(pipeline.preprocessor, preproc_path)

        # --- Log summary ---
        log_path = os.path.join(logs_dir, f"dataprep_summary.txt")
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