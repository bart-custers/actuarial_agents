import os
import joblib
import pandas as pd
from datetime import datetime
from utils.general_utils import save_json_safe
from utils.prompt_library import PROMPTS
from utils.data_pipeline import DataPipeline
from utils.message_types import Message
from agents.base_agent import BaseAgent

class DataPrepAgent(BaseAgent):
    def __init__(self, name="dataprep", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub

    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting data pipeline...")

        # --- Initiate paths ---
        dataset_path = message.metadata.get("dataset_path", "data/raw/freMTPL2freq.csv")
        processed_dir = "data/processed"
        artifacts_dir = "data/artifacts"

        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

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

        # --- Run data pipeline ---
        pipeline = DataPipeline()
        results = pipeline.clean(data)
        summary_text = pipeline.summary()

        # --- Save processed datasets ---
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]

        X_train_path = os.path.join(processed_dir, f"{base_name}_X_train.csv")
        X_test_path = os.path.join(processed_dir, f"{base_name}_X_test.csv")
        y_train_path = os.path.join(processed_dir, f"{base_name}_y_train.csv")
        y_test_path = os.path.join(processed_dir, f"{base_name}_y_test.csv")
        exposure_train_path = os.path.join(processed_dir, f"{base_name}_exposure_train.csv")
        exposure_test_path = os.path.join(processed_dir, f"{base_name}_exposure_test.csv")

        # Save the split sets
        pd.DataFrame(results["X_train"], columns=results["feature_names"]).to_csv(X_train_path, index=False)
        pd.DataFrame(results["X_test"], columns=results["feature_names"]).to_csv(X_test_path, index=False)
        results["y_train"].to_csv(y_train_path, index=False)
        results["y_test"].to_csv(y_test_path, index=False)
        results["exposure_train"].to_csv(exposure_train_path, index=False)
        results["exposure_test"].to_csv(exposure_test_path, index=False)

        # --- Save preprocessing artifacts ---
        preproc_path = os.path.join(artifacts_dir, f"preprocessor.pkl")
        features_path = os.path.join(artifacts_dir, f"feature_names.pkl")

        joblib.dump(results["feature_names"], features_path)
        joblib.dump(pipeline.preprocessor, preproc_path)

        # --- LLM-generated explanation ---
        # explain_prompt = f"""
        # You are an AI assistant summarizing a data preprocessing pipeline for an actuarial audience.
        # Write a clear explanation of the following cleaning steps and why they are important:
        # {summary_text}
        # """
        data_prep_prompt = PROMPTS["data_prep"].format(
        summary_text=summary_text,
        )

        explanation = self.llm(data_prep_prompt)

        # --- Return message and log output ---
        # Store metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "status": "success",
            "summary_log": summary_text,
            "llm_explanation": explanation,
            "processed_paths": {
                "X_train": X_train_path,
                "X_test": X_test_path,
                "y_train": y_train_path,
                "y_test": y_test_path,
                "exposure_train": exposure_train_path,
                "exposure_test": exposure_test_path,
            },
            "artifacts": {
                "preprocessor": preproc_path,
                "feature_names": features_path,
            },
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "data_preparation", summary_text)
            self.hub.memory.update("last_data_prep", summary_text)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content="Data cleaning and preprocessing completed successfully.",
            metadata=metadata,
        )