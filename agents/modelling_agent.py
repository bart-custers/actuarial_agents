# agents/modelling_agent.py
import os
from datetime import datetime
import pandas as pd
import json
import joblib
from utils.message_types import Message
from utils.general_utils import make_json_compatible
from agents.base_agent import BaseAgent
from utils.model_trainer import ModelTrainer

class ModellingAgent(BaseAgent):
    def __init__(self, name="modelling", shared_llm=None, system_prompt=None, model_type="glm", hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.hub = hub

    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting model training ({self.model_type})...")

        processed_paths = message.metadata.get("processed_paths", None)
        artifacts_dir = "data/artifacts"
        logs_dir = "data/logs"
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        if processed_paths is None:
            return Message(
                sender=self.name,
                recipient=message.sender,
                type="error",
                content="Missing processed dataset paths.",
            )

        # --- Load data ---
        X_train = pd.read_csv(processed_paths["X_train"])
        X_test = pd.read_csv(processed_paths["X_test"])
        y_train = pd.read_csv(processed_paths["y_train"]).values.ravel()
        y_test = pd.read_csv(processed_paths["y_test"]).values.ravel()
        exposure_train = pd.read_csv(processed_paths["exposure_train"]).values.ravel()
        exposure_test = pd.read_csv(processed_paths["exposure_test"]).values.ravel()
        feature_names = pd.read_pickle(os.path.join(artifacts_dir, "feature_names.pkl"))

        # --- Train model ---
        trainer = ModelTrainer(model_type=self.model_type)
        trainer.train(X_train, y_train)
        preds = trainer.predict(X_test)
        metrics = trainer.evaluate(y_test, preds, feature_names, exposure_test)

        # --- Save model and evaluation ---
        model_path = os.path.join(artifacts_dir, f"model_{self.model_type}.pkl")
        preds_path = os.path.join(artifacts_dir, f"predictions.csv")
        log_path = os.path.join(logs_dir, f"model_eval.txt")

        trainer.save(model_path)
        pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(preds_path, index=False)

        with open(log_path, "w") as f:
            f.write("=== MODEL EVALUATION ===\n")
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    f.write(f"{k}: {v:.6f}\n")
                elif isinstance(v, pd.DataFrame):
                    f.write(f"\n{k}:\n{v.to_string(index=False)}\n")
                else:
                    f.write(f"{k}: {v}\n")
        
        plot_path = metrics.get("Calibration Plot", None)
        if plot_path and os.path.exists(plot_path):
            from IPython.display import Image, display
            print("\n--- Calibration / Lift Chart ---")
            display(Image(filename=plot_path))

        # --- LLM Explanation ---
        explain_prompt = f"""
You are an actuarial data scientist reviewing model results.
Explain these evaluation metrics for a GLM claim frequency model in plain terms:

{metrics}

Highlight whether model fit is reasonable, any bias patterns, and next steps for improvement.
"""
        explanation = self.llm(explain_prompt) if self.llm else "No LLM backend available."

        # --- Return message ---
        metadata = {
            "status": "success",
            "model_path": model_path,
            "preds_path": preds_path,
            "log_path": log_path,
            "metrics": metrics,
            "llm_explanation": explanation,
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata.json")

        # Convert DataFrames to string-safe formats
        safe_meta = make_json_compatible(metadata)
        with open(meta_path, "w") as f:
            json.dump(safe_meta, f, indent=2)

        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content=f"Model ({self.model_type}) trained and evaluated successfully.",
            metadata=metadata,
        )
