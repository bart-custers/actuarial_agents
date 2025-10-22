# agents/modelling_agent.py
import os
import pandas as pd
from utils.general_utils import save_json_safe
from utils.message_types import Message
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

        # --- Initiate paths ---
        processed_paths = message.metadata.get("processed_paths", None)
        artifacts_dir = "data/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

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

        trainer.save(model_path)
        pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(preds_path, index=False)
        
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

        # --- Return message and log output ---
        # Store metadata
        metadata = {
            "status": "success",
            "model_path": model_path,
            "preds_path": preds_path,
            "metrics": metrics,
            "llm_explanation": explanation,
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "model_trained", metadata)
            history = self.hub.memory.get("model_history", [])
            history.append(metadata)
            self.hub.memory.update("model_history", history)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Model ({self.model_type}) trained and evaluated successfully.",
            metadata=metadata,
        )
