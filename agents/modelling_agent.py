# agents/modelling_agent.py
import os
from datetime import datetime
import pandas as pd
import joblib
from utils.message_types import Message
from agents.base_agent import BaseAgent
from agents.model_trainer import ModelTrainer

class ModellingAgent(BaseAgent):
    def __init__(self, name="modelling", shared_llm=None, system_prompt=None, model_type="glm"):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.model_type = model_type

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
        feature_names = pd.read_pickle(os.path.join(artifacts_dir, "feature_names.pkl"))

        # --- Train model ---
        trainer = ModelTrainer(model_type=self.model_type)
        trainer.train(X_train, y_train)
        preds = trainer.predict(X_test)
        metrics = trainer.evaluate(y_test, preds, feature_names)

        # --- Save model and evaluation ---
        model_path = os.path.join(artifacts_dir, f"model_{self.model_type}.pkl")
        preds_path = os.path.join(artifacts_dir, f"predictions.csv")
        log_path = os.path.join(logs_dir, f"model_eval.txt")

        trainer.save(model_path)
        pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(preds_path, index=False)

        with open(log_path, "w") as f:
            f.write("=== MODEL EVALUATION ===\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.6f}\n")

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

        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content=f"Model ({self.model_type}) trained and evaluated successfully.",
            metadata=metadata,
        )


# from llms.wrappers import LLMWrapper
# from utils.message_types import Message
# from agents.base_agent import BaseAgent

# class ModellingAgent(BaseAgent):
#     def __init__(self, name="modelling", shared_llm=None, system_prompt=None):
#         super().__init__(name)
#         self.llm = shared_llm or LLMWrapper(backend="mock", system_prompt=system_prompt)
#         self.system_prompt = system_prompt

#     def handle_message(self, message: Message) -> Message:
#         print(f"[{self.name}] received: {message.content}")
#         # For now: placeholder until Week 3
#         return Message(
#             sender=self.name,
#             recipient=message.sender,
#             type="response",
#             content="Modeling agent placeholder: trained GLM stub.",
#         )
