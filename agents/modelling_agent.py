import os
import pandas as pd
import json
import numpy as np
import re
from datetime import datetime
from utils.general_utils import save_json_safe
from utils.message_types import Message
from utils.model_trainer import ModelTrainer
from utils.model_evaluation import ModelEvaluation
from utils.prompt_library import PROMPTS
from langchain.memory import ConversationBufferMemory
from agents.base_agent import BaseAgent

class ModellingAgent(BaseAgent):
    def __init__(self, name="modelling", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False) # Short-term conversation memory for layered prompting

    # -------------------------------
    # Helper functions
    # -------------------------------
    # def _apply_llm_pipeline(self, df: pd.DataFrame, suggestion_text: str):
    #     """Executes LLM-generated preprocessing code safely."""
    #     code = self.extract_code_block(suggestion_text)
    #     if code is None:
    #         raise ValueError("No Python code block found in LLM suggestion.")

    #     # Safety: forbid imports or system calls
    #     if "import os" in code or "subprocess" in code or "open(" in code:
    #         raise ValueError("Unsafe code detected.")

    #     # Local execution namespace
    #     local_env = {"df": df.copy(), "np": np, "pd": pd}

    #     try:
    #         exec(code, {}, local_env)
    #     except Exception as e:
    #         raise ValueError(f"Adaptive preprocessing code failed: {e}")

    #     if "df" not in local_env:
    #         raise ValueError("Adaptive code did not modify df.")

    #     return local_env["df"]
    @staticmethod
    def extract_code_block(text: str) -> str | None:
        """Extract ```python ... ``` code block from LLM output."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _apply_llm_pipeline(self, X_train, y_train, X_test, model_code: str):
        """Executes LLM-generated model training code safely, returning predictions."""
        code = self.extract_code_block(model_code)
        if code is None:
            raise ValueError("No Python code block found in LLM suggestion.")

        # Safety: forbid imports or system calls
        if any(x in code for x in ["import os", "subprocess", "open("]):
            raise ValueError("Unsafe code detected.")

        # Local namespace with training data
        local_env = {
            "pd": pd,
            "np": np,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
        }

        # Execute LLM code
        try:
            exec(code, {}, local_env)
        except Exception as e:
            raise ValueError(f"Adaptive model training code failed: {e}")
        
        if "result" not in local_env:
            raise ValueError("Adaptive model code did not define `result`.")

        result = local_env["result"]
        if not isinstance(result, dict):
            raise ValueError("`result` must be a dictionary containing 'preds' and 'model'.")

        if "preds" not in result or "model" not in result:
            raise ValueError("`result` dictionary must include both 'preds' and 'model' keys.")

        preds = result["preds"]
        model = result["model"]

        # --- Validate predictions ---
        if not isinstance(preds, (pd.Series, np.ndarray, list)):
            raise ValueError("`predictions` must be array-like.")
        preds = np.asarray(preds).ravel()

        print(f"[{self.name}] ✅ Adaptive model produced {len(preds)} predictions.")
        return preds, model
    
    def _extract_model_choice(self, llm_text: str) -> str:
        text = llm_text.lower()

        if "use_gbm" in text:
            return "gbm"

        if "use_glm" in text:
            return "glm"
    
    def _extract_confidence(self, text):
        match = re.search(r"confidence:\s*([\d\.]+)", text.lower())
        return float(match.group(1)) if match else 0.5

    # -------------------------------
    # Main handler
    # -------------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting modelling pipeline...")

        processed_paths = message.metadata.get("processed_paths", None)
        artifacts_dir = "data/artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)

        if processed_paths is None:
            return Message(
                sender=self.name,
                recipient=message.sender,
                type="error",
                content="Missing processed dataset paths."
            )

        # --------------------
        # Load dataset
        # --------------------
        X_train = pd.read_csv(processed_paths["X_train"])
        X_test = pd.read_csv(processed_paths["X_test"])
        y_train = pd.read_csv(processed_paths["y_train"]).values.ravel()
        y_test = pd.read_csv(processed_paths["y_test"]).values.ravel()
        exposure_test = pd.read_csv(processed_paths["exposure_test"]).values.ravel()
        feature_names = pd.read_pickle(os.path.join(artifacts_dir, "feature_names.pkl"))

        dataset_desc = {
            "n_train": len(X_train),
            "n_features": len(X_train.columns),
            "features": X_train.columns.tolist(),
        }

        print(f"[{self.name}] Invoke layer 1...")

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        layer1_prompt = PROMPTS["modelling_layer1"].format(
            dataset_desc=json.dumps(dataset_desc, indent=2)
        )
        proposal = self.llm(layer1_prompt)
        self.memory.chat_memory.add_user_message(layer1_prompt)
        self.memory.chat_memory.add_ai_message(proposal)

        model_choice = self._extract_model_choice(proposal)
        print(f"[modelling] LLM selected model type: {model_choice}")

        print(f"[{self.name}] Invoke layer 2...")

        # --------------------
        # Layer 2: model code (LLM)
        # --------------------
        layer2_prompt = PROMPTS["modelling_layer2"].format(
        model_choice=model_choice,
        current_model_code=open("utils/model_trainer.py").read(),
        )
        
        model_code = self.llm(layer2_prompt)
        self.memory.chat_memory.add_user_message(layer2_prompt)
        self.memory.chat_memory.add_ai_message(model_code)

        confidence = self._extract_confidence(layer2_prompt)
        print(f"[{self.name}] Layer 2 confidence: {confidence:.2f}")

        print(model_code)

        # --------------------
        # Model training
        # --------------------
        print(f"[{self.name}] Invoke model training...")

        # Attempt LLM pipeline
        try:
            llm_model_preds, llm_model_obj = self._apply_llm_pipeline(X_train, y_train, X_test, model_code)
            llm_model_success = True
        except Exception as e:
            llm_model_preds, llm_model_obj = None, None
            llm_model_success = False
            print(f"[{self.name}] LLM model training failed: {e}")

        # Fallback pipeline
        if llm_model_success == True:
            model_predictions = llm_model_preds
        else:
            trainer = ModelTrainer(model_type="glm")
            trainer.train(X_train, y_train)
            model_predictions = trainer.predict(X_test)
            print(f"[{self.name}] Fallback to GLM model training")

        # --------------------
        # Evaluate model
        # --------------------
        print(f"[{self.name}] Invoke model evaluation...")
        evaluator = ModelEvaluation(model=trainer.model, model_type=model_choice)
        metrics = evaluator.evaluate(y_test, model_predictions, feature_names, exposure_test)

        # --------------------
        # Layer 3: model assessment (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...")

        layer3_prompt = PROMPTS["modelling_layer3"].format(
        model_type=model_choice,
        model_obj=llm_model_obj,
        metrics=json.dumps(metrics, indent=2))

        explanation = self.llm(layer3_prompt)
        self.memory.chat_memory.add_user_message(layer3_prompt)
        self.memory.chat_memory.add_ai_message(explanation)

        print(explanation)

        print(f"[{self.name}] Saving metadata...")

        # --------------------
        # Save metadata
        # --------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "status": "success",
            "proposal": proposal,
            "model_type_used": model_choice,
            "model_code": model_code,
            "code_confidence": confidence,
            "model_object": llm_model_obj,
            "metrics": metrics,
           # "llm_explanation": explanation,
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory (store both deterministic summary and LLM outputs)
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "model_training", metadata)
            history = self.hub.memory.get("model_history", [])
            history.append(metadata)
            self.hub.memory.update("model_history", history)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Model trained and evaluated successfully.",
            metadata=metadata,
        )


# import os
# import pandas as pd
# from datetime import datetime
# from utils.general_utils import save_json_safe
# from utils.message_types import Message
# from utils.model_trainer import ModelTrainer
# from utils.prompt_library import PROMPTS
# from agents.base_agent import BaseAgent

# class ModellingAgent(BaseAgent):
#     def __init__(self, name="modelling", shared_llm=None, system_prompt=None, model_type="glm", hub=None):
#         super().__init__(name)
#         self.llm = shared_llm
#         self.system_prompt = system_prompt
#         self.model_type = model_type
#         self.hub = hub

#     def handle_message(self, message: Message) -> Message:
#         print(f"[{self.name}] Starting model training ({self.model_type})...")

#         # --- Initiate paths ---
#         processed_paths = message.metadata.get("processed_paths", None)
#         artifacts_dir = "data/artifacts"
#         os.makedirs(artifacts_dir, exist_ok=True)

#         if processed_paths is None:
#             return Message(
#                 sender=self.name,
#                 recipient=message.sender,
#                 type="error",
#                 content="Missing processed dataset paths.",
#             )

#         # --- Load data ---
#         X_train = pd.read_csv(processed_paths["X_train"])
#         X_test = pd.read_csv(processed_paths["X_test"])
#         y_train = pd.read_csv(processed_paths["y_train"]).values.ravel()
#         y_test = pd.read_csv(processed_paths["y_test"]).values.ravel()
#         exposure_train = pd.read_csv(processed_paths["exposure_train"]).values.ravel()
#         exposure_test = pd.read_csv(processed_paths["exposure_test"]).values.ravel()
#         feature_names = pd.read_pickle(os.path.join(artifacts_dir, "feature_names.pkl"))

#         # --- Train model ---
#         trainer = ModelTrainer(model_type=self.model_type)
#         trainer.train(X_train, y_train)
#         preds = trainer.predict(X_test)
#         metrics = trainer.evaluate(y_test, preds, feature_names, exposure_test)

#         # --- Save model and evaluation ---
#         model_path = os.path.join(artifacts_dir, f"model_{self.model_type}.pkl")
#         preds_path = os.path.join(artifacts_dir, f"predictions.csv")

#         trainer.save(model_path)
#         pd.DataFrame({"y_true": y_test, "y_pred": preds}).to_csv(preds_path, index=False)
        
#         plot_path = metrics.get("Calibration Plot", None)
#         if plot_path and os.path.exists(plot_path):
#             from IPython.display import Image, display
#             print("\n--- Calibration / Lift Chart ---")
#             display(Image(filename=plot_path))

#         # --- LLM Explanation ---
#         modelling_prompt = PROMPTS["modelling"].format(
#         metrics=metrics,
#         )

#         explanation = self.llm(modelling_prompt) if self.llm else "No LLM backend available."

#         # --- Return message and log output ---
#         # Store metadata
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         metadata = {
#             "timestamp": timestamp,
#             "status": "success",
#             "model_path": model_path,
#             "preds_path": preds_path,
#             "metrics": metrics,
#             "llm_explanation": explanation,
#         }

#         results_dir = "data/results"
#         os.makedirs(results_dir, exist_ok=True)
#         meta_path = os.path.join(results_dir, f"{self.name}_metadata.json")
#         save_json_safe(metadata, meta_path)
#         metadata["metadata_file"] = meta_path

#         # Log to central memory
#         if self.hub and self.hub.memory:
#             self.hub.memory.log_event(self.name, "model_trained", metadata)
#             history = self.hub.memory.get("model_history", [])
#             history.append(metadata)
#             self.hub.memory.update("model_history", history)

#         # Return message to the hub
#         return Message(
#             sender=self.name,
#             recipient="hub",
#             type="response",
#             content=f"Model ({self.model_type}) trained and evaluated successfully.",
#             metadata=metadata,
#         )
