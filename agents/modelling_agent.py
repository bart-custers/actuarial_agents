import os
import joblib
import pandas as pd
import json
import numpy as np
import re
import glob
import tempfile
from datetime import datetime
from utils.general_utils import save_json_safe, make_json_compatible
from utils.message_types import Message
from utils.model_trainer import ModelTrainer
from utils.model_evaluation import ModelEvaluation
from utils.prompt_library import PROMPTS
from agents.base_agent import BaseAgent

# LangChain experimental REPL tool
try:
    from langchain_experimental.tools import PythonREPLTool
except Exception:
    # Fall back to an informative import error at runtime
    PythonREPLTool = None


class ModellingAgent(BaseAgent):
    def __init__(self, name="modelling", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub

        # Initialize Python REPL tool if available
        if PythonREPLTool is None:
            print("[ModellingAgent] WARNING: PythonREPLTool not available. LLM REPL execution will be disabled.")
            self.python_repl = None
        else:
            self.python_repl = PythonREPLTool()

    # -------------------------------
    # Helper functions
    # -------------------------------
    @staticmethod
    def extract_code_block(text: str) -> str | None:
        """Extract ```python ... ``` code block from LLM output."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _check_code_safety(self, code: str):
        """Basic safety checks on code string. Raises ValueError on dangerous content."""
        forbidden = [
            "import os", "subprocess", "open(", "socket", "shutil", "os.system",
            "pty", "fork(", "exec(", "eval(", "__import__", "from os", "requests", "urllib"
        ]
        lowered = code.lower()
        for f in forbidden:
            if f in lowered:
                raise ValueError(f"Unsafe code detected: contains '{f}'")

    def _apply_llm_pipeline(self, X_train, y_train, exposure_train, X_test, model_code: str):
        """
        Executes LLM-generated model training code via PythonREPLTool.

        The LLM code must:
        - Use the variables X_train, y_train, exposure_train, X_test (these are provided)
        - Define a dictionary `result = {"preds_train": ..., "preds_test": ..., "model": model_object}`
          where preds_* are array-like and model is a picklable model object.

        Returns: preds_train (np.ndarray), preds_test (np.ndarray), model_obj
        """
        if self.python_repl is None:
            raise RuntimeError("Python REPL tool not initialized on this agent.")

        code = self.extract_code_block(model_code)
        if code is None:
            raise ValueError("No Python code block found in LLM suggestion.")

        # Basic safety
        self._check_code_safety(code)

        # Create temp folder for data exchange
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tmpdir = tempfile.mkdtemp(prefix=f"llm_repl_{ts}_")
        xtrain_path = os.path.join(tmpdir, "X_train.joblib")
        ytrain_path = os.path.join(tmpdir, "y_train.joblib")
        exposure_path = os.path.join(tmpdir, "exposure_train.joblib")
        xtest_path = os.path.join(tmpdir, "X_test.joblib")
        result_path = os.path.join(tmpdir, "result.joblib")

        # Save training/test data to disk for REPL to load (joblib handles DataFrame/ndarray)
        joblib.dump(X_train, xtrain_path)
        joblib.dump(y_train, ytrain_path)
        joblib.dump(exposure_train, exposure_path)
        joblib.dump(X_test, xtest_path)

        # Build loader snippet to make variables available in REPL
        loader_snippet = f"""
        # ----- automatically injected loader (do not modify) -----
        import joblib
        X_train = joblib.load(r'{xtrain_path}')
        y_train = joblib.load(r'{ytrain_path}')
        exposure_train = joblib.load(r'{exposure_path}')
        X_test = joblib.load(r'{xtest_path}')
        # ----- end loader -----
        """

        # Ensure the LLM code ends up assigning a `result` dict. We'll append a small safety wrapper
        # that pickles the `result` variable back to disk for the host process to read.
        save_result_snippet = f"""
        # ----- automatically injected saver (do not modify) -----
        import joblib, types
        if 'result' in globals() and isinstance(result, dict):
            # Ensure result contains required keys
            joblib.dump(result, r'{result_path}')
        else:
            raise ValueError('LLM code did not produce a dictionary named `result`.')
        # ----- end saver -----
        """

        # Full code to run in the REPL
        full_code = "\n".join([loader_snippet, code, save_result_snippet])

        # Run the code in the REPL
        try:
            # PythonREPLTool.run executes the code and returns textual output; we rely on the saved file for results.
            repl_out = self.python_repl.run(full_code)
        except Exception as e:
            raise ValueError(f"Adaptive model training code failed in REPL: {e}")

        # Load the result dict back into this process
        if not os.path.exists(result_path):
            raise ValueError("Adaptive model code did not write the result file to disk (REPL execution).")

        result = joblib.load(result_path)

        # Validate result
        if not isinstance(result, dict):
            raise ValueError("`result` must be a dictionary.")
        for k in ("preds_train", "preds_test", "model"):
            if k not in result:
                raise ValueError(f"`result` dictionary missing required key: '{k}'")

        preds_train = result["preds_train"]
        preds_test = result["preds_test"]
        model_obj = result["model"]

        # Validate prediction shapes / types
        if not isinstance(preds_train, (pd.Series, np.ndarray, list)):
            raise ValueError("`preds_train` must be array-like.")
        if not isinstance(preds_test, (pd.Series, np.ndarray, list)):
            raise ValueError("`preds_test` must be array-like.")

        preds_train = np.asarray(preds_train).ravel()
        preds_test = np.asarray(preds_test).ravel()

        # Clean up temp dir (best-effort)
        try:
            for f in [xtrain_path, ytrain_path, exposure_path, xtest_path, result_path]:
                if os.path.exists(f):
                    os.remove(f)
            os.rmdir(tmpdir)
        except Exception:
            pass  # ignore cleanup errors

        return preds_train, preds_test, model_obj

    def _extract_model_choice(self, llm_text: str) -> str:
        text = llm_text
        # 1) Prefer explicit Decision: line (the prompt already asks for this)
        m = re.search(r'^\s*Decision\s*:\s*(USE_GLM|USE_GBM)\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return "glm" if m.group(1).upper() == "USE_GLM" else "gbm"
        # 2) Look for short explicit phrase "Decision: USE_GLM" inside any line (more tolerant)
        m2 = re.search(r'Decision\s*:\s*(USE_GLM|USE_GBM)', text, flags=re.IGNORECASE)
        if m2:
            return "glm" if m2.group(1).upper() == "USE_GLM" else "gbm"
        # 3) fallback: look for plain words
        if re.search(r'\bglm\b', text, flags=re.IGNORECASE):
            return "glm"
        if re.search(r'\bgbm\b', text, flags=re.IGNORECASE):
            return "gbm"
        return "glm"  # default

    def _extract_confidence(self, text):
        match = re.search(r"confidence:\s*([\d\.]+)", text.lower())
        return float(match.group(1)) if match else 0.5

    @staticmethod
    def load_latest_predictions(folder="data/final"):
        # Find all prediction files
        train_files = glob.glob(os.path.join(folder, "train_predictions_*.csv"))
        test_files = glob.glob(os.path.join(folder, "test_predictions_*.csv"))

        # Check that files exist
        if not train_files or not test_files:
            raise FileNotFoundError(f"No prediction files found in folder: {folder}")

        # Sort files by date descending (latest first)
        train_files.sort(reverse=True)
        test_files.sort(reverse=True)

        latest_train_file = train_files[0]
        latest_test_file = test_files[0]

        latest_train_preds = pd.read_csv(latest_train_file).iloc[:, 0].values.ravel()
        latest_test_preds = pd.read_csv(latest_test_file).iloc[:, 0].values.ravel()

        return latest_train_preds, latest_test_preds

    # -------------------------------
    # Main handler
    # -------------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting modelling pipeline...")
        metadata = message.metadata or {}

        processed_paths = metadata.get("processed_paths", None)
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
        exposure_train = pd.read_csv(processed_paths["exposure_train"]).values.ravel()
        exposure_test = pd.read_csv(processed_paths["exposure_test"]).values.ravel()
        feature_names = pd.read_pickle(os.path.join(artifacts_dir, "feature_names.pkl"))

        dataset_desc = {
            "n_train": len(X_train),
            "n_features": len(X_train.columns),
            "features": X_train.columns.tolist(),
        }

        print(f"[{self.name}] Invoke layer 1...model selection")

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        if metadata.get("revised_prompt"):
            layer1_prompt = metadata["revised_prompt"]
        else:
            layer1_prompt = PROMPTS["modelling_layer1"].format(
                dataset_desc=json.dumps(dataset_desc, indent=2)
            )
        plan = self.llm(layer1_prompt)
        print(f"[{self.name}] Layer1 plan output (short): {str(plan)[:400]!s}")

        model_choice = self._extract_model_choice(plan)
        print(f"[{self.name}] LLM selected model type: {model_choice}")

        # --------------------
        # Layer 2: model code (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 2...develop model code")

        layer2_prompt = PROMPTS["modelling_layer2"].format(
            model_choice=model_choice,
            current_model_code=open("utils/model_trainer.py").read(),
        )
        model_code = self.llm(layer2_prompt)
        print(f"[{self.name}] Received model code (first 1000 chars):\n{str(model_code)[:1000]}")

        # --------------------
        # Model training
        # --------------------
        print(f"[{self.name}] Invoke model training... (LLM pipeline via REPL)")

        # Attempt LLM pipeline (REPL)
        try:
            llm_model_preds_train, llm_model_preds_test, llm_model_obj = self._apply_llm_pipeline(
                X_train, y_train, exposure_train, X_test, model_code
            )
            llm_model_success = True
            print(f"[{self.name}] LLM pipeline succeeded: obtained preds and model object.")
        except Exception as e:
            llm_model_preds_train, llm_model_preds_test, llm_model_obj = None, None, None
            llm_model_success = False
            print(f"[{self.name}] LLM model training failed: {e}")

        # Fallback pipeline
        if llm_model_success:
            model_train_predictions = llm_model_preds_train
            model_test_predictions = llm_model_preds_test
            # if LLM returned a pickled model object, ensure we can access its .predict
            trainer_obj = None
            if hasattr(llm_model_obj, "predict"):
                trainer_obj = llm_model_obj
            else:
                # If LLM returned a trainer wrapper, try to use .model
                trainer_obj = llm_model_obj
            trainer_for_saving = trainer_obj
        else:
            print(f"[{self.name}] Fallback to deterministic model training")
            trainer = ModelTrainer(model_type=model_choice)
            # Some trainers expect exposure; keep signature backward-compatible
            try:
                trainer.train(X_train, y_train, exposure_train)
            except TypeError:
                trainer.train(X_train, y_train)
            model_train_predictions = trainer.predict(X_train)
            model_test_predictions = trainer.predict(X_test)
            trainer_for_saving = trainer

        # --------------------
        # Evaluate model
        # --------------------
        print(f"[{self.name}] Invoke model evaluation...")
        evaluator = ModelEvaluation(model=getattr(trainer_for_saving, "model", trainer_for_saving),
                                    model_type=model_choice)
        try:
            model_metrics = evaluator.evaluate(y_test, model_test_predictions, feature_names, exposure_test)
        except Exception as e:
            print(f"[{self.name}] Model evaluation failed: {e}")
            model_metrics = {"evaluation_error": str(e)}

        # Optional: impact/act_vs_exp / other analyses if implemented in evaluator
        act_vs_exp = None
        try:
            if hasattr(evaluator, "evaluate_act_vs_exp"):
                act_vs_exp = evaluator.evaluate_act_vs_exp(X_train, X_test, model_train_predictions, y_train,
                                                           model_test_predictions, y_test, feature_names)
        except Exception as e:
            print(f"[{self.name}] act_vs_exp analysis failed: {e}")
            act_vs_exp = None

        impact_analysis_tables = None
        try:
            if hasattr(evaluator, "evaluate_predicted"):
                # Safe-load previous predictions if available; otherwise pass None
                try:
                    preds_train_previous, preds_test_previous = self.load_latest_predictions()
                except Exception:
                    preds_train_previous, preds_test_previous = None, None
                impact_analysis_tables = evaluator.evaluate_predicted(
                    X_train, X_test,
                    model_train_predictions, preds_train_previous,
                    model_test_predictions, preds_test_previous,
                    feature_names
                )
        except Exception as e:
            print(f"[{self.name}] Impact analysis failed: {e}")
            impact_analysis_tables = None

        # --------------------
        # Layer 3: model assessment (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...analyse model evaluation")
        layer3_prompt = PROMPTS["modelling_layer3"].format(
            model_type=model_choice,
            act_vs_exp=json.dumps(make_json_compatible(act_vs_exp), indent=2),
            metrics=json.dumps(make_json_compatible(model_metrics), indent=2)
        )
        evaluation_text = self.llm(layer3_prompt)
        print(f"[{self.name}] Layer3 (LLM) short output: {str(evaluation_text)[:600]}")

        # --------------------
        # Layer 4: impact analysis (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 4...analyse impact analysis")
        layer4_prompt = PROMPTS["modelling_layer4"].format(
            impact_analysis_tables=json.dumps(make_json_compatible(impact_analysis_tables), indent=2)
        )
        impact_analysis_text = self.llm(layer4_prompt)
        print(f"[{self.name}] Layer4 (LLM) short output: {str(impact_analysis_text)[:600]}")

        # --------------------
        # Save metadata, model, predictions
        # --------------------
        print(f"[{self.name}] Saving metadata...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        storage_dir = "data/final"
        os.makedirs(storage_dir, exist_ok=True)

        model_path = os.path.join(storage_dir, f"{model_choice}_{timestamp}.joblib")
        # Save whatever trainer/model object we have
        try:
            joblib.dump(trainer_for_saving, model_path)
        except Exception as e:
            print(f"[{self.name}] Warning: failed to joblib.dump model object: {e}")

        # Save predictions
        train_pred_path = os.path.join(storage_dir, f"train_predictions_{timestamp}.csv")
        test_pred_path = os.path.join(storage_dir, f"test_predictions_{timestamp}.csv")
        pd.DataFrame({"train_prediction": model_train_predictions}).to_csv(train_pred_path, index=False)
        pd.DataFrame({"test_prediction": model_test_predictions}).to_csv(test_pred_path, index=False)

        # Snapshot for consistency checks
        snapshot = model_metrics

        # Metadata to save and push to central memory
        metadata_out = {
            "timestamp": timestamp,
            "status": "success",
            "plan": plan,
            "model_type_used": model_choice,
            "model_code": model_code,
            "model_metrics": model_metrics,
            "act_vs_exp": act_vs_exp,
            "evaluation": evaluation_text,
            "impact_analysis": impact_analysis_text,
            "consistency_snapshot": snapshot,
            "model_predictions_path": storage_dir,
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata_out, meta_path)
        metadata_out["metadata_file"] = meta_path

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "model_training", metadata_out)
            history = self.hub.memory.get("model_history", [])
            history.append(metadata_out)
            self.hub.memory.update("model_history", history)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Model trained and evaluated successfully.",
            metadata=metadata_out,
        )



# import os
# import joblib
# import pandas as pd
# import json
# import numpy as np
# import re
# import glob
# from datetime import datetime
# from utils.general_utils import save_json_safe, make_json_compatible
# from utils.message_types import Message
# from utils.model_trainer import ModelTrainer
# from utils.model_evaluation import ModelEvaluation
# from utils.prompt_library import PROMPTS
# #from langchain_community.memory import ConversationBufferMemory
# from agents.base_agent import BaseAgent


# class ModellingAgent(BaseAgent):
#     def __init__(self, name="modelling", shared_llm=None, system_prompt=None, hub=None):
#         super().__init__(name)
#         self.llm = shared_llm
#         self.system_prompt = system_prompt
#         self.hub = hub
#        # self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=1) # Short-term conversation memory for layered prompting

#     # -------------------------------
#     # Helper functions
#     # -------------------------------
#     @staticmethod
#     def extract_code_block(text: str) -> str | None:
#         """Extract ```python ... ``` code block from LLM output."""
#         match = re.search(r"```python(.*?)```", text, re.DOTALL)
#         return match.group(1).strip() if match else None
    
#     def _apply_llm_pipeline(self, X_train, y_train, exposure_train, X_test, model_code: str):
#         """Executes LLM-generated model training code safely, returning predictions."""
#         code = self.extract_code_block(model_code)
#         if code is None:
#             raise ValueError("No Python code block found in LLM suggestion.")

#         # Safety: forbid imports or system calls
#         if any(x in code for x in ["import os", "subprocess", "open("]):
#             raise ValueError("Unsafe code detected.")

#         # Local namespace with training data
#         local_env = {
#             "pd": pd,
#             "np": np,
#             "X_train": X_train,
#             "y_train": y_train,
#             "exposure_train": exposure_train,
#             "X_test": X_test,
#         }

#         # Execute LLM code
#         try:
#             exec(code, {}, local_env)
#         except Exception as e:
#             raise ValueError(f"Adaptive model training code failed: {e}")
        
#         if "result" not in local_env:
#             raise ValueError("Adaptive model code did not define `result`.")

#         result = local_env["result"]
#         if not isinstance(result, dict):
#             raise ValueError("`result` must be a dictionary.")

#         if "preds_train" not in result or "preds_test" not in result or "model" not in result:
#             raise ValueError("`result` dictionary must include all keys.")

#         preds_train = result["preds_train"]
#         preds_test = result["preds_test"]
#         model = result["model"]

#         # --- Validate predictions ---
#         if not isinstance(preds_train, (pd.Series, np.ndarray, list)):
#             raise ValueError("`train predictions` must be array-like.")
#         preds_train = np.asarray(preds_train).ravel()

#         if not isinstance(preds_test, (pd.Series, np.ndarray, list)):
#             raise ValueError("`train predictions` must be array-like.")
#         preds_test = np.asarray(preds_test).ravel()

#         return preds_train, preds_test, model
    
#     def _extract_model_choice(self, llm_text: str) -> str:
#         text = llm_text
#         # 1) Prefer explicit Decision: line (the prompt already asks for this)
#         m = re.search(r'^\s*Decision\s*:\s*(USE_GLM|USE_GBM)\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
#         if m:
#             return "glm" if m.group(1).upper() == "USE_GLM" else "gbm"

#         # 2) Look for short explicit phrase "Decision: USE_GLM" inside any line (more tolerant)
#         m2 = re.search(r'Decision\s*:\s*(USE_GLM|USE_GBM)', text, flags=re.IGNORECASE)
#         if m2:
#             return "glm" if m2.group(1).upper() == "USE_GLM" else "gbm"
    
#     def _extract_confidence(self, text):
#         match = re.search(r"confidence:\s*([\d\.]+)", text.lower())
#         return float(match.group(1)) if match else 0.5

#     @staticmethod
#     def load_latest_predictions(folder="data/final"):
#         # Find all prediction files
#         train_files = glob.glob(os.path.join(folder, "train_predictions_*.csv"))
#         test_files  = glob.glob(os.path.join(folder, "test_predictions_*.csv"))

#         # Check that files exist
#         if not train_files or not test_files:
#             raise FileNotFoundError(f"No prediction files found in folder: {folder}")

#         # Sort files by date descending (latest first)
#         train_files.sort(reverse=True)
#         test_files.sort(reverse=True)

#         # Take the latest file
#         latest_train_file = train_files[0]
#         latest_test_file  = test_files[0]

#         latest_train_preds = pd.read_csv(latest_train_file).iloc[:, 0].values.ravel()
#         latest_test_preds  = pd.read_csv(latest_test_file).iloc[:, 0].values.ravel()

#         return latest_train_preds, latest_test_preds 


#     # -------------------------------
#     # Main handler
#     # -------------------------------
#     def handle_message(self, message: Message) -> Message:
#         print(f"[{self.name}] Starting modelling pipeline...")

#         metadata = message.metadata or {}

#         processed_paths = message.metadata.get("processed_paths", None)
#         artifacts_dir = "data/artifacts"
#         os.makedirs(artifacts_dir, exist_ok=True)

#         if processed_paths is None:
#             return Message(
#                 sender=self.name,
#                 recipient=message.sender,
#                 type="error",
#                 content="Missing processed dataset paths."
#             )

#         # --------------------
#         # Load dataset
#         # --------------------
#         X_train = pd.read_csv(processed_paths["X_train"])
#         X_test = pd.read_csv(processed_paths["X_test"])
#         y_train = pd.read_csv(processed_paths["y_train"]).values.ravel()
#         y_test = pd.read_csv(processed_paths["y_test"]).values.ravel()
#         exposure_train = pd.read_csv(processed_paths["exposure_train"]).values.ravel()
#         exposure_test = pd.read_csv(processed_paths["exposure_test"]).values.ravel()
#         feature_names = pd.read_pickle(os.path.join(artifacts_dir, "feature_names.pkl"))

#         dataset_desc = {
#             "n_train": len(X_train),
#             "n_features": len(X_train.columns),
#             "features": X_train.columns.tolist(),
#         }

#         print(f"[{self.name}] Invoke layer 1...model selection")

#         # --------------------
#         # Layer 1: recall & plan (LLM)
#         # --------------------
#         if metadata.get("revised_prompt"):
#             layer1_prompt = metadata["revised_prompt"]
#         else:
#             layer1_prompt = PROMPTS["modelling_layer1"].format(
#                 dataset_desc=json.dumps(dataset_desc, indent=2)
#             )
#         plan = self.llm(layer1_prompt)
#        # self.memory.chat_memory.add_user_message(layer1_prompt)
#        # self.memory.chat_memory.add_ai_message(plan)

#         model_choice = self._extract_model_choice(plan)
#         print(f"[modelling] LLM selected model type: {model_choice}")

#         # --------------------
#         # Layer 2: model code (LLM)
#         # --------------------
#         print(f"[{self.name}] Invoke layer 2...develop model code")

#         layer2_prompt = PROMPTS["modelling_layer2"].format(
#         model_choice=model_choice,
#         current_model_code=open("utils/model_trainer.py").read(),
#         )
        
#         model_code = self.llm(layer2_prompt)
#       # self.memory.chat_memory.add_user_message(layer2_prompt)
#        # self.memory.chat_memory.add_ai_message(model_code)

#         #confidence = self._extract_confidence(layer2_prompt)
#         #print(f"[{self.name}] Layer 2 confidence: {confidence:.2f}")

#         print(model_code)

#         # --------------------
#         # Model training
#         # --------------------
#         print(f"[{self.name}] Invoke model training...")

#         # Attempt LLM pipeline
#         try:
#             llm_model_preds_train, llm_model_preds_test, llm_model_obj = self._apply_llm_pipeline(X_train, y_train, exposure_train, X_test, model_code)
#             llm_model_success = True
#         except Exception as e:
#             llm_model_preds_train, llm_model_preds_test, llm_model_obj = None, None, None
#             llm_model_success = False
#             print(f"[{self.name}] LLM model training failed: {e}")

#         # Fallback pipeline
#         if llm_model_success == True:
#             model_train_predictions = llm_model_preds_train
#             model_test_predictions = llm_model_preds_test
#             trainer = llm_model_obj
#         else:
#             print(f"[{self.name}] Fallback to deterministic model training")
#             trainer = ModelTrainer(model_type=model_choice)
#             trainer.train(X_train, y_train, exposure_train)
#             model_train_predictions = trainer.predict(X_train)
#             model_test_predictions = trainer.predict(X_test)

#         # --------------------
#         # Evaluate model
#         # --------------------
#         print(f"[{self.name}] Invoke model evaluation...")
#         # Get high over metrics
#         evaluator = ModelEvaluation(model=trainer.model, model_type=model_choice)
#         model_metrics = evaluator.evaluate(y_test, model_test_predictions, feature_names, exposure_test)
        
#         # Perform actual vs expected
#         act_vs_exp = evaluator.evaluate_act_vs_exp(X_train, X_test, model_train_predictions, y_train,
#         model_test_predictions, y_test, feature_names)

#         # Perform impact analysis
#         preds_train_previous, preds_test_previous = self.load_latest_predictions()
#         impact_analysis_tables = evaluator.evaluate_predicted(X_train, X_test, model_train_predictions, preds_train_previous,
#         model_test_predictions, preds_test_previous, feature_names)

#         # --------------------
#         # Layer 3: model assessment (LLM)
#         # --------------------
#         print(f"[{self.name}] Invoke layer 3...analyse model evaluation")

#         layer3_prompt = PROMPTS["modelling_layer3"].format(
#             model_type=model_choice,
#             act_vs_exp=act_vs_exp,
#             metrics=json.dumps(make_json_compatible(model_metrics), indent=2))

#         evaluation = self.llm(layer3_prompt)
#      #   self.memory.chat_memory.add_user_message(layer3_prompt)
#       #  self.memory.chat_memory.add_ai_message(evaluation)

#         print(evaluation)

#         # --------------------
#         # Layer 4: impact analysis (LLM)
#         # --------------------
#         print(f"[{self.name}] Invoke layer 4...analyse impact analysis")

#         layer4_prompt = PROMPTS["modelling_layer4"].format(
#             impact_analysis_tables=impact_analysis_tables)

#         impact_analysis = self.llm(layer4_prompt)

#         print(impact_analysis)

#         # --------------------
#         # Save metadata
#         # --------------------
#         print(f"[{self.name}] Saving metadata...")

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#         # Store model
#         storage_dir = "data/final"
#         os.makedirs(storage_dir, exist_ok=True)

#         model_path = os.path.join(storage_dir, f"{model_choice}_{timestamp}.joblib")
#         joblib.dump(trainer, model_path)

#         # Store model predictions
#         train_pred_path = os.path.join(storage_dir, f"train_predictions_{timestamp}.csv")
#         test_pred_path = os.path.join(storage_dir, f"test_predictions_{timestamp}.csv")

#         pd.DataFrame({"train_prediction": model_train_predictions}).to_csv(train_pred_path, index=False)
#         pd.DataFrame({"test_prediction": model_test_predictions}).to_csv(test_pred_path, index=False)

#         # Store snapshot
#         snapshot = model_metrics

#         # Store metadata
#         metadata = {
#             "timestamp": timestamp,
#             "status": "success",
#             "plan": plan,
#             "model_type_used": model_choice,
#             "model_code": model_code,
#          #   "code_confidence": confidence,
#          #   "model_object": llm_model_obj,
#             "model_metrics": model_metrics,
#             "act_vs_exp": act_vs_exp,
#             "evaluation": evaluation,
#             "impact_analysis": impact_analysis,
#             "consistency_snapshot": snapshot,
#             "model_predictions_path": storage_dir,
#         }

#         results_dir = "data/results"
#         os.makedirs(results_dir, exist_ok=True)
#         meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
#         save_json_safe(metadata, meta_path)
#         metadata["metadata_file"] = meta_path

#         # Log to central memory
#         if self.hub and self.hub.memory:
#             self.hub.memory.log_event(self.name, "model_training", metadata)
#             history = self.hub.memory.get("model_history", [])
#             history.append(metadata)
#             self.hub.memory.update("model_history", history)

#         # Return message to the hub
#         return Message(
#             sender=self.name,
#             recipient="hub",
#             type="response",
#             content=f"Model trained and evaluated successfully.",
#             metadata=metadata,
#         )