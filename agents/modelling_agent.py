import os
import joblib
import pandas as pd
import json
import numpy as np
import re
import glob
from datetime import datetime
from utils.general_utils import save_json_safe, make_json_compatible, extract_analysis
from utils.message_types import Message
from utils.glm_trainer_code import GLMTrainer
from utils.gbm_trainer_code import GBMTrainer
from utils.model_evaluation import ModelEvaluation
from utils.prompt_library import PROMPTS
from agents.base_agent import BaseAgent
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


class ModellingAgent(BaseAgent):
    def __init__(self, name="modelling", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub

    # -------------------------------
    # Helper functions
    # -------------------------------
    def _extract_model_choice(self, llm_text: str) -> str:
        """
        Extract the model selection decision from LLM output.

        The function looks for an answer of the form:
            Decision: USE_GLM
            Decision: USE_GBM

        Args:
            llm_text (str): LLM-generated text containing a model selection decision.

        Returns:
            str (model choice)
        """
        text = llm_text
        # 1) Prefer explicit Decision: line (the prompt already asks for this)
        m = re.search(r'^\s*Decision\s*:\s*(USE_GLM|USE_GBM)\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return "glm" if m.group(1).upper() == "USE_GLM" else "gbm"

        # 2) Look for short explicit phrase "Decision: USE_GLM" inside any line (more tolerant)
        m2 = re.search(r'Decision\s*:\s*(USE_GLM|USE_GBM)', text, flags=re.IGNORECASE)
        if m2:
            return "glm" if m2.group(1).upper() == "USE_GLM" else "gbm"
        
        return "glm" # safe fallback
    
    @staticmethod
    def extract_code_block(text: str) -> str | None:
        """
        Extract a Python code block from LLM-generated text.

        This function searches for the first fenced code block of the form
        ```python ... ``` or ``` ... ``` and returns its contents without
        the surrounding backticks.

        Args:
            text (str): Raw text output from an LLM.

        Returns:
            str | None: The extracted Python source code if a code block is found.
        """
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1) if match else None
    
    def _apply_llm_pipeline(self, X_train, y_train, exposure_train, X_test, model_code: str):
        """
        Execute LLM-generated model training code in a controlled environment and return predictions.

        The LLM-generated code is expected to define a variable named `result`
        in the execution environment. The `result` object must be a dictionary
        with the following keys:
            - 'preds_train': training set predictions
            - 'preds_test': test set predictions
            - 'model': trained model object

        Args:
            X_train (array-like or pd.DataFrame): Training feature matrix.
            y_train (array-like): Training target values.
            exposure_train (array-like): Exposure values corresponding to the training data.
            X_test (array-like or pd.DataFrame): Test feature matrix.
            model_code (str): Raw LLM output containing a Python code block that performs model training and prediction.

        Returns:
            tuple[np.ndarray, np.ndarray, object]
                - Training predictions
                - Test predictions
                - Trained model object

        Raises:
            ValueError, if no Python code block is found in the LLM output.
            ValueError, if unsafe operations are detected in the generated code.
            ValueError, if execution fails or the expected output contract is violated.
        """
        code = self.extract_code_block(model_code)
        if code is None:
            raise ValueError("No Python code block found in LLM suggestion.")

        # Safety checks
        if any(x in code for x in ["import os", "subprocess", "open("]):
            raise ValueError("Unsafe code detected.")

        # Controlled execution environment
        local_env = {
            "pd": pd,
            "np": np,
            "PoissonRegressor": PoissonRegressor,
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
            "GridSearchCV": GridSearchCV,
            "X_train": X_train,
            "y_train": y_train,
            "exposure_train": exposure_train,
            "X_test": X_test,
        }

        # Execute LLM code
        try:
            exec(code, local_env, local_env)
        except Exception as e:
            raise ValueError(f"Adaptive model training code failed: {e}")
        
        if "result" not in local_env:
            raise ValueError("Adaptive model code did not define `result`.")

        result = local_env["result"]
        if not isinstance(result, dict):
            raise ValueError("`result` must be a dictionary.")

        if "preds_train" not in result or "preds_test" not in result or "model" not in result:
            raise ValueError("`result` dictionary must include all keys.")

        preds_train = result["preds_train"]
        preds_test = result["preds_test"]
        model = result["model"]

        # --- Validate predictions ---
        if not isinstance(preds_train, (pd.Series, np.ndarray, list)):
            raise ValueError("`train predictions` must be array-like.")
        preds_train = np.asarray(preds_train).ravel()

        if not isinstance(preds_test, (pd.Series, np.ndarray, list)):
            raise ValueError("`test predictions` must be array-like.")
        preds_test = np.asarray(preds_test).ravel()

        return preds_train, preds_test, model

    @staticmethod
    def load_latest_predictions(folder="data/preds"):
        """
        Load the most recent train and test prediction files from disk, based on timestamped filenames.

        The function searches for CSV files named:
            - train_predictions_*.csv
            - test_predictions_*.csv

        Args:
            folder (str, default="data/preds"): Directory containing stored prediction files.

        Returns:
            tuple[np.ndarray, np.ndarray]
                - Latest training predictions
                - Latest test predictions
        """
        train_files = glob.glob(os.path.join(folder, "train_predictions_*.csv"))
        test_files  = glob.glob(os.path.join(folder, "test_predictions_*.csv"))
        #X_train_files = glob.glob(os.path.join(folder, "X_train_*.csv"))
        #X_test_files = glob.glob(os.path.join(folder, "X_test_*.csv"))

        # Check that files exist
        if not train_files or not test_files:
            raise FileNotFoundError(f"No prediction files found in folder: {folder}")

        # Sort files by date descending (latest first)
        train_files.sort(reverse=True)
        test_files.sort(reverse=True)
        #X_train_files.sort(reverse=True)
        #X_test_files.sort(reverse=True)

        # Take the latest file
        latest_train_file = train_files[0]
        latest_test_file  = test_files[0]
        #X_train_files = X_train_files[0]
        #X_test_files = X_test_files[0]

        latest_train_preds = pd.read_csv(latest_train_file).iloc[:, 0].values.ravel()
        latest_test_preds  = pd.read_csv(latest_test_file).iloc[:, 0].values.ravel()
        #latest_X_train = pd.read_csv(X_train_files)
        #latest_X_test = pd.read_csv(X_test_files)
        latest_X_train = {}
        latest_X_test = {}

        return latest_train_preds, latest_test_preds, latest_X_train, latest_X_test 

    # -------------------------------
    # Main handler
    # -------------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting modelling pipeline...")

        metadata = message.metadata or {}

        # --------------------
        # Load datasets
        # --------------------
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

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 1...model selection")

        # Optional: get recommendations from the ExplanationAgent from a previous iteration
        recommendations = metadata.get("recommendations", "No recommendations provided.")

        # Determine the LLM prompt, can either be normal prompt or revised prompt from the review agent
        if metadata.get("revised_prompt"):
            layer1_prompt = metadata["revised_prompt"]
        else:
            layer1_prompt = PROMPTS["modelling_layer1"].format(
                dataset_desc=json.dumps(dataset_desc, indent=2), recommendations=recommendations
            )
        plan, unc_modelling_layer1 = self.llm(layer1_prompt, return_uncertainty=True)

        model_choice = self._extract_model_choice(plan)
        print(f"[modelling] LLM selected model type: {model_choice}")

        # --------------------
        # Layer 2: model code (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 2...develop model code")

        # Load example code based on model choice
        if model_choice == "glm":
            trainer = GLMTrainer()
            example_code = "utils/glm_trainer_code.txt"
        elif model_choice == "gbm":
            trainer = GBMTrainer()
            example_code = "utils/gbm_trainer_code.txt"
        else:
            raise ValueError(f"Unknown model type: {model_choice}")
        
        with open(example_code, "r") as f:
            trainer_code = f.read()

        # Prompt to get modelling code
        layer2_prompt = PROMPTS["modelling_layer2"].format(
        model_choice=model_choice, trainer_code=trainer_code
        )
        model_code, unc_modelling_layer2 = self.llm(layer2_prompt, return_uncertainty=True)

        # --------------------
        # Model training
        # --------------------
        print(f"[{self.name}] Invoke model training...")

        # Attempt to execute the LLM model training pipeline
        try:
            llm_model_preds_train, llm_model_preds_test, llm_model_obj = self._apply_llm_pipeline(X_train, y_train, exposure_train, X_test, model_code)
            llm_model_success = True
        except Exception as e:
            llm_model_preds_train, llm_model_preds_test, llm_model_obj = None, None, None
            llm_model_success = False
            pipeline_error = str(e)
            print(f"[{self.name}] LLM model training failed: {e}")

        # Fallback pipeline
        if llm_model_success == True:
            print(f"[{self.name}] Model training with LLM-generated code succeeded.")
            model_used = "llm_adaptive"
            model_reason = "llm pipeline succeeded"
            model_train_predictions = llm_model_preds_train
            model_test_predictions = llm_model_preds_test
            trainer = llm_model_obj
        else:
            print(f"[{self.name}] Fallback to deterministic model training")
            model_used = "deterministic_fallback"
            model_reason = pipeline_error
            trainer.train(X_train, y_train, exposure_train)
            model_train_predictions = trainer.predict(X_train)
            model_test_predictions = trainer.predict(X_test)
            trainer = trainer.model

        # --------------------
        # Evaluate model
        # --------------------
        print(f"[{self.name}] Invoke model evaluation...")
        
        # Get high over metrics
        evaluator = ModelEvaluation(model=trainer, model_type=model_choice)
        model_metrics = evaluator.evaluate(y_test, model_test_predictions, feature_names, exposure_test)
        
        # Perform actual vs expected
        act_vs_exp = evaluator.evaluate_act_vs_exp(X_train, X_test, model_train_predictions, y_train,
        model_test_predictions, y_test, feature_names)

        # Perform impact analysis
        preds_train_previous, preds_test_previous, X_train_previous, X_test_previous = self.load_latest_predictions()
        impact_analysis_tables = evaluator.evaluate_predicted(X_train, X_test, X_train_previous, X_test_previous, model_train_predictions, preds_train_previous,
        model_test_predictions, preds_test_previous, feature_names)

        print(impact_analysis_tables)

        # --------------------
        # Layer 3: model assessment (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...analyse model evaluation")

        layer3_prompt = PROMPTS["modelling_layer3"].format(
            model_type=model_choice,
            act_vs_exp=act_vs_exp,
            metrics=json.dumps(make_json_compatible(dict(list(model_metrics.items())[:5])), indent=2))

        evaluation, unc_modelling_layer3 = self.llm(layer3_prompt, return_uncertainty=True)
        evaluation_text = extract_analysis(evaluation)

        # --------------------
        # Layer 4: impact analysis (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 4...analyse impact analysis")

        layer4_prompt = PROMPTS["modelling_layer4"].format(
            impact_analysis_tables=(impact_analysis_tables))

        impact_analysis, unc_modelling_layer4 = self.llm(layer4_prompt, return_uncertainty=True)
        impact_analysis_text = extract_analysis(impact_analysis)

        print(impact_analysis_text)

        # --------------------
        # Save metadata
        # --------------------
        print(f"[{self.name}] Saving metadata...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store model
        storage_dir = "data/preds"
        os.makedirs(storage_dir, exist_ok=True)

        model_path = os.path.join(storage_dir, f"{model_choice}_{timestamp}.joblib")
        joblib.dump(trainer, model_path)

        # Store model predictions
        train_pred_path = os.path.join(storage_dir, f"train_predictions_{timestamp}.csv")
        test_pred_path = os.path.join(storage_dir, f"test_predictions_{timestamp}.csv")

        pd.DataFrame({"train_prediction": model_train_predictions}).to_csv(train_pred_path, index=False)
        pd.DataFrame({"test_prediction": model_test_predictions}).to_csv(test_pred_path, index=False)

        # Store df including predictions
        train_df = X_train.copy()
        train_df['ClaimNb'] = y_train
        train_df['Exposure'] = exposure_train
        train_df['Prediction'] = model_train_predictions

        test_df = X_test.copy()
        test_df['ClaimNb'] = y_test
        test_df['Exposure'] = exposure_test
        test_df['Prediction'] = model_test_predictions

        df_predictions = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        full_path = os.path.join(storage_dir, f"df_predictions_{timestamp}.csv")
        df_predictions.to_csv(full_path, index=False)

        # Store snapshot
        snapshot = model_metrics

        # Store metadata
        metadata = {
            "timestamp": timestamp,
            "status": "success",
            "plan_modelling": plan,
            "model_type_used": model_choice,
            "model_code": model_code,
            "model_used": model_used,
            "model_reason": model_reason,
            "model_metrics": model_metrics,
            "act_vs_exp": act_vs_exp,
            "evaluation": evaluation_text,
            "impact_analysis": impact_analysis_text,
            "consistency_snapshot": snapshot,
            "model_predictions_path": storage_dir,
            "unc_modelling_layer1": 1-unc_modelling_layer1,
            "unc_modelling_layer2": 1-unc_modelling_layer2,
            "unc_modelling_layer3": 1-unc_modelling_layer3,
            "unc_modelling_layer4": 1-unc_modelling_layer4,            
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory
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