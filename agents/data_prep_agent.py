import os
import json
import re
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from utils.general_utils import save_json_safe
from utils.prompt_library import PROMPTS
from utils.data_pipeline import DataPipeline
from utils.data_cleaning import DataCleaning
from utils.message_types import Message
from agents.base_agent import BaseAgent
from utils.consistency import dataprep_consistency_snapshot


class DataPrepAgent(BaseAgent):
    def __init__(self, name="dataprep", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub

    # --------------------------    
    # Helper functions
    # --------------------------
    @staticmethod
    def extract_code_block(text: str) -> str | None:
        """
        Extract a Python code block from LLM-generated text.

        This function searches for the first fenced code block of the form
        ```python ... ``` or ``` ... ``` and returns its contents without
        the surrounding backticks.

        Parameters
        ----------
        text : str
            Raw text output from an LLM.

        Returns
        -------
        str | None
            The extracted Python source code if a code block is found.

        Notes
        -----
        - Matching is case-insensitive and supports optional 'python' tags.
        - Only the first matching code block is returned.
        """
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else None

    def _apply_llm_pipeline(self, df: pd.DataFrame, suggestion_text: str):
        """
        Execute an LLM-generated data-cleaning pipeline in a controlled environment.

        The LLM output is expected to contain a Python code block defining a
        `DataCleaning` class with a `clean(pd.DataFrame) -> pd.DataFrame` method.
        The code is executed in a restricted namespace with limited built-ins
        and explicit safety checks to reduce the risk of unsafe operations.

        Parameters
        ----------
        df : pd.DataFrame
        suggestion_text : str (raw LLM code)

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame produced by the adaptive pipeline.

        Raises
        ------
        ValueError, if no Python code block is found in the LLM output.
        ValueError, if forbidden operations or unsafe tokens are detected.
        ValueError, if the adaptive code fails to execute or violates the expected contract.
        ValueError, if the `clean` method does not return a pandas DataFrame.
        """

        code = self.extract_code_block(suggestion_text)
        if code is None:
            raise ValueError("No Python code block found in LLM suggestion.")

        # ---- Static safety checks (cheap but effective) ----
        FORBIDDEN_TOKENS = [
            "os.", "sys.", "subprocess", "open(", "exec(", "eval("
        ]
        if any(tok in code for tok in FORBIDDEN_TOKENS):
            raise ValueError("Unsafe operations detected in adaptive pipeline code.")

        # ---- Controlled execution environment ----
        SAFE_BUILTINS = {
            "__import__": __import__,
            "__build_class__": __build_class__,
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "print": print,
            "dict": dict,
            "list": list,
            "set": set,
            "tuple": tuple,
            "float": float,
            "int": int,
            "str": str,
            "bool": bool,
            "object": object, 
            "map": map,
        }

        exec_env = {
            "__builtins__": SAFE_BUILTINS,
            "__name__": "__adaptive_pipeline__",
            "pd": pd,
            "np": np,
        }

        try:
            exec(code, exec_env, exec_env)
        except Exception as e:
            raise ValueError(f"Pipeline definition failed to execute: {e}")

        if "DataCleaning" not in exec_env:
            raise ValueError("Adaptive code did not define a DataCleaning class.")

        # Instantiate and run
        pipeline = exec_env["DataCleaning"]()
        if not hasattr(pipeline, "clean"):
            raise ValueError("DataCleaning class has no clean() method.")

        try:
            df_out = pipeline.clean(df.copy())
        except Exception as e:
            raise ValueError(f"Adaptive pipeline execution failed: {e}")

        if not isinstance(df_out, pd.DataFrame):
            raise ValueError("DataCleaning.clean() must return a DataFrame.")

        return df_out

    def _compare_pipelines(self, det: pd.DataFrame, adapt: pd.DataFrame | None):
        """
        Compare deterministic and adaptive data-preparation outputs at the DataFrame level.

        This comparison is used to validate whether the adaptive pipeline
        produced a usable result and to provide summary statistics for LLM verification.

        Parameters
        ----------
        det : pd.DataFrame
            Output of the deterministic (baseline) data-cleaning pipeline.
        adapt : pd.DataFrame | None
            Output of the adaptive pipeline, or None if the adaptive pipeline failed.

        Returns
        -------
        dict
            A summary dictionary containing:
            - status : str
                One of {'adaptive_failed', 'deterministic_empty',
                        'adaptive_empty', 'adaptive_succeeded'}
            - n_rows_det, n_rows_adapt : int
            - n_cols_det, n_cols_adapt : int
            - feature_overlap : int
            - shape_det, shape_adapt : tuple
        """
        if adapt is None:
            return {"status": "adaptive_failed"}
        
        if det.empty:
            return {"status": "deterministic_empty"}
        if adapt.empty:
            return {"status": "adaptive_empty"}

        det_cols = set(det.columns)
        adapt_cols = set(adapt.columns)
        feature_overlap = len(det_cols & adapt_cols)

        return {
            "status": "adaptive_succeeded",
            "n_rows_det": len(det),
            "n_rows_adapt": len(adapt),
            "n_cols_det": len(det.columns),
            "n_cols_adapt": len(adapt.columns),
            "feature_overlap": feature_overlap,
            "shape_det": det.shape,
            "shape_adapt": adapt.shape,
        }
    
    def _extract_dataprep_choice(self, llm_text: str) -> str:
        """
        Extract the data-cleaning pipeline decision from LLM output.

        The function looks for a decision directive of the form:
            Decision: USE_ADAPTIVE
            Decision: KEEP_DETERMINISTIC

        Parameters
        ----------
        llm_text : str
            LLM-generated verification and decision text.

        Returns
        -------
        str
            Either 'adaptive' or 'deterministic'.
        """
        text = llm_text

        # 1) Prefer explicit Decision: line
        m = re.search(r'^\s*Decision\s*:\s*(USE_ADAPTIVE|KEEP_DETERMINISTIC)\s*$',
                    text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return "adaptive" if m.group(1).upper() == "USE_ADAPTIVE" else "deterministic"

        # 2) tolerant Decision: anywhere
        m2 = re.search(r'Decision\s*:\s*(USE_ADAPTIVE|KEEP_DETERMINISTIC)', text, flags=re.IGNORECASE)
        if m2:
            return "adaptive" if m2.group(1).upper() == "USE_ADAPTIVE" else "deterministic"

    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting data preparation...")

        metadata = message.metadata or {}

        # --------------------
        # Load dataset
        # --------------------
        dataset_path = message.metadata.get("dataset_path", "data/raw/freMTPL2freq.csv")
        try:
            df = pd.read_csv(dataset_path)
        except Exception as e:
            return Message(
                sender=self.name,
                recipient=message.sender,
                type="error",
                content=f"Failed to load dataset: {e}",
            )

        # Basic dataset info
        info_dict = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "missing_perc": df.isna().mean().to_dict(),
            "num_vars": df.select_dtypes(include="number").columns.tolist(),
            "cat_vars": df.select_dtypes(exclude="number").columns.tolist(),
        }

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 1...planning")

        # Optional: get recommendations from the ExplanationAgent from a previous iteration
        recommendations = metadata.get("recommendations", "No recommendations provided.")

        # Determine the LLM prompt, can either be normal prompt or revised prompt from the review agent
        if metadata.get("revised_prompt"):
            plan_prompt = metadata["revised_prompt"]
        else:
            plan_prompt = PROMPTS["dataprep_layer1"].format(info_dict=json.dumps(info_dict, indent=2), recommendations=recommendations)
        summary1, unc_dataprep_layer1 = self.llm(plan_prompt, return_uncertainty=True)

        # --------------------
        # Layer 2: suggestions (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 2...develop data preparation")

        # Prompt to get data cleaning code
        suggestion_prompt = PROMPTS["dataprep_layer2"].format(summary1=summary1,info_dict=json.dumps(info_dict, indent=2),pipeline_code=open("utils/data_cleaning.py").read())
        suggestion, unc_dataprep_layer2 = self.llm(suggestion_prompt, return_uncertainty=True)

        # Try to execute the adaptive pipeline
        try:
            adaptive_results = self._apply_llm_pipeline(df, suggestion)
            adaptive_success = True
        except Exception as e:
            adaptive_results = None
            adaptive_success = False
            print(f"[{self.name}] Adaptive pipeline failed: {e}")

        # Execute deterministic pipeline
        det_pipe = DataCleaning()
        deterministic_results = det_pipe.clean(df)

        # Compare both pipelines
        comparison_summary = self._compare_pipelines(deterministic_results, adaptive_results)

        # --------------------
        # Layer 3: verification (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...choose pipeline")

        status = comparison_summary.get("status")

        # Enforce deterministic pipeline in case of failures/empty results
        if status in ["adaptive_empty", "adaptive_failed"]:
            decision = "deterministic"
            verification = f"Forced decision due to status={status}"
            unc_dataprep_layer3 = 1
            print(f"[{self.name}] Forced decision due to status={status}")
        # Otherwise the agent will decide
        else:
            verify_prompt = PROMPTS["dataprep_layer3"].format(comparison=json.dumps(comparison_summary, indent=2))
            verification, unc_dataprep_layer3 = self.llm(verify_prompt, return_uncertainty=True)
            decision = self._extract_dataprep_choice(verification)

        # if decision == "adaptive" and adaptive_success:
        #     use_adaptive = True
        # elif decision == "deterministic":
        #     use_adaptive = False
        # else:
        #     use_adaptive = False
        
        use_adaptive = decision == "adaptive" and adaptive_success
        chosen_results = adaptive_results if use_adaptive else deterministic_results
        chosen_pipeline_name = "adaptive" if use_adaptive else "deterministic"

        print(f"[{self.name}] Final decision: using {chosen_pipeline_name} pipeline.")

        # --------------------
        # Preprocess the data
        # --------------------
        print(f"[{self.name}] Preprocessing the data.")

        preprocess_pipe = DataPipeline()
        df_processed = preprocess_pipe.process(chosen_results)

        # Save processed datasets
        processed_dir = "data/processed"
        artifacts_dir = "data/artifacts"
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        X_train_path = os.path.join(processed_dir, f"{base_name}_X_train.csv")
        X_test_path = os.path.join(processed_dir, f"{base_name}_X_test.csv")
        y_train_path = os.path.join(processed_dir, f"{base_name}_y_train.csv")
        y_test_path = os.path.join(processed_dir, f"{base_name}_y_test.csv")
        exposure_train_path = os.path.join(processed_dir, f"{base_name}_exposure_train.csv")
        exposure_test_path = os.path.join(processed_dir, f"{base_name}_exposure_test.csv")

        pd.DataFrame(df_processed["X_train"], columns=df_processed["feature_names"]).to_csv(X_train_path, index=False)
        pd.DataFrame(df_processed["X_test"], columns=df_processed["feature_names"]).to_csv(X_test_path, index=False)
        df_processed["y_train"].to_csv(y_train_path, index=False)
        df_processed["y_test"].to_csv(y_test_path, index=False)
        df_processed["exposure_train"].to_csv(exposure_train_path, index=False)
        df_processed["exposure_test"].to_csv(exposure_test_path, index=False)

        preproc_path = os.path.join(artifacts_dir, f"preprocessor.pkl")
        features_path = os.path.join(artifacts_dir, f"feature_names.pkl")
        joblib.dump(df_processed["feature_names"], features_path)
        joblib.dump(preprocess_pipe.preprocessor, preproc_path)

        print(f"[{self.name}] Invoke layer 4...summarize")

        # --------------------
        # Layer 4: LLM inspects result
        # --------------------
        explain_prompt = PROMPTS["dataprep_layer4"].format(verification=verification)
        explanation, unc_dataprep_layer4 = self.llm(explain_prompt, return_uncertainty=True)

        # --------------------
        # Save metadata
        # --------------------
        print(f"[{self.name}] Saving metadata...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store snapshot
        snapshot = dataprep_consistency_snapshot(chosen_results, target="ClaimNb")

        # Store metadata
        metadata = {
            "timestamp": timestamp,
            "status": "success",
            "used_pipeline": decision,
            "plan_dataprep": summary1,
            "adaptive_suggestion": suggestion,
            "comparison": comparison_summary,
            "verification": verification,
            "explanation": explanation,
            "consistency_snapshot": snapshot,
            "unc_dataprep_layer1": 1-unc_dataprep_layer1,
            "unc_dataprep_layer2": 1-unc_dataprep_layer2,
            "unc_dataprep_layer3": 1-unc_dataprep_layer3,
            "unc_dataprep_layer4": 1-unc_dataprep_layer4,
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
                "features": features_path,
            },
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "data_preparation", metadata)
            self.hub.memory.update("last_data_prep_summary", explanation)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content="Data cleaning and preprocessing completed successfully.",
            metadata=metadata,
        )