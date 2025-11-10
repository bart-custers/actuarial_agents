import os
import json
import re
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from utils.general_utils import save_json_safe
from utils.prompt_library import PROMPTS
from utils.data_pipeline import DataPipeline
from utils.message_types import Message
from agents.base_agent import BaseAgent

# ---------------------------
# Agent
# ---------------------------

class DataPrepAgent(BaseAgent):
    def __init__(self, name="dataprep", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub
        # Short-term conversation memory for layered prompting
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    # def _make_chain(self, prompt_key):
    #     prompt = PromptTemplate.from_template(PROMPTS[prompt_key])
    #     return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory, verbose=False)
        
    def _make_chain(self, prompt_key):
        prompt = PromptTemplate.from_template(PROMPTS[prompt_key])
        llm_runnable = RunnableLambda(lambda x: self.llm(x["input"]))
        return LLMChain(llm=llm_runnable, prompt=prompt, memory=self.memory, verbose=False)    
        # Setup LLMChains
        # If self.llm is a LangChain LLM implementor (e.g., ChatOpenAI), we can pass it to LLMChain.
        # If self.llm is a callable wrapper (returns string), we will call it directly as fallback.
        # try:
        #     self.chain_layer1 = LLMChain(llm=self.llm, prompt=PROMPTS["dataprep_layer1"], memory=self.memory)
        #     self.chain_layer2 = LLMChain(llm=self.llm, prompt=PROMPTS["dataprep_layer2"], memory=self.memory)
        #     self.chain_layer3 = LLMChain(llm=self.llm, prompt=PROMPTS["dataprep_layer3"], memory=self.memory)
        #     self.use_chains = True
        # except Exception:
        #     # Fallback: call llm directly with formatted prompt texts
        #     self.chain_layer1 = None
        #     self.chain_layer2 = None
        #     self.chain_layer3 = None
        #     self.use_chains = False

    # --------------------------
    # Helper functions
    # --------------------------

    def extract_code_block(text: str) -> str | None:
        """Extract ```python ... ``` code block from LLM output."""
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_confidence(self, suggestion_text: str) -> float:
        """Extract confidence rating (0–1) from LLM output if present."""
        import re
        match = re.search(r"confidence[:=]\s*([\d\.]+)", suggestion_text.lower())
        return float(match.group(1)) if match else 0.5

    # def _apply_llm_pipeline(self, df: pd.DataFrame, suggestion_text: str):
    #     """Safely run any code snippet proposed by LLM (sandbox-like)."""
    #     if "import" in suggestion_text or "os." in suggestion_text:
    #         raise ValueError("Unsafe LLM suggestion detected.")
    #     # This example only accepts dictionary-style mapping suggestions
    #     # In future: implement real DSL or code parsing
    #     return DataPipeline().clean(df)
    def _apply_llm_pipeline(self, df: pd.DataFrame, suggestion_text: str):
        """Executes LLM-generated preprocessing code safely."""
        code = self.extract_code_block(suggestion_text)
        if code is None:
            raise ValueError("No Python code block found in LLM suggestion.")
            raise ValueError("No Python code block found in LLM suggestion.")

        # Safety: forbid imports or system calls
        if "import os" in code or "subprocess" in code or "open(" in code:
            raise ValueError("Unsafe code detected.")

        # Local execution namespace
        local_env = {"df": df.copy(), "np": np, "pd": pd}

        try:
            exec(code, {}, local_env)
        except Exception as e:
            raise ValueError(f"Adaptive preprocessing code failed: {e}")

        if "df" not in local_env:
            raise ValueError("Adaptive code did not modify df.")

        return local_env["df"]

    def _compare_pipelines(self, det, adapt):
        """Quantitative comparison of two pipelines."""
        if adapt is None:
            return {"status": "adaptive_failed"}
        det_shape = det["X_train"].shape
        adapt_shape = adapt["X_train"].shape if "X_train" in adapt else None
        feature_overlap = len(set(det["feature_names"]) & set(adapt["feature_names"]))
        return {
            "status": "adaptive_succeeded",
            "feature_overlap": feature_overlap,
            "shape_diff": (det_shape, adapt_shape),
        }

    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting data preparation...")

        # --- Step 1: Load dataset (python) ---
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

        print(f"[{self.name}] Invoke layer 1...")

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        # plan_chain = self._make_chain("dataprep_layer1")
        # summary1 = plan_chain.run(info_dict=json.dumps(info_dict, indent=2))
        # self.memory.save_context({"input": info_dict}, {"output": summary1})

        plan_prompt = PROMPTS["dataprep_layer1"].format(info_dict=json.dumps(info_dict, indent=2))
        summary1 = self.llm(plan_prompt)
        
        print(f"[{self.name}] Invoke layer 2...")
        # --------------------
        # Layer 2: suggestions (LLM)
        # --------------------
        # adapt_chain = self._make_chain("dataprep_layer2")
        # suggestion = adapt_chain.run(summary1,
        #     info_dict=json.dumps(info_dict, indent=2),
        #     pipeline_code=open("utils/data_pipeline.py").read()
        # )
        # self.memory.save_context({"input": summary1}, {"output": suggestion})

        suggestion_prompt = PROMPTS["dataprep_layer2"].format(summary1=summary1,info_dict=json.dumps(info_dict, indent=2),pipeline_code=open("utils/data_pipeline.py").read())
        suggestion = self.llm(suggestion_prompt)

        confidence = self._extract_confidence(suggestion)
        print(f"[{self.name}] Layer 2 confidence: {confidence:.2f}")

        # === Apply deterministic pipeline
        det_pipe = DataPipeline()
        deterministic_results = det_pipe.clean(df)
        deterministic_X = deterministic_results["X_train"]
        deterministic_features = deterministic_results["feature_names"]

        # === Attempt adaptive pipeline
        try:
            adaptive_results = self._apply_llm_pipeline(df, suggestion)
            adaptive_success = True
        except Exception as e:
            adaptive_results = None
            adaptive_success = False
            print(f"[{self.name}] Adaptive pipeline failed: {e}")

        # === Compare pipelines
        comparison_summary = self._compare_pipelines(deterministic_results, adaptive_results)

        print(f"[{self.name}] Invoke layer 3...")

        # --------------------
        # Layer 3: verification (LLM)
        # --------------------
        # verify_chain = self._make_chain("dataprep_layer3")
        # verification = verify_chain.run(
        #     comparison=json.dumps(comparison_summary, indent=2),
        #     confidence=confidence,
        # )
        
        verify_prompt = PROMPTS["dataprep_layer3"].format(comparison=json.dumps(comparison_summary, indent=2),confidence=confidence)
        verification = self.llm(verify_prompt)

         # Decide based on verification judgment
        use_adaptive = "USE_ADAPTIVE" in verification.upper() and adaptive_success
        chosen_results = adaptive_results if use_adaptive else deterministic_results
        chosen_pipeline_name = "adaptive" if use_adaptive else "deterministic"

        print(f"[{self.name}] Final decision: using {chosen_pipeline_name} pipeline.")

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

        pd.DataFrame(chosen_results["X_train"], columns=chosen_results["feature_names"]).to_csv(X_train_path, index=False)
        pd.DataFrame(chosen_results["X_test"], columns=chosen_results["feature_names"]).to_csv(X_test_path, index=False)
        chosen_results["y_train"].to_csv(y_train_path, index=False)
        chosen_results["y_test"].to_csv(y_test_path, index=False)
        chosen_results["exposure_train"].to_csv(exposure_train_path, index=False)
        chosen_results["exposure_test"].to_csv(exposure_test_path, index=False)

        preproc_path = os.path.join(artifacts_dir, f"preprocessor.pkl")
        features_path = os.path.join(artifacts_dir, f"feature_names.pkl")
        joblib.dump(chosen_results["feature_names"], features_path)
        joblib.dump(det_pipe.preprocessor, preproc_path)

        print(f"[{self.name}] Invoke layer 4...")

        # --------------------
        # Layer 4: LLM inspects result
        # --------------------
        # explain_chain = self._make_chain("dataprep_layer4")
        # explanation = explain_chain.run(
        #     verification=verification
        # )

        explain_prompt = PROMPTS["dataprep_layer4"].format(verification=verification)
        explanation = self.llm(explain_prompt)

        print(f"[{self.name}] Finalize...")

        # --------------------
        # Step 5: Save metadata (keeps your original structure)
        # --------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "status": "success",
            "used_pipeline": chosen_pipeline_name,
            "confidence": confidence,
            "plan": summary1,
            "adaptive_suggestion": suggestion,
            "comparison": comparison_summary,
            "verification": verification,
            "explanation": explanation,
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

        # Log to central memory (store both deterministic summary and LLM outputs)
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

# import os
# import joblib
# import pandas as pd
# from datetime import datetime
# from utils.general_utils import save_json_safe
# from utils.prompt_library import PROMPTS
# from utils.data_pipeline import DataPipeline
# from utils.message_types import Message
# from agents.base_agent import BaseAgent

# class DataPrepAgent(BaseAgent):
#     def __init__(self, name="dataprep", shared_llm=None, system_prompt=None, hub=None):
#         super().__init__(name)
#         self.llm = shared_llm
#         self.system_prompt = system_prompt
#         self.hub = hub

#     def handle_message(self, message: Message) -> Message:
#         print(f"[{self.name}] Starting data pipeline...")

#         # --- Initiate paths ---
#         dataset_path = message.metadata.get("dataset_path", "data/raw/freMTPL2freq.csv")
#         processed_dir = "data/processed"
#         artifacts_dir = "data/artifacts"

#         os.makedirs(processed_dir, exist_ok=True)
#         os.makedirs(artifacts_dir, exist_ok=True)

#         # --- Load dataset ---
#         try:
#             data = pd.read_csv(dataset_path)
#         except Exception as e:
#             return Message(
#                 sender=self.name,
#                 recipient=message.sender,
#                 type="error",
#                 content=f"Failed to load dataset: {e}",
#             )

#         # --- Run data pipeline ---
#         pipeline = DataPipeline()
#         results = pipeline.clean(data)
#         summary_text = pipeline.summary()

#         # --- Save processed datasets ---
#         base_name = os.path.splitext(os.path.basename(dataset_path))[0]

#         X_train_path = os.path.join(processed_dir, f"{base_name}_X_train.csv")
#         X_test_path = os.path.join(processed_dir, f"{base_name}_X_test.csv")
#         y_train_path = os.path.join(processed_dir, f"{base_name}_y_train.csv")
#         y_test_path = os.path.join(processed_dir, f"{base_name}_y_test.csv")
#         exposure_train_path = os.path.join(processed_dir, f"{base_name}_exposure_train.csv")
#         exposure_test_path = os.path.join(processed_dir, f"{base_name}_exposure_test.csv")

#         # Save the split sets
#         pd.DataFrame(results["X_train"], columns=results["feature_names"]).to_csv(X_train_path, index=False)
#         pd.DataFrame(results["X_test"], columns=results["feature_names"]).to_csv(X_test_path, index=False)
#         results["y_train"].to_csv(y_train_path, index=False)
#         results["y_test"].to_csv(y_test_path, index=False)
#         results["exposure_train"].to_csv(exposure_train_path, index=False)
#         results["exposure_test"].to_csv(exposure_test_path, index=False)

#         # --- Save preprocessing artifacts ---
#         preproc_path = os.path.join(artifacts_dir, f"preprocessor.pkl")
#         features_path = os.path.join(artifacts_dir, f"feature_names.pkl")

#         joblib.dump(results["feature_names"], features_path)
#         joblib.dump(pipeline.preprocessor, preproc_path)

#         # --- LLM-generated explanation ---
#         # explain_prompt = f"""
#         # You are an AI assistant summarizing a data preprocessing pipeline for an actuarial audience.
#         # Write a clear explanation of the following cleaning steps and why they are important:
#         # {summary_text}
#         # """
#         data_prep_prompt = PROMPTS["data_prep"].format(
#         summary_text=summary_text,
#         )

#         explanation = self.llm(data_prep_prompt)

#         # --- Return message and log output ---
#         # Store metadata
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         metadata = {
#             "timestamp": timestamp,
#             "status": "success",
#             "summary_log": summary_text,
#             "llm_explanation": explanation,
#             "processed_paths": {
#                 "X_train": X_train_path,
#                 "X_test": X_test_path,
#                 "y_train": y_train_path,
#                 "y_test": y_test_path,
#                 "exposure_train": exposure_train_path,
#                 "exposure_test": exposure_test_path,
#             },
#             "artifacts": {
#                 "preprocessor": preproc_path,
#                 "feature_names": features_path,
#             },
#         }

#         results_dir = "data/results"
#         os.makedirs(results_dir, exist_ok=True)
#         meta_path = os.path.join(results_dir, f"{self.name}_metadata.json")
#         save_json_safe(metadata, meta_path)
#         metadata["metadata_file"] = meta_path

#         # Log to central memory
#         if self.hub and self.hub.memory:
#             self.hub.memory.log_event(self.name, "data_preparation", summary_text)
#             self.hub.memory.update("last_data_prep", summary_text)

#         # Return message to the hub
#         return Message(
#             sender=self.name,
#             recipient="hub",
#             type="response",
#             content="Data cleaning and preprocessing completed successfully.",
#             metadata=metadata,
#         )