# agents/data_prep_agent.py
import os
import json
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from utils.general_utils import save_json_safe
from utils.prompt_library import PROMPTS
from utils.data_pipeline import DataPipeline
from utils.message_types import Message
from agents.base_agent import BaseAgent

# ---------------------------
# Prompt templates (LangChain)
# ---------------------------

LAYER1_PROMPT = PromptTemplate(
    input_variables=["dataset_name", "n_rows", "n_cols", "sample_head", "assignment_desc"],
    template="""
You are an actuarial data-prep assistant.

Layer 1 — Task recall & initial plan:
Dataset: {dataset_name}
Rows: {n_rows}, Columns: {n_cols}
Sample head:
{sample_head}

Assignment / goal:
{assignment_desc}

1) Briefly restate the assignment in one sentence.
2) List the top 6 actions you think are most important to prepare this dataset for claim frequency modelling (short bullet list).
3) Provide any immediate warnings (e.g., very skewed numeric columns, too many missing values).
Respond concisely.
""",
)

LAYER2_PROMPT = PromptTemplate(
    input_variables=["summary1", "missing_dict", "numerical", "categorical", "pipeline_description"],
    template="""
You are an actuarial data-prep assistant.

Layer 2 — Suggest safe preprocessing adjustments.

Context (summary of your earlier recommendations):
{summary1}

Deterministic diagnostics (JSON):
Missing counts: {missing_dict}
Numerical variables: {numerical}
Categorical variables: {categorical}

Current deterministic pipeline (short description):
{pipeline_description}

Please produce a JSON object (only JSON) with the following optional keys:
- "drop_columns": [list of column names to drop]
- "impute_numeric": { "colA": "median"|"mean"|"zero"|"constant", ... }
- "impute_categorical": { "colB": "mode"|"unknown"|"constant", ... }
- "encode_categorical": [list of categorical columns to one-hot encode (optional)]
- "additional_transform": [brief text descriptions, not code]

Example (JSON):
{
  "drop_columns": ["colX"],
  "impute_numeric": {"VehAge": "median"},
  "impute_categorical": {"Region": "mode"},
  "encode_categorical": ["VehBrand"],
  "additional_transform": ["clip VehAge at 20"]
}

Return only valid JSON. If nothing to suggest, return {}
""",
)

LAYER3_PROMPT = PromptTemplate(
    input_variables=["pipeline_result_summary", "num_rows", "num_cols"],
    template="""
You are an actuarial data-prep assistant.

Layer 3 — Inspect result:
We applied the preprocessing steps and produced a cleaned dataset.

Summary of the cleaned dataset (text):
{pipeline_result_summary}

Resulting shape: rows={num_rows}, cols={num_cols}

Please:
1) Provide a short diagnostic (2-4 bullets) about whether the cleaned data looks suitable for modelling.
2) If you see remaining issues, provide short bullet list of suggestions.
Return plain text.
""",
)

# ---------------------------
# Agent
# ---------------------------

class DataPrepAgent(BaseAgent):
    def __init__(self, name="dataprep", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        # shared_llm should be a LangChain-compatible LLM or wrapper exposing __call__(str)->str
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub

        # Short-term conversation memory for layered prompting
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

        # Setup LLMChains
        # If self.llm is a LangChain LLM implementor (e.g., ChatOpenAI), we can pass it to LLMChain.
        # If self.llm is a callable wrapper (returns string), we will call it directly as fallback.
        try:
            self.chain_layer1 = LLMChain(llm=self.llm, prompt=LAYER1_PROMPT, memory=self.memory)
            self.chain_layer2 = LLMChain(llm=self.llm, prompt=LAYER2_PROMPT, memory=self.memory)
            self.chain_layer3 = LLMChain(llm=self.llm, prompt=LAYER3_PROMPT, memory=self.memory)
            self.use_chains = True
        except Exception:
            # Fallback: call llm directly with formatted prompt texts
            self.chain_layer1 = None
            self.chain_layer2 = None
            self.chain_layer3 = None
            self.use_chains = False

    # ---------------------------
    # Helper: safe parse JSON from LLM
    # ---------------------------
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Try to find and parse the first JSON object in text. Return {} if failed."""
        # Try direct load
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # Try to extract a {...} substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(text[start:end+1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {}

    # ---------------------------
    # Helper: apply safe suggestions to DataFrame
    # ---------------------------
    def _apply_suggestions(self, df: pd.DataFrame, suggestions: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        # 1) drop_columns
        drop_cols = suggestions.get("drop_columns", []) or []
        drop_cols = [c for c in drop_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # 2) impute_numeric
        for col, strategy in (suggestions.get("impute_numeric") or {}).items():
            if col in df.columns:
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)
                elif strategy == "constant":
                    # constant value could be specified as a tuple or string "constant:val"
                    df[col] = df[col].fillna(0)
                else:
                    # fallback to median
                    df[col] = df[col].fillna(df[col].median())

        # 3) impute_categorical
        for col, strategy in (suggestions.get("impute_categorical") or {}).items():
            if col in df.columns:
                if strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "UNKNOWN")
                elif strategy == "unknown":
                    df[col] = df[col].fillna("UNKNOWN")
                elif strategy == "constant":
                    df[col] = df[col].fillna("UNKNOWN")
                else:
                    df[col] = df[col].fillna("UNKNOWN")

        # 4) additional_transform (not executed as code, only warn/log)
        # We don't execute arbitrary python from LLM — just log suggestions.
        return df

    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting layered data-prep with deterministic pipeline...")

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

        # Basic dataset descriptors
        dataset_name = os.path.basename(dataset_path)
        n_rows, n_cols = df.shape
        sample_head = df.head(5).to_csv(index=False)

        # Assignment description (from prompts library or message)
        assignment_desc = message.metadata.get("assignment_desc", PROMPTS.get("data_prep_brief", "Prepare dataset for claim frequency GLM modelling."))

        print(f"[{self.name}] Invoke layer 1...")
        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        layer1_inputs = {
            "dataset_name": dataset_name,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "sample_head": sample_head,
            "assignment_desc": assignment_desc,
        }

        if self.use_chains:
            try:
                summary1 = self.chain_layer1.run(layer1_inputs)
            except Exception as e:
                summary1 = self.llm(LAYER1_PROMPT.format(**layer1_inputs)) if callable(self.llm) else str(e)
        else:
            prompt1 = LAYER1_PROMPT.format(**layer1_inputs)
            summary1 = self.llm(prompt1) if callable(self.llm) else prompt1
        print(f"[{self.name}] Starting step 2...")
        # --------------------
        # Step 2: deterministic diagnostics (python)
        # --------------------
        missing_dict = df.isna().sum().to_dict()
        # numerical and categorical heuristics: infer by dtype and small cardinality
        numerical = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # small-cardinality numeric columns may be categorical (coerce heuristic)
        for col in numerical[:]:
            if df[col].nunique() <= 10:
                # keep as numeric but note it
                pass

        # pipeline description — short human-readable description of your deterministic pipeline
        pipeline_description = (
            "Right-censor some vars, clip extreme values, map Area categories, drop NA, "
            "define numerical & categorical features, split train/test, scale numerical features and one-hot encode categoricals."
        )
        print(f"[{self.name}] Starting layer 2...")
        # --------------------
        # Layer 2: suggestions (LLM) — expects JSON
        # --------------------
        layer2_inputs = {
            "summary1": summary1,
            "missing_dict": json.dumps(missing_dict),
            "numerical": json.dumps(numerical),
            "categorical": json.dumps(categorical),
            "pipeline_description": pipeline_description,
        }

        if self.use_chains:
            try:
                layer2_raw = self.chain_layer2.run(layer2_inputs)
            except Exception:
                # fallback to direct call
                layer2_prompt_text = LAYER2_PROMPT.format(**layer2_inputs)
                layer2_raw = self.llm(layer2_prompt_text) if callable(self.llm) else "{}"
        else:
            layer2_prompt_text = LAYER2_PROMPT.format(**layer2_inputs)
            layer2_raw = self.llm(layer2_prompt_text) if callable(self.llm) else "{}"

        # parse suggested JSON
        suggestions = self._parse_json(layer2_raw)
        # sanitize suggestions -> only keep allowed keys and known columns
        allowed_keys = {"drop_columns", "impute_numeric", "impute_categorical", "encode_categorical", "additional_transform"}
        suggestions = {k: v for k, v in suggestions.items() if k in allowed_keys}

        # Validate column names
        if "drop_columns" in suggestions:
            suggestions["drop_columns"] = [c for c in suggestions["drop_columns"] if c in df.columns]

        if "impute_numeric" in suggestions:
            suggestions["impute_numeric"] = {k: v for k, v in suggestions["impute_numeric"].items() if k in df.columns}

        if "impute_categorical" in suggestions:
            suggestions["impute_categorical"] = {k: v for k, v in suggestions["impute_categorical"].items() if k in df.columns}

        if "encode_categorical" in suggestions:
            suggestions["encode_categorical"] = [c for c in suggestions["encode_categorical"] if c in df.columns]
        print(f"[{self.name}] Apply suggestions...")
        # --------------------
        # Step 3: Apply deterministic modifications per suggestions
        # --------------------
        df_mod = self._apply_suggestions(df, suggestions)
        print(f"[{self.name}] Run deterministic data pipeline...")
        # --------------------
        # Step 4: Run the deterministic pipeline (DataPipeline)
        #   - we feed the modified DataFrame to the same pipeline.clean(...)
        # --------------------
        pipeline = DataPipeline()
        try:
            results = pipeline.clean(df_mod)
            pipeline_summary = pipeline.summary()
        except Exception as e:
            return Message(
                sender=self.name,
                recipient=message.sender,
                type="error",
                content=f"Data pipeline failed: {e}",
            )

        # Save processed datasets (existing behavior)
        processed_dir = "data/processed"
        artifacts_dir = "data/artifacts"
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        X_train_path = os.path.join(processed_dir, f"{base_name}_X_train_{timestamp}.csv")
        X_test_path = os.path.join(processed_dir, f"{base_name}_X_test_{timestamp}.csv")
        y_train_path = os.path.join(processed_dir, f"{base_name}_y_train_{timestamp}.csv")
        y_test_path = os.path.join(processed_dir, f"{base_name}_y_test_{timestamp}.csv")
        exposure_train_path = os.path.join(processed_dir, f"{base_name}_exposure_train_{timestamp}.csv")
        exposure_test_path = os.path.join(processed_dir, f"{base_name}_exposure_test_{timestamp}.csv")

        pd.DataFrame(results["X_train"], columns=results["feature_names"]).to_csv(X_train_path, index=False)
        pd.DataFrame(results["X_test"], columns=results["feature_names"]).to_csv(X_test_path, index=False)
        results["y_train"].to_csv(y_train_path, index=False)
        results["y_test"].to_csv(y_test_path, index=False)
        results["exposure_train"].to_csv(exposure_train_path, index=False)
        results["exposure_test"].to_csv(exposure_test_path, index=False)

        # Save preprocessor and features
        preproc_path = os.path.join(artifacts_dir, f"preprocessor_{timestamp}.pkl")
        features_path = os.path.join(artifacts_dir, f"feature_names_{timestamp}.pkl")
        joblib.dump(results["feature_names"], features_path)
        joblib.dump(pipeline.preprocessor, preproc_path)
        print(f"[{self.name}] Starting layer 3...")
        # --------------------
        # Layer 3: LLM inspects result
        # --------------------
        pipeline_result_summary = pipeline_summary
        layer3_inputs = {
            "pipeline_result_summary": pipeline_result_summary,
            "num_rows": results.get("X_train").shape[0] if hasattr(results.get("X_train"), "shape") else len(results.get("X_train", [])),
            "num_cols": len(results.get("feature_names", [])),
        }

        if self.use_chains:
            try:
                layer3_text = self.chain_layer3.run(layer3_inputs)
            except Exception:
                layer3_prompt_text = LAYER3_PROMPT.format(**layer3_inputs)
                layer3_text = self.llm(layer3_prompt_text) if callable(self.llm) else ""
        else:
            layer3_prompt_text = LAYER3_PROMPT.format(**layer3_inputs)
            layer3_text = self.llm(layer3_prompt_text) if callable(self.llm) else ""
        print(f"[{self.name}] Finalize...")
        # --------------------
        # Step 5: Save metadata (keeps your original structure)
        # --------------------
        metadata = {
            "timestamp": timestamp,
            "status": "success",
            "summary_log": pipeline_summary,
            "llm_explanation_layer1": summary1,
            "llm_suggestions_layer2_raw": layer2_raw,
            "llm_suggestions_layer2": suggestions,
            "llm_explanation_layer3": layer3_text,
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
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory (store both deterministic summary and LLM outputs)
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "data_preparation", metadata)
            # also update last_data_prep to a compact summary
            self.hub.memory.update("last_data_prep", {
                "timestamp": timestamp,
                "summary": pipeline_summary,
                "suggestions": suggestions,
            })

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