import os
import glob
import pandas as pd
from datetime import datetime
from utils.general_utils import save_json_safe
from utils.fairness import group_fairness
from utils.message_types import Message
from utils.prompt_library import PROMPTS
from agents.base_agent import BaseAgent


class ExplanationAgent(BaseAgent):
    def __init__(self, name="explanation", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub
    
    @staticmethod
    def load_latest_prediction_df(folder="data/final"):
        # Find all prediction files
        df_files = glob.glob(os.path.join(folder, "df_predictions_*.csv"))

        # Check that files exist
        if not df_files:
            raise FileNotFoundError(f"No prediction df found in folder: {folder}")

        # Sort files by date descending (latest first)
        df_files.sort(reverse=True)

        # Take the latest file
        latest_df_file = df_files[0]

        latest_df_preds = pd.read_csv(latest_df_file)

        return latest_df_preds 

    
    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message):
        print(f"[{self.name}] Explaining results...")

        metadata = message.metadata or {}

        # --------------------
        # Layer 1: summarize and assess beliefs
        # --------------------        
        print(f"[{self.name}] Invoke layer 1...belief revision")

        PHASES = {
        "dataprep": ["verification", "explanation"],
        "modelling": ["evaluation", "impact_analysis"],
        "reviewing": ["analysis", "consistency_check", "judgement"]
        }

        belief_state = {}

        for phase, keys in PHASES.items():
            # Extract metadata items for the phase
            items = [metadata.get(key, {}) for key in keys]

            # Build the phase prompt using the standard template
            summary_prompt = PROMPTS["summary_prompt"].format(
                item1=items[0] if len(items) > 0 else None,
                item2=items[1] if len(items) > 1 else None,
                item3=items[2] if len(items) > 2 else None,
            )

            # Query the LLM
            belief_state[phase] = self.llm(summary_prompt)
        
        belief_dir = "data/memory"
        os.makedirs(belief_dir, exist_ok=True)
        meta_path = os.path.join(belief_dir, f"{self.name}_belief_state.json")
        save_json_safe(belief_state, meta_path)

        # Now assess the belief
        belief_revision_prompt = PROMPTS["belief_revision_prompt"].format(belief_summary = belief_state)
        belief_assessment = self.llm(belief_revision_prompt)

        # --------------------
        # Layer 2: TCAV
        # --------------------   


        
        # --------------------
        # Layer 3: fairness assessment
        # --------------------  
        print(f"[{self.name}] Invoke layer 3...fairness assessment")

        df_predictions = self.load_latest_prediction_df()
        table_age, table_density = group_fairness(df_predictions, 'Prediction', 'ClaimNb', 'data/final')

        #call prompt

        # 
        # 
        # 
        # --------------------
        # Layer 4: end report
        # --------------------  



        # --- Return message and log output ---
        # Store metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "status": "explained",

        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Store explanation report
        explanation_file = os.path.join(results_dir, f"explanation_report_{timestamp}.txt")
        with open(explanation_file, "w") as f:
            f.write("--- CONSISTENCY ANALYSIS ---\n")
            f.write("--- BELIEF REVISION ---\n")

        print(f"\n--- Explanation Outputs ---")
        print(f"Report saved to: {explanation_file}")

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "model_explanation", metadata)
            past_explanations = self.hub.memory.get("explanation_history", [])
            past_explanations.append(metadata)
            self.hub.memory.update("explanation_history", past_explanations)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content="Generated explanations.",
            metadata=metadata,
        )

