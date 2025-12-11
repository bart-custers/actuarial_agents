import os
import glob
import re
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

    def _extract_score(self, llm_text: str) -> str:
        text = llm_text

        # 1) Prefer explicit "Decision:" on its own line
        m = re.search(
            r'^\s*Decision\s*:\s*(NONE|MINOR|MAJOR)\s*$',
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        if m:
            decision = m.group(1).upper()
            return {"NONE": "none", "MINOR": "minor", "MAJOR": "major"}[decision]

        # 2) Tolerant match anywhere
        m2 = re.search(
            r'Decision\s*:\s*(NONE|MINOR|MAJOR)',
            text,
            flags=re.IGNORECASE
        )
        if m2:
            decision = m2.group(1).upper()
            return {"NONE": "none", "MINOR": "minor", "MAJOR": "major"}[decision]

        # Fallback
        return "none"
    
    def _extract_decision(self, llm_text: str) -> str:
        text = llm_text

        # 1) Prefer explicit "Decision:" on its own line
        m = re.search(
            r'^\s*Decision\s*:\s*(APPROVE|MINOR_ISSUES|REQUEST_RECLEAN|REQUEST_RETRAIN|ABORT)\s*$',
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        if m:
            decision = m.group(1).upper()
            return {
                "APPROVE": "approve",
                "MINOR_ISSUES": "minor_issues",
                "REQUEST_RECLEAN": "request_reclean",
                "REQUEST_RETRAIN": "request_retrain",
                "ABORT": "abort",
            }[decision]

        # 2) Tolerant match anywhere
        m2 = re.search(
            r'Decision\s*:\s*(APPROVE|MINOR_ISSUES|REQUEST_RECLEAN|REQUEST_RETRAIN|ABORT)',
            text,
            flags=re.IGNORECASE
        )
        if m2:
            decision = m2.group(1).upper()
            return {
                "APPROVE": "approve",
                "MINOR_ISSUES": "minor_issues",
                "REQUEST_RECLEAN": "request_reclean",
                "REQUEST_RETRAIN": "request_retrain",
                "ABORT": "abort",
            }[decision]

        # 3) Fallback
        return "abort"

    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message):
        print(f"[{self.name}] Explaining results...")

        metadata = message.metadata or {}
        iteration = metadata.get("explanation_iteration", 0)

        # --------------------
        # Layer 1: summarize and assess beliefs
        # --------------------        
        print(f"[{self.name}] Invoke layer 1...belief revision")

        PHASES = {
        "dataprep": "explanation",
        "modelling": "evaluation",
        "reviewing": "judgement"
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
        belief_score = self._extract_score(belief_assessment)

        print(belief_revision_prompt)
        print(belief_assessment)

        # --------------------
        # Layer 2: TCAV
        # --------------------   

        tcav_assessment = "None so far"
        tcav_score = "none"
        
        # --------------------
        # Layer 3: fairness assessment
        # --------------------  
        print(f"[{self.name}] Invoke layer 3...fairness assessment")

        df_predictions = self.load_latest_prediction_df()
        table_age, table_density = group_fairness(df_predictions, 'Prediction', 'ClaimNb', 'data/final')

        fairness_prompt = PROMPTS["fairness_prompt"].format(
            table_age=table_age,
            table_density=table_density
        )
        fairness_assessment = self.llm(fairness_prompt)
        fairness_score = self._extract_score(fairness_assessment)

        print(fairness_assessment)

        # --------------------
        # Layer 4: decision
        # --------------------  
        print(f"[{self.name}] Invoke layer 4...final assessment")

        final_prompt = PROMPTS["decision_prompt"].format(
            belief_assessment=belief_assessment,
            tcav_assessment=tcav_assessment,
            fairness_assessment=fairness_assessment,
            belief_score=belief_score,
            tcav_score=tcav_score,
            fairness_score=fairness_score
        )
        final_evaluation = self.llm(final_prompt)

        decision = self._extract_decision(final_evaluation)

        # Routing
        routing = {
            "approve": "finalize",
            "minor_issues": "consult_actuary",
            "request_reclean": "reclean_data",
            "request_retrain": "retrain_model",
            "abort": "abort_workflow"
        }
        next_action = routing.get(decision, "consult_actuary")

        print(f"[{self.name}] Decision → {decision}, Routing → {next_action}")

        # --------------------
        # Layer 5: finalize
        # --------------------  
        if decision in ["approve", "minor_issues", "abort"]:
            print(f"[{self.name}] Invoke layer 5...create final explanation report")
            report_prompt = PROMPTS["report_prompt"].format(
                final_evaluation=final_evaluation,
                decision=decision)
            recommendations = None
            final_report = self.llm(report_prompt)
        else:
            print(f"[{self.name}] Invoke layer 5...create revision recommendations")
            recommendation_prompt = PROMPTS["recommendation_prompt"].format(
                final_evaluation=final_evaluation,
                decision=decision)
            recommendations = self.llm(recommendation_prompt)
            final_report = None

        print(recommendations)

        # --------------------
        # Save metadata
        # --------------------
        print(f"[{self.name}] Saving metadata...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store review report
        #GENERATE THE REPORT HERE!!!
        #THINK ABOUT WHETHER WE WANT A FIRST PROMPT THAT RECALLS AND PLANS!!!

        # Store metadata
        metadata = {
            "timestamp": timestamp,
            "belief_assessment": belief_assessment,
            "belief_score": belief_score,
            "tcav_assessment": tcav_assessment,
            "tcav_score": tcav_score,
            "fairness_assessment": fairness_assessment,
            "fairness_score": fairness_score,
            "final_evaluation": final_evaluation,
            "decision": decision,
            "action": next_action,
            "recommendations": recommendations,
            "final_report": final_report,
            "explanation_iteration": iteration + 1
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "workflow_explanation", metadata)
            past_explanations = self.hub.memory.get("explanation_history", [])
            past_explanations.append(metadata)
            self.hub.memory.update("explanation_history", past_explanations)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content="Generated explanations. Decision: {decision}.",
            metadata=metadata,
        )

