import os
import re
import numpy as np
from agents.base_agent import BaseAgent
from utils.general_utils import save_json_safe
from utils.model_validation import evaluate_model_quality
from utils.message_types import Message

class ReviewingAgent(BaseAgent):
    def __init__(self, name="reviewing", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub

    def handle_message(self, message):
        print(f"[{self.name}] Reviewing model outputs and evaluation results...")

        if message.metadata is None:
            message.metadata = {}
        iteration = message.metadata.get("review_iteration", 0)

        # === Load metadata from ModellingAgent ===
        metrics = message.metadata.get("metrics", {})
        review_notes = []

        # === Step 0. Get historical memory context ===
        dataprep_context = self.hub.memory.get("dataprep_history", []) if self.hub else []
        model_context = self.hub.memory.get("model_history", []) if self.hub else []
        review_context = self.hub.memory.get("review_history", []) if self.hub else []

        # Only get last review's status & notes (instead of full history)
        if review_context and isinstance(review_context[-1], dict):
            last_review_notes = review_context[-1].get("review_notes", "No previous review notes")
        else:
            last_review_notes = "No previous review notes"

        memory_summary = {
            "dataprep_summary": dataprep_context[-1] if len(dataprep_context) > 0 else "No dataprep history",
            "model_summary": model_context[-1] if len(model_context) > 0 else "No model history",
            "past_reviews": last_review_notes,
        }

        # === Step 1. Numeric plausibility checks ===
        severity = evaluate_model_quality(metrics)
        if severity == "critical":
            review_notes.append("Critical model performance issue detected.")
        elif severity == "moderate":
            review_notes.append("Moderate performance deviation from expected range.")
        else:
            review_notes.append("Model performance within acceptable limits.")

        # === Step 1B. Numeric plausibility checks ===
        model_history = []
        if self.hub and self.hub.memory:
            model_history = self.hub.memory.get("model_history", [])

        consistency_info = "No prior model runs available."
        coef_drift = None

        if len(model_history) >= 1:
            prev_run = model_history[-1]
            previous = prev_run.get("metrics", {})
            current = metrics['Coef']

            prev_coef_raw = previous.get("Coef", None)
            df_coef_prev = prev_coef_raw
            prev_coef = dict(zip(df_coef_prev["Feature"], df_coef_prev["Coefficient"]))
            curr_coef = dict(zip(current["Feature"], current["Coefficient"]))
            
            if prev_coef and curr_coef and set(prev_coef.keys()) == set(curr_coef.keys()):
                prev_vals = np.array(list(prev_coef.values()))
                curr_vals = np.array(list(curr_coef.values()))
                coef_drift = float(np.mean(np.abs(prev_vals - curr_vals)))
                consistency_info = f"Mean coefficient drift vs previous model: {coef_drift:.6f}"
            else:
                consistency_info = "Coefficient structures are incompatible — cannot compare."
        
        print(f"[ReviewingAgent] Consistency info: {consistency_info}")

        # === Step 2. LLM reasoning ===
        review_prompt = f"""
        You are an actuarial model reviewer.
        Evaluate the following model results and provide an explicit decision line

        At the end of your response, include a line exactly in this format:
        Status: <APPROVED | NEEDS_REVISION | RETRAIN_REQUESTED>

        Metrics: {metrics}
        Numeric severity: {severity}
        Review notes: {review_notes}

        If previous memory of dataprep, modelling and reviews exist, ensure consistency with them.
        Historical memory summary:
        {memory_summary['past_reviews']}

        Consistency check for model coefficients:
        {consistency_info}

        Provide:
        1. One line starting with "Status:" (e.g., "Status: APPROVED")
        2. A short professional justification.
        """

        llm_review = self.llm(review_prompt)

        # === Step 3. Extract LLM classification ===
        match = re.search(r"Status\s*:\s*(APPROVED|NEEDS[_\s-]?REVISION|RETRAIN[_\s-]?REQUESTED)", llm_review, re.IGNORECASE)
        if match:
            label = match.group(1).upper().replace(" ", "_")
        else:
            text_section = llm_review.split("[/INST]")[-1].upper()
            if re.search(r"\bRETRAIN", text_section):
                label = "RETRAIN_REQUESTED"
            elif re.search(r"NEEDS", text_section):
                label = "NEEDS_REVISION"
            elif re.search(r"\bAPPROVED", text_section):
                label = "APPROVED"
            else:
                label = severity.upper()

        status_map = {
            "RETRAIN_REQUESTED": "retrain_requested",
            "NEEDS_REVISION": "needs_revision",
            "APPROVED": "approved",
            "CRITICAL": "retrain_requested",
            "MODERATE": "needs_revision",
            "MINOR": "approved",
        }
        status = status_map.get(label, "approved")

        # === Step 4. Optional prompt revision for retraining ===
        retrain_prompt = None
        if status == "retrain_requested":
            retrain_prompt = f"""
            The reviewing agent identified critical performance issues:
            {review_notes}.
            Please retrain the model with improved specification,
            e.g. additional interactions, nonlinear terms, or feature checks.
            """

        # Escalation logic if repeated failures
        if status == "retrain_requested":
            if iteration >= 1:
                print("[ReviewingAgent] Model still inadequate after retraining → escalating to DataPrepAgent.")
                status = "reclean_requested"
            if iteration >= 2:
                print("[ReviewingAgent] Multiple failures detected → aborting workflow.")
                status = "rejected"

        # --- Return message and log output ---
        # Store metadata
        metadata = {
            "status": status,
            "numeric_severity": severity,
            "review_notes": review_notes,
            "llm_review": llm_review,
            "retrain_prompt": retrain_prompt,
            "review_iteration": iteration + 1,
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_iteration_{iteration+1}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # === Step 6. Display summary ===
        print(f"\n--- Review Outcome ---")
        print(f"Status: {status}")
        print(f"Severity: {severity}")
        print(f"Iteration: {iteration}")
        print(f"LLM Review: {llm_review[:500]}{'...' if len(llm_review) > 500 else ''}")
        if retrain_prompt:
            print("\n--- Suggested Retrain Prompt ---")
            print(retrain_prompt)

        # === Step 7. Return structured Message ===
        next_action = {
            "approved": "proceed_to_explanation",
            "needs_revision": "proceed_to_explanation",
            "retrain_requested": "retrain_model",
            "reclean_requested": "reclean_data",    
            "rejected": "abort_workflow",           
        }[status]

        metadata["action"] = next_action

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "review_decision", metadata)
            history = self.hub.memory.get("review_history", [])
            history.append(metadata)
            self.hub.memory.update("review_history", history)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Model review completed ({status}).",
            metadata=metadata,
        )