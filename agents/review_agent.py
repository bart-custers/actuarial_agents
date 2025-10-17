import os
import re
from datetime import datetime
from agents.base_agent import BaseAgent
from llms.wrappers import LLMWrapper
from utils.general_utils import save_json_safe
from utils.model_validation import evaluate_model_quality

class ReviewingAgent(BaseAgent):
    """
    The ReviewingAgent evaluates model metrics and explanations,
    combining numeric checks and LLM reasoning to decide if the model
    is approved, needs minor revision, or must be retrained.
    """

    def __init__(self, name="reviewing", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        # Use shared LLM if provided, else initialize local one
        if shared_llm:
            self.llm = shared_llm
        else:
            self.llm = LLMWrapper(backend="mock", system_prompt=system_prompt)
        self.system_prompt = system_prompt or "You are a critical actuarial reviewer assessing model adequacy."
        self.hub = hub

    # ------------------------------------------------------------
    # Main message handler
    # ------------------------------------------------------------
    def handle_message(self, message):
        print(f"[{self.name}] Reviewing model outputs and evaluation results...")

        if message.metadata is None:
            message.metadata = {}
            
        iteration = message.metadata.get("review_iteration", 0)
        
        # === Load metadata from ModellingAgent ===
        metrics = message.metadata.get("metrics", {})
        review_notes = []

        # === Step 1. Numeric plausibility checks ===
        severity = evaluate_model_quality(metrics)
        if severity == "critical":
            review_notes.append("Critical model performance issue detected.")
        elif severity == "moderate":
            review_notes.append("Moderate performance deviation from expected range.")
        else:
            review_notes.append("Model performance within acceptable limits.")

        # === Step 2. LLM reasoning ===
        review_prompt = f"""
        You are an actuarial model reviewer.
        Evaluate the following model results and provide an explicit decision line

        At the end of your response, include a line exactly in this format:
        Status: <APPROVED | NEEDS_REVISION | RETRAIN_REQUESTED>

        Metrics: {metrics}
        Numeric severity: {severity}
        Review notes: {review_notes}

        Provide:
        1. One line starting with "Status:" (e.g., "Status: APPROVED")
        2. A short professional justification.
        """

        llm_review = self.llm(review_prompt)

        # === Step 3. Extract LLM classification ===
        # Look for explicit "Status:" line first
        match = re.search(r"Status\s*:\s*(APPROVED|NEEDS[_\s-]?REVISION|RETRAIN[_\s-]?REQUESTED)", llm_review, re.IGNORECASE)

        if match:
            label = match.group(1).upper().replace(" ", "_")
        else:
            # fallback: look for standalone keywords, ignoring those in prompts
            text_section = llm_review.split("[/INST]")[-1].upper()  # only look after the prompt
            if re.search(r"\bRETRAIN", text_section):
                label = "RETRAIN_REQUESTED"
            elif re.search(r"NEEDS", text_section):
                label = "NEEDS_REVISION"
            elif re.search(r"\bAPPROVED", text_section):
                label = "APPROVED"
            else:
                label = severity.upper()

        # Normalize mapping
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

        if status == "retrain_requested":
            if iteration >= 1:
                print("[ReviewingAgent] Model still inadequate after retraining → escalating to DataPrepAgent.")
                status = "reclean_requested"
            if iteration >= 2:
                print("[ReviewingAgent] Multiple failures detected → aborting workflow.")
                status = "rejected"

        # === Step 5. Save metadata ===
        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)

        metadata = {
            "status": status,
            "numeric_severity": severity,
            "review_notes": review_notes,
            "llm_review": llm_review,
            "retrain_prompt": retrain_prompt,
            "review_iteration": iteration + 1,
        }

        meta_path = os.path.join(results_dir, f"review_metadata_iteration_{iteration+1}.json")
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

        return message.__class__(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Model review completed ({status}).",
            metadata=metadata,
        )




