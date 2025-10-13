import os
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

    def __init__(self, name="reviewing", shared_llm=None, system_prompt=None):
        super().__init__(name)
        # Use shared LLM if provided, else initialize local one
        if shared_llm:
            self.llm = shared_llm
        else:
            self.llm = LLMWrapper(backend="mock", system_prompt=system_prompt)
        self.system_prompt = system_prompt or "You are a critical actuarial reviewer assessing model adequacy."

    # ------------------------------------------------------------
    # Main message handler
    # ------------------------------------------------------------
    def handle_message(self, message):
        print(f"[{self.name}] Reviewing model outputs and evaluation results...")

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
        Evaluate the following model results and classify adequacy as one of:
        - APPROVED (no action)
        - NEEDS_REVISION (minor issues, acceptable for now)
        - RETRAIN_REQUESTED (serious issues, must be retrained)

        Metrics: {metrics}
        Numeric severity: {severity}
        Review notes: {review_notes}

        Provide:
        1. One line with your classification (e.g. "Status: APPROVED")
        2. A short professional justification.
        """

        llm_review = self.llm(review_prompt)

        # === Step 3. Extract LLM classification ===
        text_upper = llm_review.upper()
        if "RETRAIN" in text_upper:
            status = "retrain_requested"
        elif "NEEDS" in text_upper:
            status = "needs_revision"
        elif "APPROVED" in text_upper:
            status = "approved"
        else:
            # Fallback to numeric severity
            status = {
                "critical": "retrain_requested",
                "moderate": "needs_revision",
                "minor": "approved",
            }[severity]

        # === Step 4. Optional prompt revision for retraining ===
        retrain_prompt = None
        if status == "retrain_requested":
            retrain_prompt = f"""
            The reviewing agent identified critical performance issues:
            {review_notes}.
            Please retrain the model with improved specification,
            e.g. additional interactions, nonlinear terms, or feature checks.
            """

        # === Step 5. Save metadata ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)

        metadata = {
            "status": status,
            "numeric_severity": severity,
            "review_notes": review_notes,
            "llm_review": llm_review,
            "retrain_prompt": retrain_prompt,
        }

        meta_path = os.path.join(results_dir, f"review_metadata_{timestamp}.json")
        save_json_safe(metadata, meta_path)
        metadata["metadata_file"] = meta_path

        # === Step 6. Display summary ===
        print(f"\n--- Review Outcome ---")
        print(f"Status: {status}")
        print(f"Severity: {severity}")
        print(f"LLM Review: {llm_review[:500]}{'...' if len(llm_review) > 500 else ''}")
        if retrain_prompt:
            print("\n--- Suggested Retrain Prompt ---")
            print(retrain_prompt)

        # === Step 7. Return structured Message ===
        next_action = {
            "approved": "proceed_to_explanation",
            "needs_revision": "proceed_to_explanation",
            "retrain_requested": "retrain_model",
        }[status]

        metadata["action"] = next_action

        return message.__class__(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Model review completed ({status}).",
            metadata=metadata,
        )




# from llms.wrappers import LLMWrapper
# from utils.message_types import Message
# from agents.base_agent import BaseAgent
    
# class ReviewingAgent(BaseAgent):
#     def __init__(self, name="reviewing", shared_llm=None, system_prompt=None):
#         super().__init__(name)
#         self.llm = shared_llm or LLMWrapper(backend="mock", system_prompt=system_prompt)
#         self.system_prompt = system_prompt
    
#     def handle_message(self, message: Message) -> Message:
#         print(f"[{self.name}] received: {message.content}")
#         # For now: placeholder until Week 3
#         return Message(
#             sender=self.name,
#             recipient=message.sender,
#             type="response",
#             content="Reviewing agent placeholder: blabla.",
#         )



