import os
from datetime import datetime
from utils.general_utils import save_json_safe
from utils.message_types import Message
from utils.prompt_library import PROMPTS
from agents.base_agent import BaseAgent

class ExplanationAgent(BaseAgent):
    def __init__(self, name="explanation", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub
    
    def handle_message(self, message):
        print(f"[{self.name}] Explaining results and checking consistency...")

        metadata = message.metadata or {}
        model_metrics = metadata.get("metrics", {})
        review_notes = metadata.get("review_notes", [])
        llm_review = metadata.get("llm_review", "")
        predictions_path = metadata.get("predictions_path", None)

        # --- Step 0. Get historical memory context ---
        if self.hub and self.hub.memory:
            review_history = self.hub.memory.get("review_history", [])
            explanation_history = self.hub.memory.get("explanation_history", [])
        else:
            review_history, explanation_history = [], []

        # Extract only useful + small elements for prompting
        last_review = review_history[-1] if review_history else {"status": "No previous review", "review_notes": []}
        last_review_status = last_review.get('status', 'N/A')
        last_review_for_prompt = last_review.get('review_notes', [])
        last_explanation = explanation_history[-1] if explanation_history else {"consistency_summary": "No previous explanation", "belief_revision_summary": "No previous explanation"} 

        # --- Define prompt templates ---
        consistency_prompt = PROMPTS["consistency_prompt"].format(
        model_metrics=model_metrics,
        review_notes=review_notes,
        last_review_status=last_review_status,
        last_review_for_prompt=last_review_for_prompt
        )

        belief_revision_prompt = PROMPTS["belief_revision_prompt"].format(
        llm_review=llm_review,
        review_notes=review_notes,
        last_review_for_prompt=last_review_for_prompt,
        )

        # --- Run the LLM ---
        consistency_explanation = self.llm(consistency_prompt)
        belief_revision_explanation = self.llm(belief_revision_prompt)

        # --- Return message and log output ---
        # Store metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": timestamp,
            "status": "explained",
            "consistency_summary": consistency_explanation,
            "belief_revision_summary": belief_revision_explanation,
            "model_metrics": model_metrics,
            "review_notes": review_notes,
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
            f.write(consistency_explanation + "\n\n")
            f.write("--- BELIEF REVISION ---\n")
            f.write(belief_revision_explanation + "\n")

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

