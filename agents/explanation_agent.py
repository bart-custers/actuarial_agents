import os
import json
from datetime import datetime
import glob
from utils.general_utils import make_json_compatible
from utils.message_types import Message
from agents.base_agent import BaseAgent


class ExplanationAgent(BaseAgent):
    """
    The ExplanationAgent consolidates model evaluations, explanations, and prior
    review logs to generate a stability and consistency analysis. It uses LLM
    reasoning to check alignment between iterations and perform belief revision
    if inconsistencies are detected.
    """

    def __init__(self, name="explanation", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        if shared_llm:
            self.llm = shared_llm
        else:
            from llms.wrappers import LLMWrapper
            self.llm = LLMWrapper(backend="mock", system_prompt=system_prompt)
        self.system_prompt = (
            system_prompt
            or "You are an explanation specialist ensuring interpretability, fairness, and stability across model runs."
        )
        self.hub = hub

    def _load_previous_logs(self):
        log_dir = "data/workflow_logs"
        if not os.path.exists(log_dir):
            return []

        log_files = sorted(glob.glob(os.path.join(log_dir, "iteration_*.json")))
        logs = []
        for path in log_files[-3:]:  # only last 3 runs
            try:
                with open(path, "r") as f:
                    logs.append(json.load(f))
            except Exception:
                continue
        return logs
    
    def handle_message(self, message):
        print(f"[{self.name}] explaining results and checking consistency...")

        metadata = message.metadata or {}
        model_metrics = metadata.get("metrics", {})
        review_notes = metadata.get("review_notes", [])
        llm_review = metadata.get("llm_review", "")
        predictions_path = metadata.get("predictions_path", None)

        # === Load previous run logs for consistency context ===
        previous_logs = self._load_previous_logs()
        num_prev = len(previous_logs)

        if num_prev == 0:
            previous_summaries = []
            consistency_context = "No previous logs available. This is the first model iteration."
        elif num_prev == 1:
            previous_summaries = [previous_logs[-1].get("review_notes", [])]
            consistency_context = (
                "Only one previous run found — stability assessment will be indicative, not conclusive."
            )
        else:
            previous_summaries = [log.get("review_notes", []) for log in previous_logs]
            consistency_context = f"{num_prev} previous runs found and used for comparison."

        # === Define prompt templates ===
        consistency_prompt = f"""
        You are an actuarial explanation specialist.
        {consistency_context}

        Compare the current model's evaluation metrics and review notes
        with the summaries of previous runs below.

        Identify:
        - Whether the model is consistent with prior iterations (metrics, direction of coefficients)
        - Any drift or unexplained changes
        - Any contradictory findings or explanations.

        Current metrics:
        {model_metrics}

        Current review notes:
        {review_notes}

        Previous summaries:
        {previous_summaries}

        Provide a concise stability summary in plain English.
        """

        belief_revision_prompt = f"""
        You are a reasoning assistant performing belief revision for model interpretation.
        Given the current explanation, past review notes, and current metrics,
        update the overall understanding of the model performance and rationale.

        Ensure that your belief update resolves contradictions and forms a coherent explanation.

        Use this structure:
        1. Consistent beliefs (what remains stable)
        2. Revised beliefs (what changed and why)
        3. Remaining uncertainties

        Current LLM review:
        {llm_review}

        Review notes:
        {review_notes}

        Model metrics:
        {model_metrics}
        """

        # === Run the LLM ===
        if num_prev == 0:
            # Only generate belief revision; skip consistency check
            consistency_explanation = (
                "No prior model runs available. This is the initial iteration, "
                "so consistency cannot yet be assessed."
            )
        else:
            consistency_explanation = self.llm(consistency_prompt)

        belief_revision_explanation = self.llm(belief_revision_prompt)

        # === Create structured summaries ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/results"
        final_repo = "data/final_repository"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(final_repo, exist_ok=True)

        explanation_file = os.path.join(results_dir, f"explanation_report_{timestamp}.txt")
        with open(explanation_file, "w") as f:
            f.write("=== CONSISTENCY ANALYSIS ===\n")
            f.write(consistency_explanation + "\n\n")
            f.write("=== BELIEF REVISION ===\n")
            f.write(belief_revision_explanation + "\n")

        summary = {
            "timestamp": timestamp,
            "status": metadata.get("status", "unknown"),
            "consistency_context": consistency_context,
            "consistency_summary": consistency_explanation,
            "belief_revision_summary": belief_revision_explanation,
            "model_metrics": model_metrics,
            "review_notes": review_notes,
            "previous_runs": num_prev,
        }

        summary_file = os.path.join(results_dir, f"explanation_summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(make_json_compatible(summary), f, indent=2)

        # === Store consolidated final results ===
        bundle = {
            "predictions_path": predictions_path,
            "metrics": model_metrics,
            "review_notes": review_notes,
            "explanations": {
                "consistency": consistency_explanation,
                "belief_revision": belief_revision_explanation,
            },
            "files": {
                "report": explanation_file,
                "summary": summary_file,
            },
        }

        final_path = os.path.join(final_repo, f"final_bundle_{timestamp}.json")
        with open(final_path, "w") as f:
            json.dump(make_json_compatible(bundle), f, indent=2)

        print(f"\n--- Explanation Outputs ---")
        print(f"Report saved to: {explanation_file}")
        print(f"Summary saved to: {summary_file}")
        print(f"Final consolidated bundle: {final_path}")

        # === Return structured message ===
        metadata.update({
            "status": "explained",
            "consistency_report": explanation_file,
            "summary_file": summary_file,
            "final_bundle": final_path,
        })

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "model_explanation", summary)
            past_explanations = self.hub.memory.get("explanation_history", [])
            past_explanations.append(summary)
            self.hub.memory.update("explanation_history", past_explanations)

        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content="Generated consistency and belief-revision explanations.",
            metadata=metadata,
        )

