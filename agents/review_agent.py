import os
import json
from datetime import datetime
import numpy as np
import re
from langchain.memory import ConversationBufferMemory
from utils.general_utils import save_json_safe, make_json_compatible
from utils.model_validation import evaluate_model_quality
from utils.message_types import Message
from utils.prompt_library import PROMPTS
from agents.base_agent import BaseAgent
from utils.consistency import compare_dataprep_consistency_snapshots, summarize_dataprep_snapshot_comparison



class ReviewingAgent(BaseAgent):
    def __init__(self, name="reviewing", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False) # Short-term conversation memory for layered prompting

    def _extract_decision(self, llm_text: str) -> str:
        text = llm_text.lower()

        if "approve" in text:
            return "approve"
        elif "request_reclean" in text:
            return "request_reclean"
        elif "request_retrain" in text:
            return "request_retrain"    
        elif "abort" in text:
            return "abort"
        else:
            return "abort"  # Default to abort if unclear
    
    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] Starting review pipeline...")

        metadata = message.metadata or {}
        phase = metadata.get("phase_before_review", "unknown")
        iteration = metadata.get("review_iteration", 0)

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 1...")

        layer1_prompt = PROMPTS["review_layer1"].format(phase=phase)
        layer1_out = self.llm(layer1_prompt)
        self.memory.chat_memory.add_user_message(layer1_prompt)
        self.memory.chat_memory.add_ai_message(layer1_out)

        # --------------------
        # Get context from memory
        # --------------------
        dataprep_context = self.hub.memory.get("dataprep_history", []) if self.hub else []
        model_context = self.hub.memory.get("model_history", []) if self.hub else []
        review_context = self.hub.memory.get("review_history", []) if self.hub else []

        # Only get last review's status & notes (instead of full history)
        if review_context and isinstance(review_context[-1], dict):
            last_review_notes = review_context[-1].get("review_notes", "No previous review notes")
        else:
            last_review_notes = "No previous review notes"

        # memory_summary = {
        #     "dataprep_summary": dataprep_context[-1] if len(dataprep_context) > 0 else "No dataprep history",
        #     "model_summary": model_context[-1] if len(model_context) > 0 else "No model history",
        #     "past_reviews": last_review_notes,
        # }
        review_memory = last_review_notes

        # --------------------
        # Get metadata for review
        # --------------------
        if phase == "dataprep":
            plan = metadata.get("plan", "N/A")
            used_pipeline = metadata.get("used_pipeline", "N/A")
            confidence = metadata.get("confidence", "N/A")
          #  adaptive_suggestion = metadata.get("adaptive_suggestion", "N/A")
            verification = metadata.get("verification", "N/A")
            explanation = metadata.get("explanation", "N/A")
        elif phase == "modelling":
            plan = metadata.get("plan", "N/A")
            model_type_used = metadata.get("model_type_used", "N/A")
          #  model_code = metadata.get("model_code", "N/A")
            model_metrics = metadata.get("model_metrics", {})
            explanation = metadata.get("explanation", "N/A")
        else:
            print(f"[{self.name}] WARNING: Unknown phase '{phase}' for review agent.")

        metrics_check = None
        if phase == "modelling" and "model_metrics" in metadata:
            try:
                metrics_check = evaluate_model_quality(metadata["model_metrics"])
            except Exception:
                metrics_check = None

        # --------------------
        # Layer 2: analysis (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 2...")
        
        if phase == "dataprep":
            layer2_prompt = PROMPTS["review_layer2_dataprep"].format(
                phase=phase,
                layer1_out=layer1_out,
                plan=plan,
                used_pipeline=used_pipeline,
                confidence=confidence,
              #  adaptive_suggestion=adaptive_suggestion,
                verification=verification,
                explanation=explanation,
                review_memory=review_memory,
            )

        elif phase == "modelling":
            layer2_prompt = PROMPTS["review_layer2_modelling"].format(
                phase=phase,
                layer1_out=layer1_out,
                plan=plan,
                model_type_used=model_type_used,
              #  model_code=model_code,
                model_metrics=model_metrics,
                metrics_check=metrics_check,
                explanation=explanation,
                review_memory=review_memory,
            )

        else:
            print(f"[{self.name}] WARNING: Unknown phase '{phase}' for review agent.")

        analysis = self.llm(layer2_prompt)
        self.memory.chat_memory.add_user_message(layer2_prompt)
        self.memory.chat_memory.add_ai_message(analysis)

        # --------------------
        # Perform consistency checks
        # --------------------
        consistency_summary = ""
        if phase == "dataprep":
            snapshot = metadata.get("consistency_snapshot", "unknown")
            dataprep_snapshot_history = self.hub.memory.get("dataprep_snapshots", [])
            comparison = compare_dataprep_consistency_snapshots(snapshot, dataprep_snapshot_history)
            consistency_summary = summarize_dataprep_snapshot_comparison(comparison)
        else:
            consistency_summary = "No dataframe available for consistency review."

        # --------------------
        # Layer 3: review decision (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...")

        layer3_prompt = PROMPTS["review_layer3"].format(
            phase=phase,
            consistency_summary=consistency_summary)
        
        consistency_check = self.llm(layer3_prompt)

        # --------------------
        # Layer 4: review decision (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 4...")

        layer4_prompt = PROMPTS["review_layer4"].format(
            analysis=analysis,
            consistency_check=consistency_check)
        review_output = self.llm(layer4_prompt)
        self.memory.chat_memory.add_user_message(layer4_prompt)
        self.memory.chat_memory.add_ai_message(review_output)

        print(review_output)

        # Extract decision from LLM output
        decision = self._extract_decision(review_output)

        # Routing
        routing = {
            "approve": "proceed",
            "request_reclean": "reclean_data",
            "request_retrain": "retrain_model",
            "abort": "abort_workflow"
        }
        next_action = routing.get(decision, "abort_workflow")

        print(f"[{self.name}] Decision → {decision}, Routing → {next_action}")

        # --------------------
        # Layer 5: prompt revision (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 5...")
        
        # give the prompt templates to the LLM, depending on the phase
        layer5_prompt = PROMPTS["review_layer5"].format(
            phase=phase,
            analysis=analysis,
            decision=decision,
            base_prompt=PROMPTS[f"{phase}_layer1"])
        
        if decision in ["approve", "abort"]:
            revision_prompt = None
        else:
            revision_prompt = self.llm(layer5_prompt)
        
        # --------------------
        # Save metadata
        # --------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_out = {
            "timestamp": timestamp,
            "phase_reviewed": phase,
            "layer1_out": layer1_out,
            "analysis": analysis,
            "metrics_check": metrics_check,
            "judgement": review_output,
            "decision": decision,
            "revision_prompt": revision_prompt,
            "decision": decision,
            "action": next_action,
            "review_iteration": iteration + 1
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{timestamp}.json")
        save_json_safe(metadata_out, meta_path)
        metadata_out["metadata_file"] = meta_path

        # Log to central memory
        if self.hub and self.hub.memory:
            self.hub.memory.log_event(self.name, "review_decision", metadata_out)
            history = self.hub.memory.get("review_history", [])
            history.append(metadata_out)
            self.hub.memory.update("review_history", history)
        
        if decision == "approve" and phase == "dataprep":
            dataprep_snapshot_history.append(snapshot)
            self.hub.memory.update("dataprep_snapshots", dataprep_snapshot_history)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Review completed. Decision: {decision}.",
            metadata=metadata_out,
        )


# import os
# import re
# from datetime import datetime
# import numpy as np
# from utils.general_utils import save_json_safe
# from utils.model_validation import evaluate_model_quality
# from utils.message_types import Message
# from utils.prompt_library import PROMPTS
# from agents.base_agent import BaseAgent

# class ReviewingAgent(BaseAgent):
#     def __init__(self, name="reviewing", shared_llm=None, system_prompt=None, hub=None):
#         super().__init__(name)
#         self.llm = shared_llm
#         self.system_prompt = system_prompt
#         self.hub = hub

#     def handle_message(self, message):
#         print(f"[{self.name}] Reviewing model outputs and evaluation results...")

#         if message.metadata is None:
#             message.metadata = {}
#         iteration = message.metadata.get("review_iteration", 0)

#         # --- Load metadata from ModellingAgent ---
#         metrics = message.metadata.get("metrics", {})
#         review_notes = []

#         # --- Get historical memory context ---
#         dataprep_context = self.hub.memory.get("dataprep_history", []) if self.hub else []
#         model_context = self.hub.memory.get("model_history", []) if self.hub else []
#         review_context = self.hub.memory.get("review_history", []) if self.hub else []

#         # Only get last review's status & notes (instead of full history)
#         if review_context and isinstance(review_context[-1], dict):
#             last_review_notes = review_context[-1].get("review_notes", "No previous review notes")
#         else:
#             last_review_notes = "No previous review notes"

#         memory_summary = {
#             "dataprep_summary": dataprep_context[-1] if len(dataprep_context) > 0 else "No dataprep history",
#             "model_summary": model_context[-1] if len(model_context) > 0 else "No model history",
#             "past_reviews": last_review_notes,
#         }
#         memory_for_prompt = memory_summary['past_reviews']

#         # --- Numeric plausibility checks ---
#         severity = evaluate_model_quality(metrics)
#         if severity == "critical":
#             review_notes.append("Critical model performance issue detected.")
#         elif severity == "moderate":
#             review_notes.append("Moderate performance deviation from expected range.")
#         else:
#             review_notes.append("Model performance within acceptable limits.")

#         # --- History plausibility checks ---
#         model_history = []
#         if self.hub and self.hub.memory:
#             model_history = self.hub.memory.get("model_history", [])

#         consistency_info = "No prior model runs available."
#         coef_drift = None

#         if len(model_history) >= 1:
#             prev_run = model_history[-1]
#             previous = prev_run.get("metrics", {})
#             current = metrics['Coef']

#             prev_coef_raw = previous.get("Coef", None)
#             df_coef_prev = prev_coef_raw
#             prev_coef = dict(zip(df_coef_prev["Feature"], df_coef_prev["Coefficient"]))
#             curr_coef = dict(zip(current["Feature"], current["Coefficient"]))
            
#             if prev_coef and curr_coef and set(prev_coef.keys()) == set(curr_coef.keys()):
#                 prev_vals = np.array(list(prev_coef.values()))
#                 curr_vals = np.array(list(curr_coef.values()))
#                 coef_drift = float(np.mean(np.abs(prev_vals - curr_vals)))
#                 consistency_info = f"Mean coefficient drift vs previous model: {coef_drift:.6f}"
#             else:
#                 consistency_info = "Coefficient structures are incompatible — cannot compare."
        
#         print(f"[ReviewingAgent] Consistency info: {consistency_info}")

#         # --- LLM reasoning ---
#         review_prompt = PROMPTS["review_model"].format(
#         metrics=metrics,
#         severity=severity,
#         review_notes=review_notes,
#         memory_for_prompt=memory_for_prompt,
#         consistency_info=consistency_info,
#         )

#         llm_review = self.llm(review_prompt)

#         # --- Extract LLM classification ---
#         match = re.search(r"Status\s*:\s*(APPROVED|NEEDS[_\s-]?REVISION|RETRAIN[_\s-]?REQUESTED)", llm_review, re.IGNORECASE)
#         if match:
#             label = match.group(1).upper().replace(" ", "_")
#         else:
#             text_section = llm_review.split("[/INST]")[-1].upper()
#             if re.search(r"\bRETRAIN", text_section):
#                 label = "RETRAIN_REQUESTED"
#             elif re.search(r"NEEDS", text_section):
#                 label = "NEEDS_REVISION"
#             elif re.search(r"\bAPPROVED", text_section):
#                 label = "APPROVED"
#             else:
#                 label = severity.upper()

#         status_map = {
#             "RETRAIN_REQUESTED": "retrain_requested",
#             "NEEDS_REVISION": "needs_revision",
#             "APPROVED": "approved",
#             "CRITICAL": "retrain_requested",
#             "MODERATE": "needs_revision",
#             "MINOR": "approved",
#         }
#         status = status_map.get(label, "approved")

#         # --- Optional prompt revision for retraining ---
#         retrain_prompt = None
#         if status == "retrain_requested":
#             retrain_prompt = f"""
#             The reviewing agent identified critical performance issues:
#             {review_notes}.
#             Please retrain the model with improved specification,
#             e.g. additional interactions, nonlinear terms, or feature checks.
#             """

#         # Escalation logic if repeated failures
#         if status == "retrain_requested":
#             if iteration >= 1:
#                 print("[ReviewingAgent] Model still inadequate after retraining → escalating to DataPrepAgent.")
#                 status = "reclean_requested"
#             if iteration >= 2:
#                 print("[ReviewingAgent] Multiple failures detected → aborting workflow.")
#                 status = "rejected"
        
#         # --- Display summary ---
#         print(f"\n--- Review Outcome ---")
#         print(f"Status: {status}")
#         print(f"Severity: {severity}")
#         print(f"Iteration: {iteration}")
#         print(f"LLM Review: {llm_review[:500]}{'...' if len(llm_review) > 500 else ''}")
#         if retrain_prompt:
#             print("\n--- Suggested Retrain Prompt ---")
#             print(retrain_prompt)

#         # --- Return structured Message ---
#         next_action = {
#             "approved": "proceed_to_explanation",
#             "needs_revision": "proceed_to_explanation",
#             "retrain_requested": "retrain_model",
#             "reclean_requested": "reclean_data",    
#             "rejected": "abort_workflow",           
#         }[status]

#         # --- Return message and log output ---
#         # Store metadata
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         metadata = {
#             "timestamp": timestamp,
#             "status": status,
#             "numeric_severity": severity,
#             "review_notes": review_notes,
#             "llm_review": llm_review,
#             "retrain_prompt": retrain_prompt,
#             "review_iteration": iteration + 1,
#             "action": next_action,
#         }

#         results_dir = "data/results"
#         os.makedirs(results_dir, exist_ok=True)
#         meta_path = os.path.join(results_dir, f"{self.name}_metadata_iteration_{iteration+1}.json")
#         save_json_safe(metadata, meta_path)
#         metadata["metadata_file"] = meta_path

#         # Log to central memory
#         if self.hub and self.hub.memory:
#             self.hub.memory.log_event(self.name, "review_decision", metadata)
#             history = self.hub.memory.get("review_history", [])
#             history.append(metadata)
#             self.hub.memory.update("review_history", history)

#         # Return message to the hub
#         return Message(
#             sender=self.name,
#             recipient="hub",
#             type="response",
#             content=f"Model review completed ({status}).",
#             metadata=metadata,
#         )