import os
import json
from datetime import datetime
import numpy as np
import re
from langchain.memory import ConversationBufferMemory
from utils.general_utils import save_json_safe, make_json_compatible
from utils.message_types import Message
from utils.prompt_library import PROMPTS
from agents.base_agent import BaseAgent
from utils.model_evaluation import ModelEvaluation
from utils.consistency import compare_dataprep_consistency_snapshots, summarize_dataprep_snapshot_comparison, compare_modelling_consistency_snapshots, summarize_modelling_snapshot_comparison


class ReviewingAgent(BaseAgent):
    def __init__(self, name="reviewing", shared_llm=None, system_prompt=None, hub=None):
        super().__init__(name)
        self.llm = shared_llm
        self.system_prompt = system_prompt
        self.hub = hub
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=1) # Short-term conversation memory for layered prompting

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
        review_context = self.hub.memory.get("review_history", []) if self.hub else []

        # Only get last review's status & notes (instead of full history)
        if review_context and isinstance(review_context[-1], dict):
            last_review_notes = review_context[-1].get("review_notes", "No previous review notes")
        else:
            last_review_notes = "No previous review notes"
        review_memory = last_review_notes

        # --------------------
        # Get metadata for review
        # --------------------
        if phase == "dataprep":
            used_pipeline = metadata.get("used_pipeline", "N/A")
            confidence = metadata.get("confidence", "N/A")
          #  adaptive_suggestion = metadata.get("adaptive_suggestion", "N/A")
            verification = metadata.get("verification", "N/A")
        elif phase == "modelling":
            model_type_used = metadata.get("model_type_used", "N/A")
          #  model_code = metadata.get("model_code", "N/A")
            model_metrics = metadata.get("model_metrics", {})
            evaluation = metadata.get("evaluation", "N/A")
        else:
            print(f"[{self.name}] WARNING: Unknown phase '{phase}' for review agent.")

        # --------------------
        # Layer 2: analysis (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 2...")
        
        if phase == "dataprep":
            layer2_prompt = PROMPTS["review_layer2_dataprep"].format(
                phase=phase,
                layer1_out=layer1_out,
                used_pipeline=used_pipeline,
                confidence=confidence,
              #  adaptive_suggestion=adaptive_suggestion,
                verification=verification,
                review_memory=review_memory,
            )

        elif phase == "modelling":
            layer2_prompt = PROMPTS["review_layer2_modelling"].format(
                phase=phase,
                layer1_out=layer1_out,
                model_type_used=model_type_used,
              #  model_code=model_code,
                model_metrics=model_metrics,
                evaluation=evaluation,
                review_memory=review_memory,
            )

        else:
            print(f"[{self.name}] WARNING: Unknown phase '{phase}' for review agent.")

        analysis = self.llm(layer2_prompt)
        self.memory.chat_memory.add_user_message(layer2_prompt)
        self.memory.chat_memory.add_ai_message(analysis)

        print (analysis)

        # --------------------
        # Perform consistency checks
        # --------------------
        consistency_summary = ""
        if phase == "dataprep":
            snapshot = metadata.get("consistency_snapshot", "unknown")
            dataprep_snapshot_history = self.hub.memory.get("dataprep_snapshots", [])
            comparison = compare_dataprep_consistency_snapshots(snapshot, dataprep_snapshot_history)
            consistency_summary = summarize_dataprep_snapshot_comparison(comparison)
        elif phase == "modelling":
            snapshot = metadata.get("consistency_snapshot", "unknown")
            modelling_snapshot_history = self.hub.memory.get("modelling_snapshots", [])
            comparison = compare_modelling_consistency_snapshots(snapshot, modelling_snapshot_history)
            consistency_summary = summarize_modelling_snapshot_comparison(comparison)
        else:
            consistency_summary = "No dataframe available for consistency review."
        
        print(consistency_summary)

        # --------------------
        # Perform impact analysis
        # --------------------

        # PLACE HERE THE IMPACT ANALYSIS

        # --------------------
        # Layer 3: consistency checks (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...")

        layer3_prompt = PROMPTS["review_layer3"].format(
            phase=phase,
            consistency_summary=consistency_summary)
        
        consistency_check = self.llm(layer3_prompt)
        self.memory.chat_memory.add_user_message(layer3_prompt)
        self.memory.chat_memory.add_ai_message(consistency_check)

        print(consistency_check)

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
        
        # give the prompt templates to the LLM, depending on the phase
        layer5_prompt = PROMPTS["review_layer5"].format(
            phase=phase,
            analysis=analysis,
            decision=decision,
            base_prompt=PROMPTS[f"{phase}_layer1"])
        
        if decision in ["approve", "abort"]:
            print(f"[{self.name}] Skip layer 5...")
            revision_prompt = None
        else:
            print(f"[{self.name}] Invoke layer 5...")
            revision_prompt = self.llm(layer5_prompt)
        
        # --------------------
        # Save metadata
        # --------------------
        print(f"[{self.name}] Saving metadata...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_out = {
            "timestamp": timestamp,
            "phase_reviewed": phase,
            "layer1_out": layer1_out,
            "analysis": analysis,
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

        if decision == "approve" and phase == "modelling":
            modelling_snapshot_history.append(snapshot)
            self.hub.memory.update("modelling_snapshots", modelling_snapshot_history)

        # Return message to the hub
        return Message(
            sender=self.name,
            recipient="hub",
            type="response",
            content=f"Review completed. Decision: {decision}.",
            metadata=metadata_out,
        )