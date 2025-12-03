import os
import json
from datetime import datetime
import numpy as np
import re
#from langchain_community.memory import ConversationBufferMemory
from utils.general_utils import save_json_safe, extract_analysis, generate_review_report_txt
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
      #  self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=1) # Short-term conversation memory for layered prompting

    # def _extract_decision(self, llm_text: str) -> str:
    #     text = llm_text.lower()

    #     if "approve" in text:
    #         return "approve"
    #     elif "request_reclean" in text:
    #         return "request_reclean"
    #     elif "request_retrain" in text:
    #         return "request_retrain"    
    #     elif "abort" in text:
    #         return "abort"
    #     else:
    #         return "abort"
    
    def _extract_decision(self, llm_text: str) -> str:
        text = llm_text

        # 1) Prefer explicit "Decision:" on its own line
        m = re.search(
            r'^\s*Decision\s*:\s*(APPROVE|REQUEST_RECLEAN|REQUEST_RETRAIN|ABORT)\s*$',
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        if m:
            decision = m.group(1).upper()
            return {
                "APPROVE": "approve",
                "REQUEST_RECLEAN": "request_reclean",
                "REQUEST_RETRAIN": "request_retrain",
                "ABORT": "abort",
            }[decision]

        # 2) Tolerant match anywhere
        m2 = re.search(
            r'Decision\s*:\s*(APPROVE|REQUEST_RECLEAN|REQUEST_RETRAIN|ABORT)',
            text,
            flags=re.IGNORECASE
        )
        if m2:
            decision = m2.group(1).upper()
            return {
                "APPROVE": "approve",
                "REQUEST_RECLEAN": "request_reclean",
                "REQUEST_RETRAIN": "request_retrain",
                "ABORT": "abort",
            }[decision]

        # 3) Fallback
        return "abort"
    
    # ---------------------------
    # Main handler
    # ---------------------------
    def handle_message(self, message: Message) -> Message:
        
        metadata = message.metadata or {}
        phase = metadata.get("phase_before_review", "unknown")
        iteration = metadata.get("review_iteration", 0)
        print(f"[{self.name}] Starting {phase} review iteration {iteration}...")

        # --------------------
        # Layer 1: recall & plan (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 1...planning")

        layer1_prompt = PROMPTS["review_layer1"].format(phase=phase)
        layer1_out = self.llm(layer1_prompt)

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
        print(f"[{self.name}] Invoke layer 2...result analysis")
        
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
              #  model_metrics=model_metrics,
                evaluation=evaluation,
                review_memory=review_memory,
            )

        else:
            print(f"[{self.name}] WARNING: Unknown phase '{phase}' for review agent.")

        analysis = self.llm(layer2_prompt)
        analysis_text = extract_analysis(analysis)

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
        
        # --------------------
        # Layer 3: consistency checks (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 3...consistency check")

        layer3_prompt = PROMPTS["review_layer3"].format(
            phase=phase,
            consistency_summary=consistency_summary)
        
        consistency_check = self.llm(layer3_prompt)
        consistency_text = extract_analysis(consistency_check)

        # --------------------
        # Layer 4: impact analysis (LLM)
        # --------------------

        if phase == "dataprep":
            print(f"[{self.name}] No impact analysis - skip layer 4...")
            impact_analysis_output = "No impact analysis for dataprep"
            impact_text = "No impact analysis for dataprep"
        elif phase == "modelling":
            print(f"[{self.name}] Invoke layer 4...impact analysis")
            impact_analysis_input = metadata.get("impact_analysis", "unknown")
            layer4_prompt = PROMPTS["review_layer4"].format(
            impact_analysis_input=impact_analysis_input)
            impact_analysis_output = self.llm(layer4_prompt)
            impact_text = extract_analysis(impact_analysis_output)

        # --------------------
        # Layer 5: review decision (LLM)
        # --------------------
        print(f"[{self.name}] Invoke layer 5...review decision")

        layer5_prompt = PROMPTS["review_layer5"].format(
            analysis=analysis_text,
            consistency_check=consistency_text,
            impact_analysis_output=impact_text)
        
        print(layer5_prompt)

        review_output = self.llm(layer5_prompt)

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
        # Layer 6: prompt revision (LLM)
        # --------------------       
        if decision in ["approve", "abort"]:
            print(f"[{self.name}] Invoke layer 6...create final review report")
            revision_prompt = None
            layer6_prompt = PROMPTS["review_layer6"]
            final_report = self.llm(layer6_prompt)
        else:
            print(f"[{self.name}] Invoke layer 6...create revision prompt")
            layer6_prompt = PROMPTS["review_revision"].format(
            phase=phase,
            analysis=analysis,
            decision=decision,
            base_prompt=PROMPTS[f"{phase}_layer1"])
            final_report = None
            revision_prompt = self.llm(layer6_prompt)
        
        # --------------------
        # Save metadata
        # --------------------
        print(f"[{self.name}] Saving metadata...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store review report
        report_path = f"data/final/review_report_{timestamp}.txt"
        if phase == "dataprep":
            model_metrics = None
        generate_review_report_txt(report_path, phase, model_metrics, analysis, consistency_summary, consistency_check, impact_analysis_output,
                               review_output, final_report)

        # Store metadata
        metadata_out = {
            "timestamp": timestamp,
            "phase_reviewed": phase,
            "layer1_out": layer1_out,
            "analysis": analysis_text,
            "consistency_summary": consistency_summary,
            "consistency_check": consistency_text,
            "impact_analysis_output": impact_text,
            "judgement": review_output,
            "decision": decision,
            "revision_prompt": revision_prompt,
            "decision": decision,
            "action": next_action,
            "review_iteration": iteration + 1
        }

        results_dir = "data/results"
        os.makedirs(results_dir, exist_ok=True)
        meta_path = os.path.join(results_dir, f"{self.name}_metadata_{phase}_{timestamp}.json")
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