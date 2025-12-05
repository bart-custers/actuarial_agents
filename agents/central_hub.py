import os
import json
import time
from datetime import datetime
from utils.message_types import Message
from utils.general_utils import make_json_compatible
from utils.performance import WorkflowAudit
from utils.central_memory import CentralMemory
from llms.wrappers import LLMWrapper
from agents.data_prep_agent import DataPrepAgent
from agents.modelling_agent import ModellingAgent
from agents.review_agent import ReviewingAgent
from agents.explanation_agent import ExplanationAgent

class CentralHub:  
    def __init__(self, backend="llama7b"):
        self.memory = CentralMemory("data/memory/central_memory.json")

        # Shared LLM for all agents
        self.shared_llm = LLMWrapper(
            backend=backend,
            system_prompt="You are a helpful actuarial assistant that can handle multiple reasoning tasks.",
        )

        # Agents
        self.agents = {
            "dataprep": DataPrepAgent(
                "dataprep",
                shared_llm=self.shared_llm,
                system_prompt="You are a meticulous data-preparation specialist who cleans insurance datasets clearly and reproducibly.",
                hub=self
            ),
            "modelling": ModellingAgent(
                "modelling",
                shared_llm=self.shared_llm,
                system_prompt="You are an actuarial modeller who builds and evaluates predictive models precisely.",
                hub=self
            ),
            "reviewing": ReviewingAgent(
                "reviewing",
                shared_llm=self.shared_llm,
                system_prompt="You are a critical reviewer checking model plausibility and regulatory compliance.",
                hub=self
            ),
            "explanation": ExplanationAgent(
                "explanation",
                shared_llm=self.shared_llm,
                system_prompt="You are an explanation specialist ensuring interpretability and fairness.",
                hub=self
            ),
        }

    def send(self, message):
        recipient = message.recipient
        if recipient not in self.agents:
            raise ValueError(f"Unknown recipient: {recipient}")

        print(f"[Hub] routing message from {message.sender} to {recipient}")
        response = self.agents[recipient].handle_message(message)
        return response

    def run_workflow(self):
        print("\n===== STARTING ACTUARIAL AGENTS WORKFLOW =====\n")

        # ------------------------------------------------------------------
        # INITIAL STATE
        # ------------------------------------------------------------------
        current_metadata = {"dataset_path": "data/raw/freMTPL2freq.csv"}
        phase = "dataprep"
        iteration = 0
        continue_workflow = True
        MAX_ITERATIONS = 4

        log_dir = "data/workflow_logs"
        os.makedirs(log_dir, exist_ok=True)

        audit = WorkflowAudit(log_dir="data/audit")

        r1 = r2 = r3 = r4 = None
        summary_records = []
        start_time = time.time()

        # ------------------------------------------------------------------
        # MAIN LOOP
        # ------------------------------------------------------------------
        while continue_workflow:
            print(f"\n=== Iteration {iteration} | Phase: {phase.upper()} ===\n")
            completed_phase = phase  # used for logging filename

            # --------------------------------------------------------------
            # DATA PREPARATION
            # --------------------------------------------------------------
            if phase == "dataprep":
                msg = Message(
                    sender="hub",
                    recipient="dataprep",
                    type="task",
                    content="Clean dataset and summarize.",
                    metadata=current_metadata,
                )
                r1 = self.send(msg)
                current_metadata.update(r1.metadata or {})

                print("\n--- Data Preparation Completed ---")
                print(r1.content)

                # Validation
                phase = "reviewing"
                current_metadata["phase_before_review"] = "dataprep"

                audit.record_event("dataprep", iteration, "data_cleaning",
                                   current_metadata, sent=msg, received=r1)

            # --------------------------------------------------------------
            # MODELLING
            # --------------------------------------------------------------
            elif phase == "modelling":
                msg = Message(
                    sender="hub",
                    recipient="modelling",
                    type="task",
                    content=f"Train predictive model (iteration {iteration}).",
                    metadata=current_metadata,
                )
                r2 = self.send(msg)
                current_metadata.update(r2.metadata or {})

                print("\n--- Modelling Completed ---")
                print(r2.content)
                
                #Validation
                phase = "reviewing"
                current_metadata["phase_before_review"] = "modelling"

                audit.record_event("modelling", iteration, "model_training",
                                   current_metadata, sent=msg, received=r2)

            # --------------------------------------------------------------
            # REVIEWING
            # --------------------------------------------------------------
            elif phase == "reviewing":

                reviewing_task = (
                    "Review model outputs (iteration {})".format(iteration)
                    if current_metadata["phase_before_review"] == "modelling"
                    else "Review data-preparation quality (iteration {})".format(iteration)
                )

                msg = Message(
                    sender="hub",
                    recipient="reviewing",
                    type="task",
                    content=reviewing_task,
                    metadata=current_metadata,
                )
                r3 = self.send(msg)
                current_metadata.update(r3.metadata or {})

                print("\n--- Review Completed ---")
                print(r3.content)

                action = current_metadata.get("action", "proceed_to_explanation")
                print(f"[Hub] Review decision → {action.upper()}")

                audit.record_event("reviewing", iteration, action,
                                   current_metadata, sent=msg, received=r3)

                # ========= ROUTE BASED ON DECISION =========

                prev_phase = current_metadata.get("phase_before_review")

                # --- APPROVED ---
                if action == "proceed":
                    if prev_phase == "dataprep":
                        print("[Hub] Dataprep approved → proceeding to modelling.\n")
                        phase = "modelling"
                    elif prev_phase == "modelling":
                        print("[Hub] Modelling approved → proceeding to explanation.\n")
                        phase = "explanation"
                
                # --- RECLEAN DATA ---
                elif action == "reclean_data":
                    print("[Hub] Reviewer requests data cleaning → restarting dataprep.\n")
                    iteration += 1
                    if iteration >= MAX_ITERATIONS:
                        print("Maximum iterations reached — aborting workflow.")
                        current_metadata["status"] = "terminated"
                        continue_workflow = False
                        break

                    phase = "dataprep"
                    current_metadata["revised_prompt"] = current_metadata.get("revision_prompt")
                    continue

                # --- RETRAIN MODEL ---
                elif action == "retrain_model":
                    print("[Hub] Reviewer requests model retraining → restarting modelling.\n")
                    iteration += 1
                    if iteration >= MAX_ITERATIONS:
                        print("Maximum retraining iterations reached — aborting workflow.")
                        current_metadata["status"] = "terminated"
                        continue_workflow = False
                        break

                    phase = "modelling"
                    current_metadata["revised_prompt"] = current_metadata.get("revision_prompt")
                    continue

                # --- ABORT ---
                elif action == "abort_workflow":
                    print("Error detected — aborting workflow.")
                    current_metadata["status"] = "terminated"
                    continue_workflow = False

                # --- FALLBACK ---
                else:
                    print(f"[Hub] Unknown review action '{action}' — defaulting to explanation.")
                    phase = "explanation"

            # --------------------------------------------------------------
            # EXPLANATION
            # --------------------------------------------------------------
            elif phase == "explanation":
                msg = Message(
                    sender="hub",
                    recipient="explanation",
                    type="task",
                    content=f"Generate workflow explanations (iteration {iteration}).",
                    metadata=current_metadata,
                )
                r4 = self.send(msg)
                current_metadata.update(r4.metadata or {})
                print("\n--- Explanation Completed ---")
                print(r4.content)

                explanation_action = current_metadata.get("action", "finalize")
                print(f"[Hub] Explanation decision → {explanation_action.upper()}")

                audit.record_event("explanation", iteration, explanation_action,
                                   current_metadata, sent=msg, received=r4)

                # ========= ROUTE BASED ON DECISION =========

                # --- APPROVED ---
                if explanation_action == "finalize":
                    print("[Hub] Explanation approves model → finalizing workflow.\n")
                    continue_workflow = False
                    break

                # --- APPROVED BUT CONSULT ACTUARY ---
                elif explanation_action == "consult_actuary":
                    print("[Hub] Explanation detected minor issues → finalizing workflow but please consult an actuary.\n")
                    continue_workflow = False
                    break

                # --- RECLEAN DATA ---
                elif explanation_action == "reclean_data":
                    print("[Hub] Explanation requests data cleaning → restarting dataprep.\n")
                    iteration += 1
                    if iteration >= MAX_ITERATIONS:
                        print("Maximum iterations reached — aborting workflow.")
                        current_metadata["status"] = "terminated"
                        continue_workflow = False
                        break
                    phase = "dataprep"
                    current_metadata["recommendations"] = current_metadata.get("recommendations")
                    continue

                # --- RETRAIN MODEL ---
                elif explanation_action == "retrain_model":
                    print("[Hub] Explanation requests model retraining → restarting modelling.\n")
                    iteration += 1
                    if iteration >= MAX_ITERATIONS:
                        print("Maximum iterations reached — aborting workflow.")
                        current_metadata["status"] = "terminated"
                        continue_workflow = False
                        break
                    phase = "modelling"
                    current_metadata["recommendations"] = current_metadata.get("recommendations")
                    continue

                # --- ABORT ---
                elif explanation_action == "abort_workflow":
                    print("Error detected — aborting workflow.")
                    current_metadata["status"] = "terminated"
                    continue_workflow = False
                    break

                # --- FALLBACK ---
                else:
                    print(f"[Hub] Unknown explanation action '{action}' — defaulting to abort workflow.")
                    continue_workflow = False
                    break

            # --------------------------------------------------------------
            # LOGGING
            # --------------------------------------------------------------
            log_path = os.path.join(log_dir, f"iter{iteration}_{completed_phase}.json")
            with open(log_path, "w") as f:
                json.dump(make_json_compatible(current_metadata), f, indent=2)

            print(f"Saved log for iteration {iteration}, phase '{completed_phase}' → {log_path}")

            summary_records.append({
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "iteration": iteration,
                "phase": completed_phase,
                "status": current_metadata.get("status", "unknown"),
                "action": current_metadata.get("action", None),
            })

            if continue_workflow:
                print("\n------------------------------------------")
                print("Preparing for next phase...\n")

        # ------------------------------------------------------------------
        # FINAL SUMMARY
        # ------------------------------------------------------------------
        end_time = time.time()
        print(f"\nWorkflow runtime: {end_time - start_time:.2f} seconds")

        audit.finalize()

        summary_path = os.path.join(log_dir, "workflow_summary.json")
        with open(summary_path, "w") as f:
            json.dump(make_json_compatible(summary_records), f, indent=2)

        print("\n===== WORKFLOW FINISHED =====\n")
        print(f"\nWorkflow summary saved to: {summary_path}")

