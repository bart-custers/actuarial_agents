import os
import json
from datetime import datetime
from utils.message_types import Message
from utils.general_utils import make_json_compatible
from llms.wrappers import LLMWrapper
from agents.data_prep_agent import DataPrepAgent
from agents.modelling_agent import ModellingAgent
from agents.review_agent import ReviewingAgent
from agents.explanation_agent import ExplanationAgent

class CentralHub:  
    def __init__(self):
        # 1. Create ONE shared LLM instance
        self.shared_llm = LLMWrapper(
            backend="llama7b",
            system_prompt="You are a helpful actuarial assistant that can handle multiple reasoning tasks.",
        )

        # 2. Create agents that reuse this instance
        self.agents = {
            "dataprep": DataPrepAgent(
                "dataprep",
                shared_llm=self.shared_llm,
                system_prompt="You are a meticulous data-preparation specialist who cleans insurance datasets clearly and reproducibly."
            ),
            "modelling": ModellingAgent(
                "modelling",
                shared_llm=self.shared_llm,
                system_prompt="You are an actuarial modeller who builds and evaluates predictive models precisely."
            ),
            "reviewing": ReviewingAgent(
                "reviewing",
                shared_llm=self.shared_llm,
                system_prompt="You are a critical reviewer checking model plausibility and regulatory compliance."
            ),
            "explanation": ExplanationAgent(
                "explanation",
                shared_llm=self.shared_llm,
                system_prompt="You are an explanation specialist ensuring interpretability and fairness."
            ),
        }

    def send(self, message):
        recipient = message.recipient
        if recipient not in self.agents:
            raise ValueError(f"Unknown recipient: {recipient}")
        print(f"[Hub] routing message from {message.sender} to {recipient}")
        response = self.agents[recipient].handle_message(message)
        return response


    def workflow_demo(self):
        """Run adaptive workflow with retraining, escalation, and per-iteration logging."""
        print("\n===== STARTING MULTI-AGENT WORKFLOW DEMO =====\n")

        # === Initialize metadata ===
        current_metadata = {
            "dataset_path": "data/raw/freMTPL2freq.csv",
            "review_iteration": 0,
        }
        task = "dataprep"
        continue_workflow = True
        iteration = 0
        MAX_ITERATIONS = 4

        # === Prepare iteration log folder ===
        log_dir = "data/workflow_logs"
        os.makedirs(log_dir, exist_ok=True)

        # === Keep responses for summary ===
        r1 = r2 = r3 = r4 = None

        while continue_workflow:
            iteration += 1
            print(f"\n=== Iteration {iteration} — Starting {task.upper()} phase ===\n")

            if iteration > MAX_ITERATIONS:
                print(f"Maximum iteration limit ({MAX_ITERATIONS}) reached. Aborting workflow.")
                current_metadata["status"] = "terminated"
                current_metadata["reason"] = "Exceeded maximum iterations without model approval."
                break

            # ===============================
            # Data Preparation Phase
            # ===============================
            if task == "dataprep":
                msg = Message(
                    sender="hub",
                    recipient="dataprep",
                    type="task",
                    content="Clean dataset and summarize.",
                    metadata=current_metadata,
                )
                r1 = self.send(msg)
                current_metadata = r1.metadata or {}
                print("\n--- Data Preparation Completed ---")
                print(r1.content)
                task = "modelling"

            # ===============================
            # Modelling Phase
            # ===============================
            elif task == "modelling":
                msg = Message(
                    sender="hub",
                    recipient="modelling",
                    type="task",
                    content="Train predictive model.",
                    metadata=current_metadata,
                )
                r2 = self.send(msg)
                current_metadata = r2.metadata or {}
                print("\n--- Modelling Completed ---")
                print(r2.content)
                task = "reviewing"

            # ===============================
            # Reviewing Phase
            # ===============================
            elif task == "reviewing":
                msg = Message(
                    sender="hub",
                    recipient="reviewing",
                    type="task",
                    content="Review model outputs.",
                    metadata=current_metadata,
                )
                r3 = self.send(msg)
                current_metadata = r3.metadata or {}
                print("\n--- Review Completed ---")
                print(r3.content)

                # Show LLM review
                if r3.metadata and "llm_review" in r3.metadata:
                    print("\n--- LLM Review ---")
                    print(r3.metadata["llm_review"][:500], "..." if len(r3.metadata["llm_review"]) > 500 else "")

                # === Decision logic ===
                action = current_metadata.get("action", "proceed_to_explanation")

                if action == "retrain_model":
                    print("[Hub] Review requested retraining — forwarding to modelling.\n")
                    task = "modelling"
                elif action == "reclean_data":
                    print("[Hub] Model still inadequate — re-cleaning dataset.\n")
                    task = "dataprep"
                elif action == "proceed_to_explanation":
                    print("[Hub] Model approved — proceeding to explanation.\n")
                    task = "explanation"
                elif action == "abort_workflow":
                    print("[Hub] Workflow aborted after repeated model failure.\n")
                    continue_workflow = False
                else:
                    print(f"[Hub] Unknown action: {action} — defaulting to explanation.\n")
                    task = "explanation"

            # ===============================
            # Explanation Phase
            # ===============================
            elif task == "explanation":
                msg = Message(
                    sender="hub",
                    recipient="explanation",
                    type="task",
                    content="Generate explanations for model results and predictions.",
                    metadata=current_metadata,
                )
                r4 = self.send(msg)
                current_metadata = r4.metadata or {}
                print("\n--- Explanation Completed ---")
                print(r4.content)
                continue_workflow = False  # End of pipeline

            else:
                print(f"[Hub] Unknown next task: {task}")
                continue_workflow = False

            # ===============================
            # Save iteration logs
            # ===============================
            log_path = os.path.join(log_dir, f"iteration_{iteration}_{task}.json")
            with open(log_path, "w") as f:
                json.dump(make_json_compatible(current_metadata), f, indent=2)
            print(f"\nSaved iteration metadata to: {log_path}")

            # Optional visual separator
            if continue_workflow:
                print("\n------------------------------------------")
                print("Preparing for next iteration...\n")

        # ===============================
        # Final summary
        # ===============================
        print("\n===== WORKFLOW FINISHED =====\n")
        print("--- Summary ---")
        if r1: print(f"dataprep → {r1.content}")
        if r2: print(f"modelling → {r2.content}")
        if r3: print(f"reviewing → {r3.content}")
        if r4: print(f"explanation → {r4.content}")

        print(f"\nAll iteration metadata stored in: {log_dir}")



    # def workflow_demo(self):
    #     """Run full demonstration of the agent workflow."""
    #     print("\n===== STARTING MULTI-AGENT WORKFLOW DEMO =====\n")

    #     # Data preparation phase
    #     msg = Message(
    #         sender="hub",
    #         recipient="dataprep",
    #         type="task",
    #         content="Clean dataset and summarize.",
    #         metadata={"dataset_path": "data/raw/freMTPL2freq.csv"},
    #     )
    #     r1 = self.send(msg)

    #     print("\n--- Data Preparation Completed ---")
    #     print(r1.content)
        
    #     if "llm_explanation" in msg.metadata:
    #         print("\n--- LLM Explanation ---")
    #         print(msg.metadata["llm_explanation"])

    #     # Auto-display LLM explanation if available
    #     if r1.metadata and "llm_explanation" in r1.metadata:
    #         print("\n LLM Explanation:")
    #         print(r1.metadata["llm_explanation"])

    #     # Modelling phase
    #     r2 = self.send(Message(
    #         sender="hub",
    #         recipient="modelling",
    #         type="task",
    #         content="Train predictive model.",
    #         metadata=r1.metadata,
    #     ))
    #     print("\n--- Modelling Completed ---")
    #     print(r2.content)

    #     if "llm_explanation" in msg.metadata:
    #         print("\n--- LLM Explanation ---")
    #         print(msg.metadata["llm_explanation"])

    #     # Reviewing phase
    #     r3 = self.send(Message(
    #         sender="hub",
    #         recipient="reviewing",
    #         type="task",
    #         content="Review model outputs.",
    #         metadata=r2.metadata,
    #     ))
    #     print("\n--- Review Completed ---")
    #     print(r3.content)

    #     # Display LLM review text if available
    #     if r3.metadata and "llm_review" in r3.metadata:
    #         print("\n--- LLM Review ---")
    #         print(r3.metadata["llm_review"])

    #     # === Decide next routing based on review status ===
    #     status = r3.metadata.get("status", "approved")
    #     action = r3.metadata.get("action", "proceed_to_explanation")

    #     if action == "retrain_model":
    #         print("[Hub] Review requested retraining — forwarding to modelling.\n")
    #         retrain_msg = Message(
    #             sender="hub",
    #             recipient="modelling",
    #             type="task",
    #             content=r3.metadata.get("retrain_prompt", "Retrain model with adjustments."),
    #             metadata=r3.metadata,  # includes iteration count
    #         )
    #         r4 = self.send(retrain_msg)

    #     elif action == "reclean_data":
    #         print("[Hub] Escalation: sending workflow back to DataPrepAgent for data review.\n")
    #         re_prep_msg = Message(
    #             sender="hub",
    #             recipient="dataprep",
    #             type="task",
    #             content="Review and enhance data preprocessing, addressing model instability.",
    #             metadata=r3.metadata,
    #         )
    #         r4 = self.send(re_prep_msg)

    #     elif action == "proceed_to_explanation":
    #         print("[Hub] Review approved model — forwarding to explanation.\n")
    #         r4 = self.send(Message(
    #             sender="hub",
    #             recipient="explanation",
    #             type="task",
    #             content="Explain model results and predictions.",
    #             metadata=r3.metadata,
    #         ))

    #     else:
    #         print(f"[Hub] Unknown action: {action}. Defaulting to explanation step.")
    #         r4 = self.send(Message(
    #             sender="hub",
    #             recipient="explanation",
    #             type="task",
    #             content="Explain model results and predictions.",
    #             metadata=r3.metadata,
    #         ))

    #     print("\n--- Explanation Completed ---")
    #     print(r4.content)

    #     if r4.metadata and "llm_explanation" in r4.metadata:
    #         print("\n--- LLM Explanation ---")
    #         print(r4.metadata["llm_explanation"]) 
    #     print("\n===== WORKFLOW FINISHED =====\n")

    #     print("--- Summary ---")
    #     print(f"dataprep → {r1.content}")
    #     print(f"modelling → {r2.content}")
    #     print(f"reviewing → {r3.content}")
    #     print(f"explanation → {r4.content}")
