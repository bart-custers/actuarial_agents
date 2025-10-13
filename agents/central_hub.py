from utils.message_types import Message
from llms.wrappers import LLMWrapper
from agents.data_prep_agent import DataPrepAgent
from agents.modelling_agent import ModellingAgent
from agents.review_agent import ReviewingAgent
from agents.explanation_agent import ExplanationAgent

class CentralHub:  
    def __init__(self):
        # ✅ 1. Create ONE shared LLM instance
        self.shared_llm = LLMWrapper(
            backend="llama7b",
            system_prompt="You are a helpful actuarial assistant that can handle multiple reasoning tasks.",
        )

        # ✅ 2. Create agents that reuse this instance
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
        """Run full demonstration of the agent workflow."""
        print("\n===== STARTING MULTI-AGENT WORKFLOW DEMO =====\n")

        # Data preparation phase
        msg = Message(
            sender="hub",
            recipient="dataprep",
            type="task",
            content="Clean dataset and summarize.",
            metadata={"dataset_path": "data/raw/freMTPL2freq.csv"},
        )
        r1 = self.send(msg)

        print("\n--- Data Preparation Completed ---")
        print(r1.content)
        
        if "llm_explanation" in msg.metadata:
            print("\n--- LLM Explanation ---")
            print(msg.metadata["llm_explanation"])

        # Auto-display LLM explanation if available
        if r1.metadata and "llm_explanation" in r1.metadata:
            print("\n LLM Explanation:")
            print(r1.metadata["llm_explanation"])

        # Modelling phase
        r2 = self.send(Message(
            sender="hub",
            recipient="modelling",
            type="task",
            content="Train predictive model.",
            metadata=r1.metadata,
        ))
        print("\n--- Modelling Completed ---")
        print(r2.content)

        if "llm_explanation" in msg.metadata:
            print("\n--- LLM Explanation ---")
            print(msg.metadata["llm_explanation"])

        # Reviewing phase
        r3 = self.send(Message(
            sender="hub",
            recipient="reviewing",
            type="task",
            content="Review model outputs.",
            metadata=r2.metadata,
        ))
        print("\n--- Review Completed ---")
        print(r3.content)

        # Display LLM review text if available
        if r3.metadata and "llm_review" in r3.metadata:
            print("\n--- LLM Review ---")
            print(r3.metadata["llm_review"])

        # === Decide next routing based on review status ===
        status = r3.metadata.get("status", "approved")
        action = r3.metadata.get("action", "proceed_to_explanation")

        if action == "retrain_model":
            print("[Hub] Review requested retraining — forwarding to modelling.\n")
            retrain_msg = Message(
                sender="hub",
                recipient="modelling",
                type="task",
                content=r3.metadata.get("retrain_prompt", "Retrain model with adjustments."),
                metadata=r2.metadata,  # reuse same data
            )
            r4 = self.send(retrain_msg)

        elif action == "proceed_to_explanation":
            print("[Hub] Review approved model — forwarding to explanation.\n")
            r4 = self.send(Message(
                sender="hub",
                recipient="explanation",
                type="task",
                content="Explain model results and predictions.",
                metadata=r3.metadata,
            ))

        else:
            print(f"[Hub] Unknown action: {action}. Defaulting to explanation step.")
            r4 = self.send(Message(
                sender="hub",
                recipient="explanation",
                type="task",
                content="Explain model results and predictions.",
                metadata=r3.metadata,
            ))

        print("\n--- Explanation Completed ---")
        print(r4.content)

        if r4.metadata and "llm_explanation" in r4.metadata:
            print("\n--- LLM Explanation ---")
            print(r4.metadata["llm_explanation"]) 
        print("\n===== WORKFLOW FINISHED =====\n")

        print("--- Summary ---")
        print(f"dataprep → {r1.content}")
        print(f"modelling → {r2.content}")
        print(f"reviewing → {r3.content}")
        print(f"explanation → {r4.content}")
