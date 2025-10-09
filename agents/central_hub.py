from utils.message_types import Message
from agents.data_prep_agent import DataPrepAgent
from agents.modelling_agent import ModellingAgent
from agents.review_agent import ReviewingAgent
from agents.explanation_agent import ExplanationAgent

class CentralHub:
    def __init__(self):
        self.agents = {
            "dataprep": DataPrepAgent("dataprep"),
            "modelling": ModellingAgent("modelling"),
            "reviewing": ReviewingAgent("reviewing"),
            "explanation": ExplanationAgent("explanation"),
        }

    def send(self, message: Message) -> Message:
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

        print("\n--- 🧹 Data Preparation Completed ---")
        print(r1.content)

        # Auto-display LLM explanation if available
        if r1.metadata and "llm_explanation" in r1.metadata:
            print("\n LLM Explanation:")
            print(r1.metadata["llm_explanation"])

        # Modelling phase
        r2 = self.send(Message(
            sender="hub",
            recipient="modelling",
            type="task",
            content="Train predictive model."
        ))
        print("\n--- Modelling Completed ---")
        print(r2.content)

        # Reviewing phase
        r3 = self.send(Message(
            sender="hub",
            recipient="reviewing",
            type="task",
            content="Review model outputs."
        ))
        print("\n--- Review Completed ---")
        print(r3.content)

        # Explanation phase
        r4 = self.send(Message(
            sender="hub",
            recipient="explanation",
            type="task",
            content="Generate explanations."
        ))
        print("\n--- Explanation Completed ---")
        print(r4.content)

        print("\n===== WORKFLOW FINISHED =====\n")

        print("--- Summary ---")
        print(f"dataprep → {r1.content}")
        print(f"modelling → {r2.content}")
        print(f"reviewing → {r3.content}")
        print(f"explanation → {r4.content}")

    # def workflow_demo(self):
    #     """Simple end-to-end placeholder run."""
    #     msg = Message(
    #         sender="hub",
    #         recipient="dataprep",
    #         type="task",
    #         content="Clean dataset and summarize.",
    #         metadata={"dataset_name": "freMTPL2freq.csv"}
    #     )
    #     r1 = self.send(msg)
    #     r2 = self.send(Message(sender="hub", recipient="modelling", type="task", content="Train predictive model."))
    #     r3 = self.send(Message(sender="hub", recipient="reviewing", type="task", content="Review model outputs."))
    #     r4 = self.send(Message(sender="hub", recipient="explanation", type="task", content="Generate explanations."))
    #     print("\n--- Final results ---")
    #     for r in [r1, r2, r3, r4]:
    #         print(f"{r.sender} → {r.content}")


# import datetime

# class CentralHub:
#     def __init__(self):
#         self.logs = []  # store all messages
#         self.memory = {} # Shared state: predictions, explanations, etc.

#     def log_message(self, sender, receiver, content):
#         entry = {
#             "timestamp": datetime.datetime.now().isoformat(),
#             "sender": sender,
#             "receiver": receiver,
#             "content": content
#         }
#         self.logs.append(entry)

#     def update_memory(self, key: str, value):
#         """Store or update shared memory entries."""
#         self.memory[key] = value
#         self.log_message("Hub", "Memory", {key: value})

#     def send(self, sender, receiver, content):
#         """Route message from sender to receiver (agent method call)."""
#         self.log_message(sender, receiver, content)
#         return receiver.receive(content, sender)

#     def show_logs(self):
#         for log in self.logs:
#             print(f"[{log['timestamp']}] {log['sender']} -> {log['receiver']}: {log['content']}")
