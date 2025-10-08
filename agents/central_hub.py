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
        """Simple end-to-end placeholder run."""
        msg = Message(
            sender="hub",
            recipient="dataprep",
            type="task",
            content="Clean dataset and summarize.",
            metadata={"dataset_name": "insurance_claims.csv"}
        )
        r1 = self.send(msg)
        r2 = self.send(Message(sender="hub", recipient="modelling", type="task", content="Train predictive model."))
        r3 = self.send(Message(sender="hub", recipient="reviewing", type="task", content="Review model outputs."))
        r4 = self.send(Message(sender="hub", recipient="explanation", type="task", content="Generate explanations."))
        print("\n--- Final results ---")
        for r in [r1, r2, r3, r4]:
            print(f"{r.sender} → {r.content}")


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
