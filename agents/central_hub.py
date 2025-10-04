import datetime

class CentralHub:
    def __init__(self):
        self.logs = []  # store all messages
        self.memory = {} # Shared state: predictions, explanations, etc.

    def log_message(self, sender, receiver, content):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "content": content
        }
        self.logs.append(entry)

    def update_memory(self, key: str, value):
        """Store or update shared memory entries."""
        self.memory[key] = value
        self.log_message("Hub", "Memory", {key: value})

    def send(self, sender, receiver, content):
        """Route message from sender to receiver (agent method call)."""
        self.log_message(sender, receiver, content)
        return receiver.receive(content, sender)

    def show_logs(self):
        for log in self.logs:
            print(f"[{log['timestamp']}] {log['sender']} -> {log['receiver']}: {log['content']}")
