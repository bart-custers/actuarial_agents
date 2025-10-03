import datetime

class CentralHub:
    def __init__(self):
        self.logs = []  # store all messages

    def log_message(self, sender, receiver, content):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sender": sender,
            "receiver": receiver,
            "content": content
        }
        self.logs.append(entry)

    def send(self, sender, receiver, content):
        """Route message from sender to receiver (agent method call)."""
        self.log_message(sender, receiver, content)
        return receiver.receive(content, sender)

    def show_logs(self):
        for log in self.logs:
            print(f"[{log['timestamp']}] {log['sender']} -> {log['receiver']}: {log['content']}")
