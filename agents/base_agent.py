class BaseAgent:
    def __init__(self, name, hub):
        self.name = name
        self.hub = hub

    def receive(self, content, sender):
        """Default behavior: echo the message."""
        return {"agent": self.name, "status": "received", "input": content}