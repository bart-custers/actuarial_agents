import json

# ----------------------------
# Base deterministic agent
# ----------------------------
class BaseAgent:
    def __init__(self, name, hub):
        self.name = name
        self.hub = hub
        self.memory = {}

    def receive(self, message, sender=None):
        """Default behavior: echo the message."""
        output = {"agent": self.name, "status": "received", "input": message}
        self.log_memory(output)
        return output

    def log_memory(self, message):
        """Store memory and log to central hub."""
        self.memory.update(message)
        self.hub.log_message(self.name, message)


# ----------------------------
# LLM-based agent
# ----------------------------
class LLMBaseAgent(BaseAgent):  # inherit BaseAgent for consistent interface
    def __init__(self, name, hub, llm):
        super().__init__(name, hub)
        self.llm = llm

    def receive(self, content, sender):
        """Use LLM to process message."""
        prompt = self.create_prompt(content)
        llm_output = self.llm(prompt)
        try:
            output = json.loads(llm_output)
        except json.JSONDecodeError:
            output = {"error": "failed to parse LLM output", "raw": llm_output}
        self.log_memory(output)
        return output

    def create_prompt(self, content):
        raise NotImplementedError("LLM agents must implement create_prompt()")

