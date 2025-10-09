from llms.wrappers import LLMWrapper
from utils.message_types import Message
from agents.base_agent import BaseAgent

class ModellingAgent(BaseAgent):
    def __init__(self, name="modelling", llm_backend="mock", system_prompt=None):
        # Only pass name to BaseAgent
        super().__init__(name)
        # Store the LLM with backend + role prompt
        self.llm = LLMWrapper(backend=llm_backend, system_prompt=system_prompt)

    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] received: {message.content}")
        # For now: placeholder until Week 3
        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content="Modeling agent placeholder: trained GLM stub.",
        )
