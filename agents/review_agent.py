from llms.wrappers import LLMWrapper
from utils.message_types import Message
from agents.base_agent import BaseAgent

class ReviewingAgent(BaseAgent):
    def __init__(self, name="reviewing", llm_backend="mock", system_prompt=None):
        super().__init__(name)
        self.llm = LLMWrapper(backend=llm_backend, system_prompt=system_prompt)
    
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] received: {message.content}")
        # For now: placeholder until Week 3
        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content="Reviewing agent placeholder: blabla.",
        )


