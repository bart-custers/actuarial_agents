from llms.wrappers import LLMWrapper
from utils.message_types import Message
from agents.base_agent import BaseAgent

class ExplanationAgent(BaseAgent):
    def __init__(self, name="explanation", shared_llm=None, system_prompt=None):
        super().__init__(name)
        self.llm = shared_llm or LLMWrapper(backend="mock", system_prompt=system_prompt)
        self.system_prompt = system_prompt
    
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] received: {message.content}")
        # For now: placeholder until Week 3
        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content="Explaining agent placeholder: blabla.",
        )


