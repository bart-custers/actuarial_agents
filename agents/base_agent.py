import json
from abc import ABC, abstractmethod
from utils.message_types import Message

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def handle_message(self, message: Message) -> Message:
        """Process incoming message and return a reply message."""
        pass
