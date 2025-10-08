from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Message:
    sender: str
    recipient: str
    type: str                  # e.g., "task", "response", "review", "explanation"
    content: str               # natural-language or structured prompt
    metadata: Optional[Dict[str, Any]] = None