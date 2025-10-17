import json, os
from datetime import datetime
from utils.general_utils import make_json_compatible

class CentralMemory:
    """Persistent memory shared across agents, stored as a JSON file."""

    def __init__(self, memory_path="data/central_memory.json"):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                try:
                    return json.load(f)
                except Exception:
                    print("Warning: Failed to load existing memory; starting fresh.")
        return {"logs": []}

    def _save(self):
        with open(self.memory_path, "w") as f:
            json.dump(make_json_compatible(self.data), f, indent=2)

    def update(self, key, value, append=False):
        """Update or append to memory."""
        if append:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        else:
            self.data[key] = value
        self._save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def log_event(self, agent, event_type, content):
        """Append structured event to global memory logs."""
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "event_type": event_type,
            "content": make_json_compatible(content),
        }
        self.data.setdefault("logs", []).append(event)
        self._save()
        print(f"[Memory] Logged event from {agent}: {event_type}")
