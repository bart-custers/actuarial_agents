import json, os
from datetime import datetime
from utils.general_utils import make_json_compatible

class CentralMemory:
    """
    Persistent shared memory for multi-agent systems.

    Stores data in a JSON file, allowing agents to read, update, and log events
    in a centralized manner. Automatically handles JSON serialization and directory creation.
    """

    def __init__(self, memory_path="data/central_memory.json"):
        """
        Initialize the central memory.

        Args:
            memory_path (str): File path to store memory as a JSON file.
        """
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        self.data = self._load()

    def _load(self):
        """
        Load memory from JSON file if it exists.

        Returns:
            dict: Loaded memory data, or default structure if file missing/corrupted.
        """
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                try:
                    return json.load(f)
                except Exception:
                    print("Warning: Failed to load existing memory; starting fresh.")
        return {"logs": []}

    def _save(self):
        """
        Save the current memory state to the JSON file.

        Automatically converts non-JSON-compatible data using `make_json_compatible`.
        """
        with open(self.memory_path, "w") as f:
            json.dump(make_json_compatible(self.data), f, indent=2)

    def update(self, key, value, append=False):
        """
        Update memory with a new key-value pair.

        Args:
            key (str): Memory key to update.
            value (any): Value to store.
            append (bool): If True, append to list at key; otherwise, overwrite.
        """
        if append:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        else:
            self.data[key] = value
        self._save()

    def get(self, key, default=None):
        """
        Retrieve a value from memory.

        Args:
            key (str): Key to access.
            default (any): Value to return if key is missing.

        Returns:
            any: Value stored under key, or default if missing.
        """
        return self.data.get(key, default)

    def log_event(self, agent, event_type, content):
        """
        Append a structured event to the global memory logs.

        Args:
            agent (str): Name or ID of the agent generating the event.
            event_type (str): Type/category of the event.
            content (any): Event-specific data (will be JSON-compatible).
        """
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "event_type": event_type,
            "content": make_json_compatible(content),
        }
        self.data.setdefault("logs", []).append(event)
        self._save()
        print(f"[Memory] Logged event from {agent}: {event_type}")
