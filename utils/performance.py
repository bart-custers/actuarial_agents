import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.general_utils import make_json_compatible

class WorkflowAudit:
    def __init__(self, log_dir="data/audit"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.records = []
        self.start_time = time.time()

    def record_event(self, phase, iteration, action, metadata=None, sent=None, received=None):
        """Record an audit event. Optionally include the sent message and received response.

        sent / received should be JSON-serializable or will be converted using
        make_json_compatible.
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "iteration": iteration,
            "action": action,
            "status": metadata.get("status") if metadata else None,
            "sent": make_json_compatible(sent) if sent is not None else None,
            "received": make_json_compatible(received) if received is not None else None,
        }
        self.records.append(entry)

    def finalize(self):
        df = pd.DataFrame(self.records)
        df.to_csv(os.path.join(self.log_dir, "audit_log.csv"), index=False)
        runtime = time.time() - self.start_time
        print(f"\n Total runtime: {runtime:.2f} seconds")
        print(f"Audit log saved to: {self.log_dir}/audit_log.csv")
        return df
