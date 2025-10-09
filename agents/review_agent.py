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

# from .base_agent import BaseAgent
# import pandas as pd
# import numpy as np
# from sklearn.metrics import r2_score

# class ReviewingAgent(BaseAgent):
#     def receive(self, content, sender):
#         task = content.get("task", "")
#         if task == "review_model":
#             # Load model output from hub
#             results = self.hub.memory.get("glm_results", None)
#             if results is None:
#                 return {"status": "error", "message": "No GLM results found in memory"}

#             preds_path = results.get("preds_path")
#             preds = pd.read_csv(preds_path)

#             # Basic audit metrics
#             preds["error"] = preds["y_true"] - preds["y_pred"]
#             overfit_ratio = np.mean(preds["y_pred"]) / np.mean(preds["y_true"])
#             r2 = r2_score(preds["y_true"], preds["y_pred"])

#             # Fairness proxy: group by risk-related feature (needs to match your dataset)
#             data_info = self.hub.memory.get("cleaned_data")
#             X_test = pd.read_csv(data_info["test_path"])
#             group = X_test.copy()
#             group["y_true"] = preds["y_true"]
#             group["y_pred"] = preds["y_pred"]
#             fairness_df = group.groupby("DrivAge")[["y_true","y_pred"]].mean().reset_index()

#             audit_report = {
#                 "r2_score": r2,
#                 "overfit_ratio": overfit_ratio,
#                 "fairness_by_age": fairness_df.to_dict(orient="records"),
#                 "status": "success"
#             }

#             # Store report in central memory
#             self.hub.update_memory("audit_report", audit_report)
#             print(f"{self.name}: Audit complete. R²={r2:.3f}, Overfit ratio={overfit_ratio:.3f}")
#             return audit_report

#         else:
#             return {"status": "unknown_task"}
