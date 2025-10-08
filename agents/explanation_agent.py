from agents.base_agent import BaseAgent
from utils.message_types import Message

class ExplanationAgent(BaseAgent):
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] explaining results.")
        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content="Generated placeholder explanation for model predictions.",
            metadata={"explanation_quality": "draft"}
        )

# from .base_agent import BaseAgent
# import shap
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import os

# class ExplanationAgent(BaseAgent):
#     def receive(self, content, sender):
#         task = content.get("task", "")
#         if task == "explain_model":
#             results = self.hub.memory.get("glm_results", None)
#             data_info = self.hub.memory.get("cleaned_data", None)
#             if results is None or data_info is None:
#                 return {"status": "error", "message": "Missing inputs in memory"}

#             model = joblib.load(results["model_path"])
#             X_test = pd.read_csv(data_info["test_path"])
#             preprocessor = data_info["preprocessor"]
#             X_test_prep = preprocessor.transform(X_test)

#             # Compute SHAP values
#             explainer = shap.Explainer(model, X_test_prep)
#             shap_values = explainer(X_test_prep[:200])  # subset for speed

#             # Save summary plot
#             os.makedirs("data/final", exist_ok=True)
#             shap_path = "data/final/shap_summary.png"
#             shap.summary_plot(shap_values, X_test, show=False)
#             plt.tight_layout()
#             plt.savefig(shap_path, dpi=150)
#             plt.close()

#             explanation_result = {
#                 "status": "success",
#                 "shap_path": shap_path,
#                 "top_features": shap_values.abs.mean(0).argsort()[-5:].tolist()
#             }

#             self.hub.update_memory("explanations", explanation_result)
#             print(f"{self.name}: SHAP explanations saved to {shap_path}")
#             return explanation_result

#         else:
#             return {"status": "unknown_task"}
