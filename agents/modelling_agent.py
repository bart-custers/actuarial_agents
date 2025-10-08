from agents.base_agent import BaseAgent
from utils.message_types import Message

class ModellingAgent(BaseAgent):
    def handle_message(self, message: Message) -> Message:
        print(f"[{self.name}] received: {message.content}")
        return Message(
            sender=self.name,
            recipient=message.sender,
            type="response",
            content="Modeling agent placeholder: trained GLM stub.",
            metadata={"status": "success"}
        )


# from .base_agent import BaseAgent
# from sklearn.linear_model import PoissonRegressor
# from sklearn.metrics import mean_poisson_deviance, mean_absolute_error, root_mean_squared_error
# import pandas as pd
# import numpy as np
# import os
# import joblib  # for saving model

# class ModellingAgent(BaseAgent):
#     def receive(self, content, sender):
#         task = content.get("task", "")
#         if task == "train_glm":
#             # --- 1. Load paths and preprocessor from memory ---
#             data_info = self.hub.memory.get("cleaned_data", None)
#             if data_info is None:
#                 return {"status": "error", "message": "No cleaned data found in memory"}

#             train_path = data_info.get("train_path", "data/processed/X_train.csv")
#             test_path = data_info.get("test_path", "data/processed/X_test.csv")
#             preprocessor = data_info.get("preprocessor", None)

#             if preprocessor is None:
#                 return {"status": "error", "message": "No preprocessor found in memory"}

#             # --- 2. Load data from disk ---
#             X_train = pd.read_csv(train_path)
#             X_test = pd.read_csv(test_path)
#             y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
#             y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

#             print(f"{self.name}: Loaded training data {X_train.shape} and test data {X_test.shape}")

#             # --- 3. Preprocess data ---
#             X_train_prep = preprocessor.fit_transform(X_train)
#             X_test_prep = preprocessor.transform(X_test)

#             # --- 4. Train GLM (Poisson Regression) ---
#             model = PoissonRegressor(alpha=1e-12, max_iter=300)
#             model.fit(X_train_prep, y_train)

#             y_pred = model.predict(X_test_prep)

#             # --- 5. Evaluate performance ---
#             metrics = {
#                 "poisson_dev": mean_poisson_deviance(y_test, y_pred),
#                 "mae": mean_absolute_error(y_test, y_pred),
#                 "rmse": root_mean_squared_error(y_test, y_pred)
#             }

#             # --- 6. Save results ---
#             os.makedirs("data/final", exist_ok=True)
#             preds_path = "data/final/glm_predictions.csv"
#             model_path = "data/final/glm_model.pkl"

#             pd.DataFrame({
#                 "y_true": y_test,
#                 "y_pred": y_pred
#             }).to_csv(preds_path, index=False)

#             joblib.dump(model, model_path)

#             # --- 7. Update hub memory ---
#             result = {
#                 "status": "success",
#                 "metrics": metrics,
#                 "model_path": model_path,
#                 "preds_path": preds_path
#             }

#             self.hub.update_memory("glm_results", result)
#             print(f"{self.name}: GLM trained successfully – RMSE={metrics['rmse']:.4f}")

#             return result

#         else:
#             return {"status": "unknown_task"}