from .base_agent import BaseAgent
from .base_agent import LLMBaseAgent
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# class DataPrepAgent(BaseAgent):
#     def receive(self, content, sender):
#         task = content.get("task", "")
#         if task == "clean_data":
#             data_path = content.get("data_path", "data/raw/freMTPL2freq.csv")
#             print(f"Attempting to load: {data_path}")
#             df = pd.read_csv(data_path)
#             print(f"{self.name}: Loaded dataset with shape {df.shape}")

#             # --- Basic cleaning ---
#             df = df.dropna(subset=["ClaimNb", "Exposure"])
#             df = df[df["Exposure"] > 0]

#             # --- Feature selection ---
#             features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
#             target = "ClaimNb"

#             X = df[features]
#             y = df[target]

#             # --- Split into train/test ---
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )

#             # --- Save to processed folder ---
#             os.makedirs("data/processed", exist_ok=True)
#             X_train.to_csv("data/processed/X_train.csv", index=False)
#             X_test.to_csv("data/processed/X_test.csv", index=False)
#             y_train.to_csv("data/processed/y_train.csv", index=False)
#             y_test.to_csv("data/processed/y_test.csv", index=False)

#             # --- Define preprocessing pipeline ---
#             numeric_features = ["VehPower", "VehAge", "DrivAge", "Density"]
#             preprocessor = ColumnTransformer(
#                 transformers=[("num", Pipeline([("scaler", StandardScaler())]), numeric_features)],
#                 remainder="drop"
#             )

#             # --- Store metadata in memory ---
#             data_bundle = {
#                 "preprocessor": preprocessor,
#                 "train_path": "data/processed/X_train.csv",
#                 "test_path": "data/processed/X_test.csv"
#             }

#             self.hub.update_memory("cleaned_data", data_bundle)

#             return {"status": "success", "message": "Data cleaned and saved to processed/"}
#         else:
#             return {"status": "unknown_task"}

class DataPrepAgent(LLMBaseAgent):
    def create_prompt(self, message):
        return f"""
        You are a Data Prep Agent.
        Task: {message['task']}
        Data Path: {message.get('data_path', '')}
        Respond with a JSON containing:
        {{
            "cleaned_data_summary": "...",
            "confidence": 0.0
        }}
        Only respond in JSON.
        """
