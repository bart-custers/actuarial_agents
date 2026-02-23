import pandas as pd
import numpy as np

class DataCleaning:
    def __init__(self):
        self.actions_log = []

    def clean(self, data: pd.DataFrame):
        data = data.copy() 

        clip_rules = {
            "VehAge": 20,
            "DrivAge": 90,
            "BonusMalus": 150,
            "ClaimNb": 5
        }

        for col, upper_bound in clip_rules.items():
            if col in data.columns:
                data[col] = data[col].clip(upper=upper_bound)
                self.actions_log.append(f"Clipped {col} at upper={upper_bound}.")
            else:
                self.actions_log.append(f"Skipped clipping: {col} not found.")

        if "Area" in data.columns:
            data["Area"] = data["Area"].astype("category").cat.codes + 1
            self.actions_log.append("Mapped Area categories to numeric scale.")
        else:
            self.actions_log.append("Skipped encoding: Area not found.")

        empty_cols = data.columns[data.isna().all()].tolist()
        if empty_cols:
            data = data.drop(columns=empty_cols)
            self.actions_log.append(f"Dropped fully empty columns: {empty_cols}")

        before_rows = len(data)
        data = data.dropna()
        after_rows = len(data)
        self.actions_log.append(
            f"Dropped {before_rows - after_rows} rows containing missing values."
        )

        return data
