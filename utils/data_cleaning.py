import pandas as pd
import numpy as np

class DataCleaning:
    """
    Executes a predefined insurance data cleaning.
    """

    def __init__(self):
        self.actions_log = []

    def clean(self, data: pd.DataFrame):
        """
        Perform deterministic data cleaning and preprocessing.
        Returns:
            df with cleaned DataFrame
        """

        # --- Step 1: Clip outliers / right-censoring ---
        self.actions_log.append("Clipping VehAge (<=20), DrivAge (<=90), BonusMalus (<=150), ClaimNb (<=5).")
        data['VehAge'] = data['VehAge'].clip(upper=20)
        data['DrivAge'] = data['DrivAge'].clip(upper=90)
        data['BonusMalus'] = data['BonusMalus'].clip(upper=150)
        data['ClaimNb'] = data['ClaimNb'].clip(upper=5)

        # --- Step 2: Encode Area as ordinal numeric ---
        area_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
        data['Area'] = data['Area'].map(area_mapping)
        self.actions_log.append("Mapped Area categories to numeric scale (A→1,...,F→6).")

        # --- Step 3: Drop missing values ---
        before = len(data)
        data = data.dropna()
        after = len(data)
        self.actions_log.append(f"Dropped {before - after} rows with missing values.")
        
        return data
