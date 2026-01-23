import pandas as pd
import numpy as np

class DataCleaning:
    def __init__(self):
        self.actions_log = []

    def clean(self, data: pd.DataFrame):
        # Clip outliers / right-censoring 
        self.actions_log.append("Clipping VehAge (<=20), DrivAge (<=90), BonusMalus (<=150), ClaimNb (<=5).")
        data['VehAge'] = data['VehAge'].clip(upper=20)
        data['DrivAge'] = data['DrivAge'].clip(upper=90)
        data['BonusMalus'] = data['BonusMalus'].clip(upper=150)
        data['ClaimNb'] = data['ClaimNb'].clip(upper=5)

        # Encode Area as ordinal numeric 
        data['Area'] = data['Area'].astype('category').cat.codes + 1

        self.actions_log.append("Mapped Area categories to numeric scale (A→1,...,F→6).")
        
        empty_cols = data.columns[data.isna().all()].tolist()
        if empty_cols:
            data = data.drop(columns=empty_cols)
        
        # Remove missing values
        data = data.dropna()
    
        return data
