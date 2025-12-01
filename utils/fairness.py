import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def group_fairness(df, pred_col='Prediction', true_col='ClaimNb', storage_dir='storage_dir'):  
    df = df.copy()
    
    # Create bins based on predicted frequency
    df['pred_bin'] = pd.qcut(df[pred_col], q=5, labels=False) + 1
    
    # Create age groups
    bins = [0, 25, 75, 120]  # adjust max age
    labels = ['young', 'middle', 'old']
    df['age_group'] = pd.cut(df['DrivAge'], bins=bins, labels=labels, right=False)
    
    # Create population density groups
    median_density = df['Density'].median()
    df['density_group'] = np.where(df['Density'] <= median_density, 'low', 'high')
    
    # Group by predicted bin and feature, compute mean actual and predicted
    agg_age = df.groupby(['pred_bin', 'age_group'])[[true_col, pred_col]].mean().reset_index()
    agg_age['diff'] = agg_age[true_col] - agg_age[pred_col]

    agg_density = df.groupby(['pred_bin', 'density_group'])[[true_col, pred_col]].mean().reset_index()
    agg_density['diff'] = agg_density[true_col] - agg_density[pred_col]
    
    # Create a table
    table_age = agg_age.pivot(index='pred_bin', columns='age_group', values='diff')
    table_density = agg_density.pivot(index='pred_bin', columns='density_group', values='diff')
    
    print("Mean difference by predicted bin and age group:")
    print(table_age)
    print("\nMean difference by predicted bin and density group:")
    print(table_density)
    
    # Plot
    plt.figure(figsize=(10,5))
    
    # Age groups
    for group in table_age.columns:
        plt.plot(table_age.index, table_age[group], marker='o', label=f'Age: {group}')
    
    plt.xlabel('Predicted frequency bin')
    plt.ylabel('Mean difference (predicted - actual)')
    plt.title('Mean difference per bin by age group')
    plt.legend()
    plt.show()
    
    # Density groups
    plt.figure(figsize=(10,5))
    for group in table_density.columns:
        plt.plot(table_density.index, table_density[group], marker='o', label=f'Density: {group}')
    
    plt.xlabel('Predicted frequency bin')
    plt.ylabel('Mean difference (predicted - actual)')
    plt.title('Mean difference per bin by population density')
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    density_plot_path = os.path.join(storage_dir, f'group_fairness_plots_{timestamp}.png')
    plt.savefig(density_plot_path)

    plt.show()
    
    return table_age, table_density
