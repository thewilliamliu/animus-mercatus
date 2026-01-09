import pandas as pd
import numpy as np
import os 
from pathlib import Path 

def load_data(filename):
    
    # Locates the project root based on where this data loader currently is.
    current_dir = Path(__file__).parent 
    project_root = current_dir.parent

    # Constructs filepath to whatever data must be loaded.
    filepath = project_root / 'data' / filename

    # Reads CSV file via pandas.
    df = pd.read_csv(filepath)

    # Keeps only the columns that are useful + standardizes names to data_generator.
    columns_to_keep = ['merge_date', 'close', 'weighted_sentiment']
    df = df[columns_to_keep]
    df = df.rename(columns={
        'merge_date': 'date',
        'close': 'price',
        'weighted_sentiment':'avg_sentiment'        
    })

    # Sorts values by date so they are in order.
    df = df.sort_values('date').reset_index(drop=True)
    # Calculate daily returns, removes first column with NaN return.
    df['return'] = df['price'].pct_change()
    df = df.dropna(subset=['return'])

    return df