import pandas as pd
import numpy as np

def generate_data():
     # Generates 365 days of data.
    dates = pd.date_range(start="2025-01-01", periods=365, freq='D')

    # Simulates a starting price for BTC, then has it go on a random walk with varying volatility.
    starting_price = 45000
    drift = 0.0005 # How much stock price probably changes little by little.
    
    # Varies the volatility over time.
    base_volatility = 0.02
    volatility_noise = np.random.randn(len(dates)) * 0.01
    daily_volatility = base_volatility + np.cumsum(volatility_noise) * 0.001
    daily_volatility = np.clip(daily_volatility, 0.005, 0.08)  # Keep between 0.5% and 8% b/c the market would never go that crazy...
    
    # Generate returns using volatility calculation.
    returns = drift + np.random.randn(len(dates)) * daily_volatility
    
    # Apply geometric random walk
    price = starting_price * np.cumprod(1 + returns)

    # Simulates random sentiment, either -1 or 1.
    sentiment = np.random.uniform(-1, 1, len(dates))

    # Assembles a DataFrame. We did this in the notebook, but this is just a random data version.
    data = pd.DataFrame({
        'date': dates,
        'price': price,
        'return': returns,
        'avg_sentiment': sentiment
    })

    return data

