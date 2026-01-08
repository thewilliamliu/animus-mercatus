"""
Configuration loader for API keys and settings.
Loads environment variables from .env file.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Reddit API credentials (if using PRAW)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Crypto API keys (if not using yfinance)
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")