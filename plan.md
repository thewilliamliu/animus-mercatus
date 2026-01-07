## Project TODO: Crypto Sentiment + Price ML

### 1. Project Setup
- **Create repo & environment**
  - [ ] Initialize git repo
  - [ ] Create `README.md` with project overview and goals
  - [ ] Set up Python virtual environment
  - [ ] Choose main stack (e.g., `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `praw`/`psaw` or Reddit API wrapper, `yfinance`/`ccxt`/Crypto data API, `matplotlib`/`seaborn`)

- **Config & structure**
  - [ ] Create basic folder structure: `data/`, `notebooks/`, `src/`, `configs/`, `reports/`
  - [ ] Add `.env` and config for API keys (Reddit, price data provider)
  - [ ] Create starter script/notebook for EDA

### 2. Data Collection – Reddit
- **Define sources**
  - [ ] List target subreddits (e.g., `r/CryptoCurrency`, `r/Bitcoin`, major coin subs)
  - [ ] Decide time window (e.g., last 1–3 years) and granularity (daily)
  - [ ] Define which fields to pull (title, body, score, comments count, created_utc, etc.)

- **Implement Reddit scraper**
  - [ ] Set up Reddit API client (PRAW/PSAW or direct API)
  - [ ] Implement function to pull posts for a given subreddit and date range
  - [ ] Save raw Reddit data to `data/raw/reddit_*.parquet` or `.csv`
  - [ ] Add basic logging and simple retry for API limits

### 3. Data Collection – Crypto Prices
- **Define assets & frequency**
  - [ ] Choose which coins (e.g., BTC, ETH, total market cap, etc.)
  - [ ] Choose frequency (daily close, open/high/low/close, volume)

- **Implement price fetcher**
  - [ ] Implement function to pull historical prices for selected assets
  - [ ] Align date range to Reddit data period
  - [ ] Save raw price data to `data/raw/prices_*.parquet` or `.csv`

### 4. Preprocessing & Alignment
- **Reddit text preprocessing**
  - [ ] Clean text (lowercase, remove URLs, emojis, stopwords, etc.)
  - [ ] Decide unit of aggregation (per post vs per day)
  - [ ] Aggregate Reddit data to daily level (e.g., volume of posts, avg score)

- **Time alignment**
  - [ ] Convert all timestamps to same timezone and date floor (e.g., UTC day)
  - [ ] Join Reddit daily metrics with corresponding daily price data
  - [ ] Handle missing days (no posts, no prices, weekends if needed)

### 5. Sentiment Modeling
- **Baseline sentiment labeling**
  - [ ] Start with simple sentiment model (e.g., VADER, TextBlob, or pre-trained transformer sentiment model)
  - [ ] Map outputs to -1 / 0 / 1 labels
  - [ ] Aggregate sentiment per day (mean, median, counts per class)

- **Explore richer sentiment features**
  - [ ] Consider continuous sentiment scores instead of discrete labels
  - [ ] Add additional text features (e.g., TF-IDF, topic modeling, embeddings)
  - [ ] Save processed sentiment dataset to `data/processed/sentiment_prices.parquet`

### 6. Target Definition (What to Predict)
- **Define prediction goal**
  - [ ] Decide whether to predict next-day price direction (up/down), return sign, or volatility
  - [ ] Create target variable (e.g., `y = sign(return_{t+1})` or volatility regime)
  - [ ] Add lagged features (e.g., sentiment at t, prices at t, returns at t-1..t-k)

### 7. Baseline ML Models
- **Train-test split & evaluation setup**
  - [ ] Use time-series split (no leakage) for train/validation/test
  - [ ] Define metrics (accuracy, F1, ROC-AUC, hit rate vs baseline)

- **Logistic Regression**
  - [ ] Implement baseline logistic regression classifier on sentiment + price features
  - [ ] Run hyperparameter search (regularization, C, class weights)
  - [ ] Log results and save model

- **Random Forest**
  - [ ] Implement Random Forest classifier/regressor (depending on target)
  - [ ] Tune key hyperparameters (n_estimators, max_depth, etc.)
  - [ ] Compare performance to logistic regression

### 8. Time-Series Models (GARCH / LSTM)
- **GARCH (if modeling volatility / returns)**
  - [ ] Implement GARCH model using `arch` / `statsmodels`
  - [ ] Incorporate sentiment as exogenous variable (if applicable)
  - [ ] Evaluate whether sentiment improves volatility/return forecasts

- **LSTM (if desired)**
  - [ ] Prepare sequence data (windowed time-series with sentiment + prices)
  - [ ] Build simple LSTM (Keras/PyTorch)
  - [ ] Train, validate, and compare to classical models

### 9. Analysis of Predictions & Sentiment Regimes
- **Correct predictions over time**
  - [ ] Plot rolling accuracy of models over time (e.g., 30-day window)
  - [ ] Analyze which periods models perform best/worst (bull vs bear markets)
  - [ ] Check whether performance is regime-dependent (high vs low volatility)

- **High vs low sentiment days**
  - [ ] Define high/low sentiment thresholds (e.g., top/bottom quantiles)
  - [ ] Compare average returns/volatility on high vs low sentiment days
  - [ ] Analyze model performance conditional on sentiment regime

### 10. Visualization & Reporting
- **Plots**
  - [ ] Time series of sentiment vs price
  - [ ] Distribution of daily sentiment
  - [ ] Confusion matrices and ROC curves
  - [ ] Rolling accuracy and feature importance plots

- **Summary report**
  - [ ] Create a notebook or markdown report summarizing:
    - Data sources and preprocessing
    - Model setup and performance
    - Insights on sentiment vs price behavior
  - [ ] Document limitations and ideas for future improvements

### 11. Polishing & Next Steps
- **Code quality**
  - [ ] Refactor scripts into reusable modules under `src/`
  - [ ] Add basic tests for key data/feature functions
  - [ ] Add a simple `Makefile` or `run.py` entrypoint to reproduce pipeline

- **Future ideas**
  - [ ] Add more data sources (Twitter/X, news APIs)
  - [ ] Try more advanced NLP models (finetuned transformers on finance text)
  - [ ] Explore different targets (drawdowns, regime switches, options-implied metrics)