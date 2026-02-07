# 🤖 Indian Stock Market AI Prediction System

A complete AI-powered stock prediction system for the Indian stock market (NSE). Uses machine learning to analyze technical indicators, fundamental data, and market patterns to predict the best-performing stocks.

## 🎯 Features

- **Automated Data Collection**: Fetches historical data from Yahoo Finance for NSE stocks
- **Advanced Feature Engineering**: 40+ technical indicators including RSI, MACD, Bollinger Bands, ATR, and more
- **Machine Learning Models**: XGBoost and LightGBM with walk-forward validation
- **February 2026 Predictions**: Identifies stocks with highest probability of rising
- **Comprehensive Backtesting**: Test predictions on any historical month
- **Performance Metrics**: Accuracy, ROC-AUC, precision/recall, and actual returns
- **Interactive Notebook**: Jupyter notebook for analysis and visualization

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Command Line Interface

**Train model and predict for February 2026:**
```bash
python run_predictor.py
```

**Run backtests only:**
```bash
python run_predictor.py backtest
```

**Quick prediction (requires pre-trained model):**
```bash
python run_predictor.py quick
```

### Option 2: Jupyter Notebook (Recommended)

```bash
jupyter notebook stock_analysis.ipynb
```

The notebook provides an interactive interface with:
- Step-by-step execution
- Visualizations of model performance
- Custom backtesting capabilities
- Detailed analysis and insights

### Option 3: Python Script

```python
from indian_stock_predictor import IndianStockPredictor
from datetime import datetime, timedelta

# Initialize predictor
predictor = IndianStockPredictor()

# Build dataset
end_date = datetime.now()
start_date = end_date - timedelta(days=1095)  # 3 years
df = predictor.build_dataset(start_date, end_date, prediction_days=30)

# Train model
model, results = predictor.train_walk_forward(df, n_splits=5)

# Predict for February 2026
predictions = predictor.predict_stocks(target_month=2, target_year=2026)

# Show top 5 picks
print(predictions.head(5))

# Backtest on a specific month
backtest_result = predictor.backtest_month(month=12, year=2025)
```

## 📊 System Architecture

### 1. Data Pipeline
```
Yahoo Finance API → Historical OHLCV Data → Feature Engineering → Dataset
```

**Data Sources:**
- Price data: Open, High, Low, Close, Volume
- Time period: 3 years of historical data
- Update frequency: Daily

**Technical Indicators Calculated:**
- **Momentum**: RSI, ROC, Momentum
- **Trend**: SMA (5, 10, 20, 50, 200), EMA (12, 26), MACD
- **Volatility**: ATR, Bollinger Bands
- **Volume**: Volume Ratio, Volume Change
- **Pattern**: Higher Highs, Lower Lows
- **Oscillators**: Stochastic K & D

### 2. Feature Engineering

**Feature Categories:**
1. **Price-based**: Returns (1D, 5D, 20D, 60D)
2. **Moving Averages**: SMA, EMA, Price-to-MA ratios
3. **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
4. **Volume Features**: Volume ratios and changes
5. **Momentum**: ROC, momentum indicators
6. **Pattern Recognition**: Price patterns and crossovers

Total Features: **40+**

### 3. Model Training

**Algorithm**: XGBoost (Gradient Boosting)

**Why XGBoost?**
- Handles non-linear relationships well
- Robust to outliers
- Fast training and prediction
- Built-in regularization prevents overfitting

**Model Parameters:**
```python
n_estimators=200
max_depth=6
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
```

**Validation Strategy**: Walk-Forward Validation (Time Series Split)
- Prevents look-ahead bias
- Respects temporal ordering
- 5-fold cross-validation

### 4. Prediction & Backtesting

**Target**: Binary classification (Up vs Down)
- **Label 1**: Stock price increases after 30 days
- **Label 0**: Stock price decreases after 30 days

**Output**: 
- Probability of price increase (0-100%)
- Confidence score
- Predicted direction (UP/DOWN)

## 📈 Output Format

### Predictions for February 2026

```
🔥 TOP STOCK PREDICTION FOR FEBRUARY 2026

Ticker: RELIANCE
Predicted Upside Probability: 82.5%
Confidence Score: 0.85
Latest Price: ₹2,845.30
Prediction: UP

📊 Recommendation: Long position with 2% portfolio allocation
```

### Backtest Results

```
BACKTEST SUMMARY
================================================================================
Month          | Accuracy | Top Pick      | Top Pick Return | Top 5 Avg Return
12/2025        | 68.2%    | TCS           | +8.4%          | +5.2%
11/2025        | 71.5%    | INFY          | +12.1%         | +7.8%
10/2025        | 65.8%    | HDFCBANK      | +6.3%          | +4.1%

Average Accuracy: 68.5%
Top Pick Success Rate: 73.3%
Average Top 5 Return: 5.7%
```

## ⚠️ Important Disclaimers

### Risk Warnings

1. **No Guarantees**: Past performance does not guarantee future results
2. **Market Volatility**: Stock markets are inherently unpredictable
3. **Model Limitations**: AI predictions are probabilistic, not certain
4. **External Factors**: Models cannot account for unexpected events (news, policy changes, global events)

### Recommended Usage

✅ **DO**:
- Use as ONE input among many for investment decisions
- Combine with fundamental analysis
- Apply proper risk management (position sizing, stop losses)
- Regularly retrain models with new data
- Backtest thoroughly before live trading

❌ **DON'T**:
- Invest based solely on AI predictions
- Risk more than you can afford to lose
- Ignore your own research and due diligence
- Trade without understanding the underlying stocks

### Disclaimer

**This system is for educational and research purposes only. It is NOT financial advice. Always consult with a licensed financial advisor before making investment decisions. The creators assume no liability for any financial losses incurred from using this system.**

## 🧪 Backtesting

### Test Any Historical Month

```python
# Backtest January 2026
result = predictor.backtest_month(month=1, year=2026)

print(f"Accuracy: {result['accuracy']*100:.1f}%")
print(f"Top Pick: {result['top_pick']['symbol']}")
print(f"Top Pick Return: {result['top_pick']['actual_return']*100:.2f}%")
```

### Batch Backtesting

```python
from run_predictor import run_backtests

# Test multiple months automatically
results = run_backtests()
```

This will test the model on the past 6 months and generate:
- `backtest_summary.csv`: Overall performance
- `backtest_MM_YYYY.csv`: Detailed results for each month

## 📄 Files Generated

- `predictions_feb_2026.csv` - February 2026 predictions
- `backtest_summary.csv` - Backtest results summary
- `stock_predictor_model.pkl` - Trained model
- Monthly backtest files - Detailed results per month

---

**Built with ❤️ for the Indian stock market community**
