# 🚀 QUICK START GUIDE
## Indian Stock Market AI Prediction System

### Installation (5 minutes)

1. **Install Python packages:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import pandas, xgboost, yfinance; print('✓ All packages installed')"
```

### Usage Options

#### Option A: Full Training & Prediction (Recommended for first run)
**Time: 15-30 minutes**

```bash
python run_predictor.py
```

This will:
1. Download 3 years of data for Nifty 50 stocks
2. Calculate 40+ technical indicators
3. Train XGBoost model with 5-fold validation
4. Predict top stocks for February 2026
5. Save model and predictions

**Output files:**
- `predictions_feb_2026.csv` - All stock predictions ranked
- `stock_predictor_model.pkl` - Trained model (reusable)

#### Option B: Backtesting Mode
**Time: 20-40 minutes**

```bash
python run_predictor.py backtest
```

Tests model on past 6 months to validate accuracy.

**Output files:**
- `backtest_summary.csv` - Performance summary
- `backtest_12_2025.csv`, etc. - Detailed results per month

#### Option C: Interactive Analysis (Best for exploration)

```bash
jupyter notebook stock_analysis.ipynb
```

Features:
- Step-by-step execution with explanations
- Beautiful visualizations
- Custom backtesting on any month
- Feature importance analysis

### Quick Results

After running, check your top prediction:

```python
import pandas as pd
predictions = pd.read_csv('predictions_feb_2026.csv')
best = predictions.iloc[0]

print(f"🏆 Best Stock: {best['Symbol']}")
print(f"📈 Probability: {best['Probability_Up']*100:.1f}%")
print(f"💰 Price: ₹{best['Latest_Close']:.2f}")
```

### Customization

**Use fewer stocks for faster testing:**
```python
from indian_stock_predictor import IndianStockPredictor

predictor = IndianStockPredictor(
    stock_list=['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
)
```

**Test different months:**
```python
result = predictor.backtest_month(month=1, year=2026, prediction_days=30)
```

**Change prediction horizon:**
```python
df = predictor.build_dataset(start, end, prediction_days=60)  # 2 months
```

### Understanding the Output

**Probability_Up**: Chance stock will rise (0-100%)
- >70% = High confidence
- 50-70% = Moderate confidence
- <50% = Predicted to fall

**Confidence_Score**: Model certainty (0.5-1.0)
- >0.8 = Very confident
- 0.6-0.8 = Confident
- <0.6 = Less confident

### Troubleshooting

**Error: "No module named 'xyz'"**
→ Run: `pip install xyz`

**Error: "No data for stock ABC"**
→ Stock may be delisted. Remove from list or use default Nifty 50

**Low accuracy (<55%)**
→ Retrain with more recent data or add more features

### Next Steps

1. ✅ Run `python run_predictor.py` to get February 2026 predictions
2. ✅ Review `predictions_feb_2026.csv` for ranked stocks
3. ✅ Run backtests to validate: `python run_predictor.py backtest`
4. ✅ Use Jupyter notebook for deeper analysis
5. ✅ Retrain weekly/monthly with fresh data

### Important Reminders

⚠️ This is NOT financial advice
⚠️ Always do your own research
⚠️ Use proper risk management
⚠️ Never invest more than you can afford to lose
⚠️ Markets are unpredictable - AI helps but doesn't guarantee profits

### Support

For issues or questions, review:
- README.md for detailed documentation
- Comments in indian_stock_predictor.py for code details
- Jupyter notebook for step-by-step explanations

---

**Ready to start? Run:**
```bash
python run_predictor.py
```

**Good luck! 📈**
