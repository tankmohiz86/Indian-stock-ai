"""
Indian Stock Market AI Prediction System
Complete pipeline: Data Collection -> Feature Engineering -> Model Training -> Prediction -> Backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
import json
import pickle
warnings.filterwarnings('ignore')

class IndianStockPredictor:
    """
    AI-powered stock prediction system for Indian markets
    """
    
    def __init__(self, stock_list=None):
        """
        Initialize the predictor
        
        Args:
            stock_list: List of NSE stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS'])
        """
        self.stock_list = stock_list or self._get_nifty50_stocks()
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.results = {}
        
    def _get_nifty50_stocks(self):
        """Get top Nifty 50 stocks for Indian market"""
        # Top liquid stocks from NSE (add .NS suffix for Yahoo Finance)
        nifty50 = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'TITAN.NS', 'BAJFINANCE.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS',
            'HCLTECH.NS', 'TECHM.NS', 'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS',
            'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'TATASTEEL.NS', 'ADANIENT.NS',
            'COALINDIA.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 'CIPLA.NS', 'GRASIM.NS',
            'DRREDDY.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS', 'DIVISLAB.NS', 'HINDALCO.NS',
            'APOLLOHOSP.NS', 'BRITANNIA.NS', 'SBILIFE.NS', 'ADANIPORTS.NS', 'TATACONSUM.NS',
            'BPCL.NS', 'UPL.NS', 'HDFCLIFE.NS', 'BAJAJ-AUTO.NS', 'SHREECEM.NS'
        ]
        return nifty50
    
    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetch historical stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data for {symbol}")
                return None
                
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with additional technical indicator columns
        """
        df = df.copy()
        
        # Price-based features
        df['Returns_1D'] = df['Close'].pct_change(1)
        df['Returns_5D'] = df['Close'].pct_change(5)
        df['Returns_20D'] = df['Close'].pct_change(20)
        df['Returns_60D'] = df['Close'].pct_change(60)
        
        # Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Price relative to moving averages
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility (ATR - Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Momentum
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        return df
    
    def create_labels(self, df, prediction_days=30):
        """
        Create target labels for prediction
        
        Args:
            df: DataFrame with price data
            prediction_days: Number of days ahead to predict (default 30 for monthly)
        
        Returns:
            DataFrame with labels
        """
        df = df.copy()
        
        # Future return over prediction period
        df['Future_Return'] = df['Close'].shift(-prediction_days) / df['Close'] - 1
        
        # Binary label: 1 if price goes up, 0 if down
        df['Target'] = (df['Future_Return'] > 0).astype(int)
        
        # Multi-class label for different return ranges
        df['Target_Multi'] = pd.cut(df['Future_Return'], 
                                      bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
                                      labels=['Strong_Down', 'Down', 'Neutral', 'Up', 'Strong_Up'])
        
        return df
    
    def prepare_features(self, df):
        """
        Select and prepare features for model training
        
        Args:
            df: DataFrame with all indicators
        
        Returns:
            Feature DataFrame
        """
        # Select feature columns (exclude price, volume, and target columns)
        feature_cols = [col for col in df.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
                        'Future_Return', 'Target', 'Target_Multi']]
        
        # Remove any columns with all NaN
        feature_cols = [col for col in feature_cols if df[col].notna().any()]
        
        return df[feature_cols]
    
    def build_dataset(self, start_date, end_date, prediction_days=30):
        """
        Build complete dataset for all stocks
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            prediction_days: Days ahead to predict
        
        Returns:
            Combined DataFrame for all stocks
        """
        all_data = []
        
        print(f"Fetching data for {len(self.stock_list)} stocks...")
        
        for symbol in self.stock_list:
            print(f"Processing {symbol}...", end=' ')
            
            # Fetch data
            df = self.fetch_stock_data(symbol, start_date, end_date)
            
            if df is None or len(df) < 250:  # Need sufficient history
                print("Skipped (insufficient data)")
                continue
            
            # Calculate indicators
            df = self.calculate_technical_indicators(df)
            
            # Create labels
            df = self.create_labels(df, prediction_days)
            
            # Add stock identifier
            df['Symbol'] = symbol
            
            all_data.append(df)
            print("✓")
        
        # Combine all stocks
        combined_df = pd.concat(all_data, axis=0)
        
        # Drop rows with NaN (from indicator calculations)
        combined_df = combined_df.dropna()
        
        print(f"\nTotal samples: {len(combined_df)}")
        print(f"Positive samples: {combined_df['Target'].sum()} ({combined_df['Target'].mean()*100:.1f}%)")
        
        return combined_df
    
    def train_model(self, X_train, y_train, X_val, y_val, model_type='xgboost'):
        """
        Train prediction model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_type: 'xgboost' or 'lightgbm'
        
        Returns:
            Trained model
        """
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:  # lightgbm
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        
        # Train
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 verbose=False)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test, y_test: Test data
        
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics.update(report)
        
        return metrics, y_pred, y_pred_proba
    
    def train_walk_forward(self, df, prediction_days=30, n_splits=5):
        """
        Train model using walk-forward validation (time series split)
        
        Args:
            df: Complete dataset
            prediction_days: Days to predict ahead
            n_splits: Number of time series splits
        
        Returns:
            Best model and validation results
        """
        print("\n=== Starting Walk-Forward Training ===")
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['Target']
        dates = df.index
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}/{n_splits}")
            print(f"Train: {dates[train_idx[0]]} to {dates[train_idx[-1]]}")
            print(f"Val: {dates[val_idx[0]]} to {dates[val_idx[-1]]}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            model = self.train_model(X_train_scaled, y_train, X_val_scaled, y_val, 'xgboost')
            
            # Evaluate
            metrics, _, _ = self.evaluate_model(model, X_val_scaled, y_val)
            
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
            
            fold_results.append({
                'fold': fold,
                'model': model,
                'metrics': metrics
            })
        
        # Select best model based on ROC-AUC
        best_fold = max(fold_results, key=lambda x: x['metrics']['roc_auc'])
        print(f"\nBest Model: Fold {best_fold['fold']} (ROC-AUC: {best_fold['metrics']['roc_auc']:.3f})")
        
        self.models['best'] = best_fold['model']
        return best_fold['model'], fold_results
    
    def predict_stocks(self, target_month, target_year=2026):
        """
        Predict best stocks for a target month
        
        Args:
            target_month: Target month (e.g., 2 for February)
            target_year: Target year
        
        Returns:
            DataFrame with predictions and probabilities
        """
        print(f"\n=== Predicting for {target_month}/{target_year} ===")
        
        # Get latest data (up to prediction date)
        end_date = datetime(target_year, target_month, 1) - timedelta(days=1)
        start_date = end_date - timedelta(days=365)
        
        predictions = []
        
        for symbol in self.stock_list:
            try:
                # Fetch recent data
                df = self.fetch_stock_data(symbol, start_date, end_date)
                
                if df is None or len(df) < 100:
                    continue
                
                # Calculate indicators
                df = self.calculate_technical_indicators(df)
                df = df.dropna()
                
                if len(df) == 0:
                    continue
                
                # Get latest features
                latest_features = self.prepare_features(df).iloc[-1:]
                
                # Ensure all feature columns are present
                for col in self.feature_columns:
                    if col not in latest_features.columns:
                        latest_features[col] = 0
                
                latest_features = latest_features[self.feature_columns]
                
                # Scale
                latest_scaled = self.scaler.transform(latest_features)
                
                # Predict
                probability = self.models['best'].predict_proba(latest_scaled)[0, 1]
                prediction = self.models['best'].predict(latest_scaled)[0]
                
                predictions.append({
                    'Symbol': symbol.replace('.NS', ''),
                    'Probability_Up': probability,
                    'Prediction': 'UP' if prediction == 1 else 'DOWN',
                    'Confidence_Score': max(probability, 1-probability),
                    'Latest_Close': df['Close'].iloc[-1],
                    'Latest_Date': df.index[-1].strftime('%Y-%m-%d')
                })
                
            except Exception as e:
                print(f"Error predicting {symbol}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('Probability_Up', ascending=False)
        
        return results_df
    
    def backtest_month(self, test_month, test_year, prediction_days=30):
        """
        Backtest model on a specific month
        
        Args:
            test_month: Month to test
            test_year: Year to test
            prediction_days: Days ahead predicted
        
        Returns:
            Backtest results
        """
        print(f"\n=== Backtesting {test_month}/{test_year} ===")
        
        # Define date ranges
        # Training data: up to 1 day before test month
        train_end = datetime(test_year, test_month, 1) - timedelta(days=1)
        train_start = train_end - timedelta(days=730)  # 2 years of history
        
        # Test period: the month itself
        test_start = datetime(test_year, test_month, 1)
        if test_month == 12:
            test_end = datetime(test_year + 1, 1, 1) - timedelta(days=1)
        else:
            test_end = datetime(test_year, test_month + 1, 1) - timedelta(days=1)
        
        # Actual outcome period: prediction_days after test_start
        outcome_date = test_start + timedelta(days=prediction_days)
        
        print(f"Training period: {train_start.date()} to {train_end.date()}")
        print(f"Prediction date: {test_start.date()}")
        print(f"Outcome date: {outcome_date.date()}")
        
        # Build training dataset
        train_df = self.build_dataset(train_start, train_end, prediction_days)
        
        if len(train_df) < 100:
            return {"error": "Insufficient training data"}
        
        # Train model
        X_train = self.prepare_features(train_df)
        y_train = train_df['Target']
        self.feature_columns = X_train.columns.tolist()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Split for validation
        split_idx = int(len(X_train_scaled) * 0.8)
        model = self.train_model(
            X_train_scaled[:split_idx], y_train.iloc[:split_idx],
            X_train_scaled[split_idx:], y_train.iloc[split_idx:],
            'xgboost'
        )
        
        self.models['backtest'] = model
        
        # Get predictions for test month
        predictions = []
        
        for symbol in self.stock_list:
            try:
                # Get data up to prediction date
                hist_df = self.fetch_stock_data(symbol, train_start, test_start)
                
                if hist_df is None or len(hist_df) < 100:
                    continue
                
                # Get actual outcome data
                outcome_df = self.fetch_stock_data(symbol, test_start, outcome_date)
                
                if outcome_df is None or len(outcome_df) < 2:
                    continue
                
                # Calculate indicators
                hist_df = self.calculate_technical_indicators(hist_df)
                hist_df = hist_df.dropna()
                
                if len(hist_df) == 0:
                    continue
                
                # Get latest features
                latest_features = self.prepare_features(hist_df).iloc[-1:]
                
                for col in self.feature_columns:
                    if col not in latest_features.columns:
                        latest_features[col] = 0
                
                latest_features = latest_features[self.feature_columns]
                latest_scaled = self.scaler.transform(latest_features)
                
                # Predict
                probability = model.predict_proba(latest_scaled)[0, 1]
                prediction = model.predict(latest_scaled)[0]
                
                # Calculate actual return
                start_price = outcome_df['Close'].iloc[0]
                end_price = outcome_df['Close'].iloc[-1]
                actual_return = (end_price - start_price) / start_price
                actual_direction = 1 if actual_return > 0 else 0
                
                predictions.append({
                    'Symbol': symbol.replace('.NS', ''),
                    'Predicted_Probability': probability,
                    'Predicted_Direction': prediction,
                    'Actual_Direction': actual_direction,
                    'Actual_Return': actual_return,
                    'Correct': prediction == actual_direction,
                    'Start_Price': start_price,
                    'End_Price': end_price
                })
                
            except Exception as e:
                print(f"Error backtesting {symbol}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        
        if len(results_df) == 0:
            return {"error": "No predictions generated"}
        
        # Calculate metrics
        accuracy = results_df['Correct'].mean()
        
        # Top pick performance
        top_pick = results_df.nlargest(1, 'Predicted_Probability').iloc[0]
        
        # Top 5 picks average return
        top_5 = results_df.nlargest(5, 'Predicted_Probability')
        top_5_avg_return = top_5['Actual_Return'].mean()
        
        backtest_results = {
            'month': f"{test_month}/{test_year}",
            'accuracy': accuracy,
            'total_predictions': len(results_df),
            'correct_predictions': results_df['Correct'].sum(),
            'top_pick': {
                'symbol': top_pick['Symbol'],
                'predicted_prob': top_pick['Predicted_Probability'],
                'actual_return': top_pick['Actual_Return'],
                'correct': top_pick['Correct']
            },
            'top_5_avg_return': top_5_avg_return,
            'all_predictions': results_df
        }
        
        return backtest_results
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.models.get('best'),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models['best'] = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Indian Stock Market AI Prediction System")
    print("=" * 60)
    
    # Example usage
    predictor = IndianStockPredictor()
    
    # For demonstration with a smaller set
    # predictor.stock_list = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    
    print("\nThis is the core prediction system.")
    print("Use the Jupyter notebook or run_predictor.py to execute full training and backtesting.")
