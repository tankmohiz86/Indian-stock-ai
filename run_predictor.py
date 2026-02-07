"""
Execution Script for Indian Stock Market Prediction System
Handles training, prediction, and backtesting workflows
"""

from indian_stock_predictor import IndianStockPredictor
from datetime import datetime, timedelta
import pandas as pd
import json

def train_and_predict():
    """
    Main workflow: Train model and predict for February 2026
    """
    print("=" * 80)
    print("INDIAN STOCK MARKET AI PREDICTION SYSTEM")
    print("=" * 80)
    
    # Initialize predictor (will use Nifty 50 stocks)
    predictor = IndianStockPredictor()
    
    print(f"\nTotal stocks to analyze: {len(predictor.stock_list)}")
    
    # Step 1: Build dataset for training
    print("\n" + "=" * 80)
    print("STEP 1: BUILDING DATASET")
    print("=" * 80)
    
    # Training period: 3 years of history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # ~3 years
    
    print(f"Training period: {start_date.date()} to {end_date.date()}")
    
    df = predictor.build_dataset(
        start_date=start_date,
        end_date=end_date,
        prediction_days=30  # Predict 30 days ahead (monthly)
    )
    
    if len(df) < 1000:
        print("ERROR: Insufficient data for training")
        return
    
    # Step 2: Train model with walk-forward validation
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING MODEL (Walk-Forward Validation)")
    print("=" * 80)
    
    model, fold_results = predictor.train_walk_forward(df, prediction_days=30, n_splits=5)
    
    # Display fold results
    print("\n--- Validation Results ---")
    for result in fold_results:
        print(f"Fold {result['fold']}: "
              f"Accuracy={result['metrics']['accuracy']:.3f}, "
              f"ROC-AUC={result['metrics']['roc_auc']:.3f}")
    
    avg_accuracy = sum(r['metrics']['accuracy'] for r in fold_results) / len(fold_results)
    avg_roc_auc = sum(r['metrics']['roc_auc'] for r in fold_results) / len(fold_results)
    
    print(f"\nAverage Accuracy: {avg_accuracy:.3f}")
    print(f"Average ROC-AUC: {avg_roc_auc:.3f}")
    
    # Step 3: Predict for February 2026
    print("\n" + "=" * 80)
    print("STEP 3: PREDICTIONS FOR FEBRUARY 2026")
    print("=" * 80)
    
    predictions_df = predictor.predict_stocks(target_month=2, target_year=2026)
    
    # Display top 10 predictions
    print("\n🔥 TOP 10 STOCKS FOR FEBRUARY 2026:")
    print("=" * 80)
    
    top_10 = predictions_df.head(10)
    
    for idx, row in top_10.iterrows():
        print(f"\n#{idx+1}. {row['Symbol']}")
        print(f"   Probability of Rise: {row['Probability_Up']*100:.1f}%")
        print(f"   Prediction: {row['Prediction']}")
        print(f"   Confidence: {row['Confidence_Score']*100:.1f}%")
        print(f"   Latest Price: ₹{row['Latest_Close']:.2f} (as of {row['Latest_Date']})")
    
    # Show THE BEST pick
    best_stock = predictions_df.iloc[0]
    
    print("\n" + "=" * 80)
    print("🏆 BEST STOCK PREDICTION FOR FEBRUARY 2026")
    print("=" * 80)
    print(f"\nTicker: {best_stock['Symbol']}")
    print(f"Predicted Upside Probability: {best_stock['Probability_Up']*100:.1f}%")
    print(f"Confidence Score: {best_stock['Confidence_Score']*100:.1f}%")
    print(f"Latest Price: ₹{best_stock['Latest_Close']:.2f}")
    print(f"Prediction: {best_stock['Prediction']}")
    
    # Technical analysis summary
    print(f"\nRationale: AI model analyzed {len(df)} historical data points")
    print(f"across {len(predictor.stock_list)} stocks using {len(predictor.feature_columns)} features")
    print("including technical indicators, momentum, volatility, and volume patterns.")
    
    # Save predictions
    predictions_df.to_csv('/home/claude/predictions_feb_2026.csv', index=False)
    print("\n✓ Full predictions saved to: predictions_feb_2026.csv")
    
    # Save model
    predictor.save_model('/home/claude/stock_predictor_model.pkl')
    
    return predictor, predictions_df


def run_backtests():
    """
    Run backtests for multiple historical months
    """
    print("\n" + "=" * 80)
    print("BACKTESTING MODE")
    print("=" * 80)
    
    predictor = IndianStockPredictor()
    
    # Define months to backtest (testing on recent history)
    # Format: (month, year)
    test_months = [
        (12, 2025),  # December 2025
        (11, 2025),  # November 2025
        (10, 2025),  # October 2025
        (9, 2025),   # September 2025
        (8, 2025),   # August 2025
        (7, 2025),   # July 2025
    ]
    
    backtest_results = []
    
    for month, year in test_months:
        result = predictor.backtest_month(month, year, prediction_days=30)
        
        if 'error' in result:
            print(f"\nSkipping {month}/{year}: {result['error']}")
            continue
        
        backtest_results.append(result)
        
        # Display results
        print(f"\n--- Results for {month}/{year} ---")
        print(f"Accuracy: {result['accuracy']*100:.1f}%")
        print(f"Total Predictions: {result['total_predictions']}")
        print(f"Correct Predictions: {result['correct_predictions']}")
        print(f"\nTop Pick: {result['top_pick']['symbol']}")
        print(f"  Predicted Probability: {result['top_pick']['predicted_prob']*100:.1f}%")
        print(f"  Actual Return: {result['top_pick']['actual_return']*100:.2f}%")
        print(f"  Correct: {'✓' if result['top_pick']['correct'] else '✗'}")
        print(f"\nTop 5 Average Return: {result['top_5_avg_return']*100:.2f}%")
    
    # Summary statistics
    if backtest_results:
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        
        avg_accuracy = sum(r['accuracy'] for r in backtest_results) / len(backtest_results)
        avg_top5_return = sum(r['top_5_avg_return'] for r in backtest_results) / len(backtest_results)
        
        top_picks_correct = sum(1 for r in backtest_results if r['top_pick']['correct'])
        top_pick_accuracy = top_picks_correct / len(backtest_results)
        
        print(f"\nMonths Tested: {len(backtest_results)}")
        print(f"Average Accuracy: {avg_accuracy*100:.1f}%")
        print(f"Top Pick Accuracy: {top_pick_accuracy*100:.1f}%")
        print(f"Average Top 5 Return: {avg_top5_return*100:.2f}%")
        
        # Save backtest results
        summary_df = pd.DataFrame([
            {
                'Month': r['month'],
                'Accuracy': r['accuracy'],
                'Top_Pick_Symbol': r['top_pick']['symbol'],
                'Top_Pick_Return': r['top_pick']['actual_return'],
                'Top_5_Avg_Return': r['top_5_avg_return']
            }
            for r in backtest_results
        ])
        
        summary_df.to_csv('/home/claude/backtest_summary.csv', index=False)
        print("\n✓ Backtest summary saved to: backtest_summary.csv")
        
        # Save detailed results
        for result in backtest_results:
            month_str = result['month'].replace('/', '_')
            result['all_predictions'].to_csv(
                f'/home/claude/backtest_{month_str}.csv',
                index=False
            )
        
        print("✓ Detailed backtest results saved")
    
    return backtest_results


def quick_predict():
    """
    Quick prediction for current top stocks (no training)
    Requires pre-trained model
    """
    print("Loading pre-trained model...")
    
    predictor = IndianStockPredictor()
    
    try:
        predictor.load_model('/home/claude/stock_predictor_model.pkl')
    except:
        print("ERROR: No pre-trained model found. Please run train_and_predict() first.")
        return
    
    predictions_df = predictor.predict_stocks(target_month=2, target_year=2026)
    
    print("\n🔥 TOP 5 STOCKS FOR FEBRUARY 2026:")
    print("=" * 60)
    
    for idx, row in predictions_df.head(5).iterrows():
        print(f"\n{idx+1}. {row['Symbol']} - {row['Probability_Up']*100:.1f}% probability")
    
    return predictions_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'backtest':
            run_backtests()
        elif mode == 'quick':
            quick_predict()
        else:
            train_and_predict()
    else:
        # Default: Full training and prediction
        train_and_predict()
        
        # Ask if user wants to run backtests
        print("\n" + "=" * 80)
        response = input("Do you want to run backtests? (y/n): ")
        
        if response.lower() == 'y':
            run_backtests()
