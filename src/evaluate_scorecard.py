import pandas as pd
import numpy as np
import os

def evaluate_and_finalize_scorecard():
    print("--- Unified Scorecard Evaluation & Phase 4 News Effect Analysis ---")
    
    scorecard_path = "outputs/forecast_daily_scorecard.csv"
    if not os.path.exists(scorecard_path):
        raise FileNotFoundError(f"Could not find {scorecard_path}. Please run run_forecasts.py first.")
        
    df = pd.read_csv(scorecard_path)
    
    actuals = df['actual_close'].to_numpy(dtype=float)
    # Using naive_prediction gives us the true previous session close for every test date
    previous_close = df['naive_prediction'].to_numpy(dtype=float)
    
    models = {
        'N - Naive Baseline': df['naive_prediction'].to_numpy(dtype=float),
        'C0 - Chronos-2 prices': df['c0_prediction'].to_numpy(dtype=float),
        'C1 - Chronos-2 + news': df['c1_prediction'].to_numpy(dtype=float)
    }
    
    metrics = []
    raw_maes = {}
    
    for name, preds in models.items():
        # Unrounded Error Metrics for precise calculation
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        
        raw_maes[name] = mae
        
        # Correct Directional Accuracy (Evaluated against Naive/Previous Close)
        actual_direction = np.sign(actuals - previous_close)
        predicted_direction = np.sign(preds - previous_close)
        dir_acc = np.mean(actual_direction == predicted_direction) * 100
        
        metrics.append({
            'Configuration': name,
            'MAE (INR)': round(mae, 2),
            'RMSE (INR)': round(rmse, 2),
            'Directional Accuracy (%)': round(dir_acc, 2)
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/final_model_scorecard.csv"
    metrics_df.to_csv(output_path, index=False)
    
    print("\n--- Final Model Scorecard Table ---")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved summary to {output_path}")
    
    # Calculate SOW News Effect percentage using FULL precision MAEs
    mae_c0 = raw_maes['C0 - Chronos-2 prices']
    mae_c1 = raw_maes['C1 - Chronos-2 + news']
    
    news_effect = 100 * (mae_c0 - mae_c1) / mae_c0
    
    print("\n--- SOW News Effect Analysis ---")
    print(f"News Effect (Relative MAE Change from C0 to C1): {news_effect:.2f}%")
    
    if news_effect > 0:
        print("Interpretation: Adding news reduced MAE.")
    elif news_effect < 0:
        print("Interpretation: Adding news worsened MAE.")
    else:
        print("Interpretation: Adding news made no difference.")

if __name__ == "__main__":
    evaluate_and_finalize_scorecard()