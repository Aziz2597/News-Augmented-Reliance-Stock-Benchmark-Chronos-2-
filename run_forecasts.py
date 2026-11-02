import torch
import numpy as np
import pandas as pd
from datetime import datetime
from chronos import Chronos2Pipeline
import os

# Guarantee reproducible inference
torch.manual_seed(42)
np.random.seed(42)

def run_forecasting_pipeline():
    print("--- Starting Phase 3: Walk-Forward Forecasting Evaluation ---")

    # 1. Load Pinned Chronos-2 Model
    print("Loading Chronos-2 with pinned revision (float32, CPU)...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        dtype=torch.float32,
        revision="29ec3766d36d6f73f0696f85560a422f50e8498c"
    )

    # 2. Load Processed Features from Phase 2
    df = pd.read_csv("outputs/forecast_features_ready.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 3. Define Training Context vs Held-Out Test Period
    test_start_date = pd.to_datetime('2026-07-01')
    test_end_date = pd.to_datetime('2026-09-30')

    test_indices = df.index[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].tolist()
    
    if not test_indices:
        raise ValueError("No matching dates found for the held-out test period!")

    results_list = []
    os.makedirs("outputs", exist_ok=True, mode=0o777)
    output_path = "outputs/forecast_daily_scorecard.csv"

    print(f"Executing expanding-window, rolling one-step-ahead evaluation across {len(test_indices)} test sessions...")

    # 4. Walk-Forward Loop
    for idx in test_indices:
        current_row = df.loc[idx]
        current_date = current_row['date']
        actual_close = current_row['close']
        naive_pred = current_row['naive_prediction']

        # Expanding historical context strictly up to idx - 1 (preventing lookahead bias)
        history_df = df.iloc[:idx]
        
        prices = history_df['close'].values.astype(np.float32)
        log_articles = history_df['log_article_count'].values.astype(np.float32)
        mean_tones = history_df['mean_tone'].values.astype(np.float32)

        # Pre-inference Validations
        if len(prices) == 0:
            raise ValueError(f"Empty history array at index {idx}")
        if not (len(prices) == len(log_articles) == len(mean_tones)):
            raise ValueError(f"Feature array length mismatch at index {idx}")
        if not np.isfinite(prices).all() or not np.isfinite(log_articles).all() or not np.isfinite(mean_tones).all():
            raise ValueError(f"Non-finite numerical values found in history at index {idx}")

        # --- C0: Prices Only ---
        c0_tensor = torch.from_numpy(np.array([[prices]], dtype=np.float32))
        
        c0_forecast = pipeline.predict_quantiles(
            inputs=c0_tensor,
            prediction_length=1,
            quantile_levels=[0.5]
        )
        c0_pred = c0_forecast[0][0][0].item()

        # --- C1: Prices + Past News Covariates (Native Interface) ---
        c1_input = [{
            "target": prices,
            "past_covariates": {
                "log_article_count": log_articles,
                "mean_tone": mean_tones
            }
        }]
        
        c1_result = pipeline.predict_quantiles(
            inputs=c1_input,
            prediction_length=1,
            quantile_levels=[0.5]
        )
        
        # Handle Chronos-2 native tuple vs tensor return structure safely
        c1_quantiles = c1_result[0] if isinstance(c1_result, tuple) else c1_result
        c1_pred = c1_quantiles[0][0][0].item()

        if not np.isfinite(c0_pred) or not np.isfinite(c1_pred):
            raise ValueError(f"Non-finite prediction generated at index {idx}")

        # Store daily record
        results_list.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'actual_close': actual_close,
            'naive_prediction': naive_pred,
            'c0_prediction': c0_pred,
            'c1_prediction': c1_pred
        })

        print(f"Date: {current_date.strftime('%Y-%m-%d')} | Actual: ₹{actual_close:.2f} | Naive: ₹{naive_pred:.2f} | C0: ₹{c0_pred:.2f} | C1: ₹{c1_pred:.2f}")

        # Incremental saving: Write to disk after every test session
        pd.DataFrame(results_list).to_csv(output_path, index=False)

    print(f"\nWalk-forward forecasting complete! Saved daily scorecard to {output_path}.")

if __name__ == "__main__":
    run_forecasting_pipeline()