import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

def upload_benchmark_results():
    print("--- Starting Phase 5: Supabase Integration & Persistence ---")
    
    # Load environment variables from the .env file
    load_dotenv()
    
    # 1. Initialize Supabase Client (Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        raise ValueError("Missing Supabase environment variables. Please check your .env file.")
        
    supabase: Client = create_client(url, key)
    
    # 2. Load generated artifacts
    daily_df = pd.read_csv("outputs/forecast_daily_scorecard.csv")
    metrics_df = pd.read_csv("outputs/final_model_scorecard.csv")
    
    # Extract News Effect from metrics_df to store in the run metadata
    mae_c0 = metrics_df.loc[metrics_df['Configuration'] == 'C0 - Chronos-2 prices', 'MAE (INR)'].values[0]
    mae_c1 = metrics_df.loc[metrics_df['Configuration'] == 'C1 - Chronos-2 + news', 'MAE (INR)'].values[0]
    news_effect = float(100 * (mae_c0 - mae_c1) / mae_c0)
    
    if news_effect > 0:
        interpretation = "Adding news reduced MAE."
    elif news_effect < 0:
        interpretation = "Adding news worsened MAE."
    else:
        interpretation = "Adding news made no difference."
        
    # 3. Create a new benchmark run record
    run_response = supabase.table("benchmark_runs").insert({
        "status": "completed",
        "news_effect_percentage": round(news_effect, 2),
        "interpretation": interpretation
    }).execute()
    
    run_id = run_response.data[0]['id']
    print(f"Created benchmark run record: {run_id}")
    
    # 4. Format and upload daily scorecard
    daily_records = []
    for _, row in daily_df.iterrows():
        daily_records.append({
            "run_id": run_id,
            "trading_date": row['date'],
            "actual_close": float(row['actual_close']),
            "naive_prediction": float(row['naive_prediction']),
            "c0_prediction": float(row['c0_prediction']),
            "c1_prediction": float(row['c1_prediction'])
        })
        
    supabase.table("forecast_daily_scorecard").insert(daily_records).execute()
    print(f"Uploaded {len(daily_records)} daily scorecard records.")
    
    # 5. Format and upload final metrics
    metrics_records = []
    for _, row in metrics_df.iterrows():
        metrics_records.append({
            "run_id": run_id,
            "configuration": str(row['Configuration']),
            "mae": float(row['MAE (INR)']),
            "rmse": float(row['RMSE (INR)']),
            "directional_accuracy": float(row['Directional Accuracy (%)'])
        })
        
    supabase.table("final_model_scorecard").insert(metrics_records).execute()
    print("Uploaded final metrics summary. Handoff complete!")

if __name__ == "__main__":
    upload_benchmark_results()