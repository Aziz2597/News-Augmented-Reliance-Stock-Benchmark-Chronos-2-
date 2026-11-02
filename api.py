from fastapi import FastAPI, HTTPException
import subprocess
import os

app = FastAPI()

@app.post("/trigger-pipeline")
def trigger_pipeline():
    try:
        print("Starting FORECAST-001 Pipeline...")
        os.makedirs("outputs", exist_ok=True)
        
        # Execute the established pipeline sequentially
        subprocess.run(["python", "feature_builder.py"], check=True)
        subprocess.run(["python", "run_forecasts.py"], check=True)
        subprocess.run(["python", "evaluate_scorecard.py"], check=True)
        subprocess.run(["python", "upload_results.py"], check=True)
        
        return {"status": "success", "message": "Pipeline executed and persisted to Supabase."}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed at step: {e.cmd}")