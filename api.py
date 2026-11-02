
import os
import secrets
import subprocess
import threading

from fastapi import FastAPI, HTTPException, Header

app = FastAPI(
    title="FORECAST-001 Benchmark API",
    version="1.0.0"
)

pipeline_lock = threading.Lock()


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "FORECAST-001 Benchmark API"
    }


@app.post("/trigger-pipeline")
def trigger_pipeline(
    x_api_key: str = Header(default="")
):
    # Authenticate the caller
    expected_key = os.getenv("PIPELINE_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Pipeline API key is not configured."
        )

    if not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )

    # Prevent simultaneous benchmark executions
    if not pipeline_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A benchmark pipeline is already running."
        )

    try:
        print("Starting FORECAST-001 Pipeline...")

        os.makedirs("outputs", exist_ok=True)

        scripts = [
            "feature_builder.py",
            "run_forecasts.py",
            "evaluate_scorecard.py",
            "upload_results.py"
        ]

        for script in scripts:
            print(f"Executing {script}...")

            subprocess.run(
                ["python", script],
                check=True
            )

            print(f"Completed {script}")

        return {
            "status": "success",
            "message": "Pipeline executed and persisted to Supabase."
        }

    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed at: {e.cmd}"
        )

    finally:
        pipeline_lock.release()
