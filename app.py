from fastapi import FastAPI
import subprocess
import threading
import os

app = FastAPI()

training_status = {"status": "Starting up..."}

def run_training():
    global training_status
    training_status["status"] = "Training in progress. Check Space Logs for Unsloth output..."
    try:
        # Run the training script
        process = subprocess.run(
            ["python", "train.py"],
            capture_output=False, # Let it stream to Space logs
            text=True
        )
        if process.returncode == 0:
            training_status["status"] = "Training completed successfully! Model should be pushed to Hub."
        else:
            training_status["status"] = f"Training failed with exit code {process.returncode}. Check logs."
    except Exception as e:
        training_status["status"] = f"Error starting training: {str(e)}"

@app.on_event("startup")
def startup_event():
    # Start training in a background thread so FastAPI can bind to 7860 immediately
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

@app.get("/")
def get_status():
    return training_status
