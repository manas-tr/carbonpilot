from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np

# Load artifacts once at startup
MODEL_PATH = "models/energy_model.joblib"
FEATURE_PATH = "models/feature_config.json"

model = joblib.load(MODEL_PATH)
with open(FEATURE_PATH) as f:
    feature_cols = json.load(f)

app = FastAPI(title="CarbonPilot API")

class JobInput(BaseModel):
    cpus_req: int
    mem_req: int
    nodes_alloc: int
    priority: int
    timelimit: int
    job_duration_sec: float
    avgmemoryutilization_pct: float
    avgsmutilization_pct: float

def build_features(input_data):
    df = pd.DataFrame([input_data])

    df["log_job_duration"] = np.log1p(df["job_duration_sec"])
    df["log_timelimit"] = np.log1p(df["timelimit"])
    df["log_mem_req"] = np.log1p(df["mem_req"])

    df["cpu_node_ratio"] = df["cpus_req"] / (df["nodes_alloc"] + 1)
    df["duration_utilization"] = df["job_duration_sec"] * df["avgsmutilization_pct"]
    df["mem_cpu_ratio"] = df["mem_req"] / (df["cpus_req"] + 1)
    df["gpu_activity"] = df["avgmemoryutilization_pct"] * df["avgsmutilization_pct"]

    return df

def recommend_logic(input_data):
    base_cpu = input_data["cpus_req"]
    base_mem = input_data["mem_req"]

    cpu_options = [base_cpu, base_cpu * 2]
    mem_options = [base_mem, base_mem * 2]

    candidates = []

    for cpus in cpu_options:
        for mem in mem_options:
            config = input_data.copy()
            config["cpus_req"] = cpus
            config["mem_req"] = mem

            df = build_features(config)
            X = df[feature_cols]

            pred_log = model.predict(X)[0]
            pred_energy = float(np.expm1(pred_log))

            # Balanced penalty
            cpu_ratio = cpus / base_cpu
            mem_ratio = mem / base_mem

            cpu_penalty = abs(1 - cpu_ratio)
            mem_penalty = abs(1 - mem_ratio)

            penalty = 1 + 0.5 * (cpu_penalty + mem_penalty)
            adjusted_score = pred_energy * penalty

            candidates.append({
                "cpus": cpus,
                "mem": mem,
                "pred_energy": round(pred_energy, 2),
                "score": round(adjusted_score, 2)
            })

    candidates = sorted(candidates, key=lambda x: x["score"])
    return candidates


@app.get("/")
def home():
    return {"message": "CarbonPilot API is on the run!"}


@app.post("/recommend")
def recommend(job: JobInput):
    results = recommend_logic(job.dict())
    return {"recommendations": results}