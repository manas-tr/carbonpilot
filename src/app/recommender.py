import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/energy_model.joblib"
FEATURE_PATH = "models/feature_config.json"


def load_model():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def build_features(input_data):
    df = pd.DataFrame([input_data]).copy()
    df["log_job_duration"] = np.log1p(df["job_duration_sec"])
    df["log_timelimit"] = np.log1p(df["timelimit"])
    df["log_mem_req"] = np.log1p(df["mem_req"])

    df["cpu_node_ratio"] = df["cpus_req"] / (df["nodes_alloc"] + 1)
    df["duration_utilization"] = df["job_duration_sec"] * df["avgsmutilization_pct"]
    df["mem_cpu_ratio"] = df["mem_req"] / (df["cpus_req"] + 1)
    df["gpu_activity"] = df["avgmemoryutilization_pct"] * df["avgsmutilization_pct"]

    return df


def recommend(input_data):
    model, feature_cols = load_model()

    base_cpu = int(input_data["cpus_req"])
    base_mem = int(input_data["mem_req"])

    # Only recommend configs that do not go below the user's request
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

            pred_log_energy = model.predict(X)[0]
            pred_energy = float(np.expm1(pred_log_energy))

            cpu_ratio = cpus / base_cpu
            mem_ratio = mem / base_mem

            # penalty
            cpu_penalty = abs(1 - cpu_ratio)
            mem_penalty = abs(1 - mem_ratio)

            # weight penalties
            penalty = 1 + 0.5 * (cpu_penalty + mem_penalty)

            adjusted_score = pred_energy * penalty

            candidates.append({
                "cpus": cpus,
                "mem": mem,
                "pred_energy": round(pred_energy, 2),
                "adjusted_score": round(adjusted_score, 2)
            })

    candidates = sorted(candidates, key=lambda x: x["adjusted_score"])
    return candidates


if __name__ == "__main__":
    sample_input = {
        "cpus_req": 8,
        "mem_req": 32000,
        "nodes_alloc": 1,
        "priority": 100,
        "timelimit": 3600,
        "job_duration_sec": 2000,
        "avgmemoryutilization_pct": 50,
        "avgsmutilization_pct": 60
    }

    results = recommend(sample_input)

    print("\nTop recommendations:\n")
    for r in results:
        print(r)