import mlflow, pandas as pd
from ray import tune
from src.auto_model_factory import create_auto_model

mlflow.set_tracking_uri("file:///C:/test/mlruns")
mlflow.set_experiment("neuralforecast_demo")

n=200
df = pd.DataFrame({
  "unique_id": ["A"]*n,
  "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
  "y": [0.1*i for i in range(n)]
})

config = {
  "max_steps": tune.choice([500, 1000]),
  "learning_rate": tune.loguniform(1e-4, 1e-2),
  "batch_size": tune.choice([32, 64])
}

with mlflow.start_run(run_name="smoke_ray"):
    m = create_auto_model("NHITS", h=7, dataset=df,
                          backend="ray", num_samples=1, config=config,
                          cpus=2, gpus=0, use_mlflow=True, verbose=True)
    pred = m.predict(dataset=df)
    pred.to_csv("predictions_ray.csv", index=False)
    mlflow.log_artifact("predictions_ray.csv")
    print("OK: smoke_ray")
