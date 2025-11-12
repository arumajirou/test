import mlflow, pandas as pd
from src.auto_model_factory import create_auto_model

mlflow.set_tracking_uri("file:///C:/test/mlruns")
mlflow.set_experiment("neuralforecast_demo")

n=200
df = pd.DataFrame({
  "unique_id": ["A"]*n,
  "ds": pd.date_range("2023-01-01", periods=n, freq="D"),
  "y": [0.1*i for i in range(n)]
})

with mlflow.start_run(run_name="smoke_optuna"):
    m = create_auto_model("NHITS", h=7, dataset=df,
                          backend="optuna", num_samples=1,
                          use_mlflow=True, verbose=True)
    pred = m.predict(dataset=df)
    pred.to_csv("predictions_optuna.csv", index=False)
    mlflow.log_artifact("predictions_optuna.csv")
    print("OK: smoke_optuna")
