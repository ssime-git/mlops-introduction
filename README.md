# mlops-introduction

## Setting up the environment

Let's start by setting up the environment :

```sh
# create a venv
python -m venv mlflow_env

# activate the venv
source mlflow_env/bin/activate

# install mlflow
pip install -r requirements.txt

# create metric store (the place to store our metrics)
mkdir metrics_store

# a place for the artefact
mkdire artifact_store
```

## Getting start

Now we are ready to start the MLFLOW serveur that will hold all our ML experiments.

### running MLFLOW server locally

```sh
#mlflow server --backend-store-uri sqlite:////home/mlflow/mlflow_server/metrics_store/mlflow.db --default-artifact-root /home/mlflow/mlflow_server/artifact_store/
mlflow server --backend-store-uri sqlite:///./metrics_store/mlflow.db --default-artifact-root ./artifact_store/
```

### running MLFLOW remotely on modal

When using Modal for MLflow, you don't need to set up external databases or storage services. Modal provides persistent storage that you can use for both the tracking database and artifacts. Here's how you can set it up:

1. First, create a Modal volume for persistent storage:

```python
mlflow_volume = modal.Volume.persisted("mlflow-volume")
```

2. Then, modify your MLflow server function to use this volume:

```python
import modal

app = modal.App("mlflow-server")

mlflow_image = modal.Image.debian_slim().pip_install("mlflow")
mlflow_volume = modal.Volume.from_name("mlflow-volume", create_if_missing=True)

@app.function(
    image=mlflow_image,
    volumes={"/mlflow": mlflow_volume},
    cpu=1,
    memory=1024,
)
@modal.web_endpoint(method="GET")
def mlflow_server():
    import os
    import mlflow
    from mlflow.server import get_app_for_run

    # Set up MLflow server using Modal's persistent volume
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////mlflow/mlflow.db"
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "/mlflow/artifacts"

    # Create and return the MLflow app
    return get_app_for_run()

@app.local_entrypoint()
def main():
    print(f"MLflow server URL: {mlflow_server.url}")
```

In this setup:

- `MLFLOW_TRACKING_URI` is set to a SQLite database stored in the Modal volume.
- `MLFLOW_DEFAULT_ARTIFACT_ROOT` is set to a directory in the Modal volume.

3. Deploy the MLflow server:

```sh
modal deploy mlflow_server.py
```

4. Use the MLflow server in your experiments:

```python
import mlflow
import modal

app = modal.App("mlflow-experiment")

@app.function(
    image=modal.Image.debian_slim().pip_install("mlflow"),
)
def run_experiment():
    mlflow.set_tracking_uri("https://your-modal-mlflow-server-url")
    mlflow.set_experiment("my-experiment")

    with mlflow.start_run():
        # Your experiment code here
        mlflow.log_param("param1", 1)
        mlflow.log_metric("metric1", 0.1)

@app.local_entrypoint()
def main():
    run_experiment.remote()
```

This setup uses Modal exclusively:

- The MLflow tracking server runs on Modal.
- The tracking database and artifacts are stored in a Modal volume.
- Your experiments can be run on Modal and log to the Modal-hosted MLflow server.

Remember to replace "https://your-modal-mlflow-server-url" with the actual URL Modal provides when you deploy the MLflow server.

This approach leverages Modal's infrastructure for both compute and storage, simplifying your MLflow setup and management.