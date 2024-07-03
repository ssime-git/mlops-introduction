import modal

stub = modal.Stub("mlflow-server")

mlflow_image = modal.Image.debian_slim().pip_install("mlflow")
mlflow_volume = modal.Volume.persisted("mlflow-volume")

@stub.function(
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

@stub.local_entrypoint()
def main():
    print(f"MLflow server URL: {mlflow_server.url}")