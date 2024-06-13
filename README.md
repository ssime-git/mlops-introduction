# mlops-introduction

## Setup

```sh
# create a venv
python -m venv mlflow_env

# activate the venv
source mlflow_venv/bin/activate

# install mlflow
pip install mlflow

# create a dire (the plce to store our metrics)
mkdir metrics_store

# a place for the artefact
mkdire artifact_store

#mlflow server --backend-store-uri sqlite:////home/mlflow/mlflow_server/metrics_store/mlflow.db --default-artifact-root /home/mlflow/mlflow_server/artifact_store/
mlflow server --backend-store-uri sqlite:///./metrics_store/mlflow.db --default-artifact-root ./artifact_store/
```

## Getting start