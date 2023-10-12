#! /bin/bash

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_EXPERIMENT_NAME="detection"

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'
export MLFLOW_TRACKING_URI="https://projet-slums-detection-128833.user.lab.sspcloud.fr"

mlflow run ~/work/detection-habitat-spontane/ --entry-point detection --env-manager=local \
-P remote_server_uri=$MLFLOW_TRACKING_URI \
-P experiment_name=$MLFLOW_EXPERIMENT_NAME \
-P run_name=$MLFLOW_RUN_NAME
