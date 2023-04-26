#! /bin/bash

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_EXPERIMENT_NAME="segmentation"

# Set MLFLOW_RUN_NAME environment variable
export MLFLOW_RUN_NAME="testraya_1"

# Set MLFLOW_EXPERIMENT_NAME environment variable
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

# Set MLFLOW_TRACKING_URI environment variable
GET_PODS=`kubectl get pods`

while IFS= read -r line; do
    VAR=`echo "${line}" | sed -n "s/.*mlflow-\([0-9]\+\)-.*/\1/p"`
    if [ -z "$VAR" ]; then
        :
    else
        POD_ID=$VAR
    fi
done <<< "$GET_PODS"

export MLFLOW_TRACKING_URI="https://projet-slums-detection-$POD_ID.user.lab.sspcloud.fr"

mlflow run ~/work/detection-bidonvilles/ --entry-point segmentation --env-manager=local \
-P remote_server_uri=$MLFLOW_TRACKING_URI \
-P experiment_name=$MLFLOW_EXPERIMENT_NAME \
-P run_name=$MLFLOW_RUN_NAME
