#!/bin/bash

PROJECT_DIR=~/work/detection-bidonvilles
git clone https://github.com/InseeFrLab/detection-bidonvilles.git $PROJECT_DIR
cd $PROJECT_DIR

git config --global credential.helper store

pre-commit install

AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY_ID onyxia-kv/projet-slums-detection/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_ACCESS_KEY onyxia-kv/projet-slums-detection/s3` && export AWS_SECRET_ACCESS_KEY
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

chown -R onyxia:users $PROJECT_DIR/
