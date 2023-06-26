#!/bin/bash

PROJECT_DIR=~/work/detection-habitat-spontane
git clone https://github.com/InseeFrLab/detection-habitat-spontane.git $PROJECT_DIR
cd $PROJECT_DIR

git config --global credential.helper store

pre-commit install

AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY_ID onyxia-kv/projet-slums-detection/s3` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_ACCESS_KEY onyxia-kv/projet-slums-detection/s3` && export AWS_SECRET_ACCESS_KEY
export MC_HOST_s3=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$AWS_S3_ENDPOINT

### Create a Json file containing credentials fot GCP authentification
# Declare an array to store the GCP variable names
declare -a gcp_variables=("GCP_TYPE" "GCP_PROJECT_ID" "GCP_PRIVATE_KEY_ID" "GCP_PRIVATE_KEY" "GCP_CLIENT_EMAIL" "GCP_CLIENT_ID" "GCP_AUTH_URI" "GCP_TOKEN_URI" "GCP_AUTH_PROVIDER" "GCP_CLIENT_CERT")

# Declare an array to store the EE variable names
declare -a ee_variables=("type" "project_id" "private_key_id" "private_key" "client_email" "client_id" "auth_uri" "token_uri" "auth_provider_x509_cert_url" "client_x509_cert_url")

# Create an associative array to store variable names and their corresponding values
declare -A variables

# Loop through the GCP variable names and retrieve their values from Vault
for ((i=0; i<${#gcp_variables[@]}; i++)); do
  var_gcp="${gcp_variables[$i]}"
  var_ee="${ee_variables[$i]}"
  variables["$var_ee"]=$(vault kv get -field="$var_gcp" "onyxia-kv/projet-slums-detection/GCP")
done

# Loop through the associative array and construct the JSON string
json_string="{"
for key in "${!variables[@]}"; do
  json_string+=" \"$key\": \"${variables[$key]}\""
  # Add a comma and a newline character after each key-value pair, except for the last one
  if [[ $key != "${!variables[@]}" ]]; then
    printf -v json_string '%s,\n' "$json_string"
  fi
done

# Remove the trailing comma and close the JSON object
json_string="${json_string%,*} }"

# Write the JSON string to a file
echo "$json_string" > GCP_credentials.json

chown -R onyxia:users $PROJECT_DIR/
