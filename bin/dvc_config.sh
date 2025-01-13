#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define download URL
FILE_URL="https://drive.google.com/file/d/1zigdBblc1XcV8OXXE8O4oMl6xx9owTpV/view?usp=drive_link"
DESTINATION="$SCRIPT_DIR/../endorse_dvc.apps.googleusercontent.com.json"

# Download the file
wget -O "$DESTINATION" "$FILE_URL"

# Extract client_id and client_secret from the JSON
CLIENT_ID=$(jq -r '.installed.client_id' "$DESTINATION")
CLIENT_SECRET=$(jq -r '.installed.client_secret' "$DESTINATION")

# Configure DVC remote
dvc remote modify gdrive gdrive_client_id "$CLIENT_ID"
dvc remote modify gdrive gdrive_client_secret "$CLIENT_SECRET"
