#!/bin/bash
# Launch MLflow server for experiment tracking

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Create mlruns directory if it doesn't exist
mkdir -p ./mlruns
mkdir -p ./mlflow_artifacts

# Start MLflow server
echo "Starting MLflow server on http://localhost:5000"
echo "Logs will be saved to ./mlruns"
echo "Artifacts will be saved to ./mlflow_artifacts"
echo ""
echo "Access the MLflow UI at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root ./mlflow_artifacts \
    --backend-store-uri ./mlruns