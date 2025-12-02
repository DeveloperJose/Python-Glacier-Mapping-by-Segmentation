#!/bin/bash
"""Run Ray training orchestration on all servers."""

set -e

echo "Starting Ray training orchestration on all servers..."
echo "=================================================="

# Function to run experiments on a server
run_server() {
    local server=$1
    local ssh_host=$2
    local code_path=$3
    
    echo "Starting experiments on $server..."
    
    if [ "$server" = "desktop" ]; then
        # Local execution
        cd /home/devj/local-debian/Python-Glacier-Mapping-by-Segmentation
        nohup uv run python scripts/ray_train.py --server $server > ${server}.log 2>&1 &
        echo "$server: Started locally (PID: $!)"
    else
        # Remote execution via SSH
        ssh $ssh_host "cd $code_path && nohup uv run python scripts/ray_train.py --server $server > ${server}.log 2>&1 &"
        echo "$server: Started remotely"
    fi
}

# Read server configurations from YAML
run_server "desktop" "" "/home/devj/local-debian/Python-Glacier-Mapping-by-Segmentation"
run_server "bilbo" "jperez@bilbo" "/home/jperez/Python-Glacier-Mapping-by-Segmentation"
run_server "frodo" "jperez@frodo" "/home/jperez/Python-Glacier-Mapping-by-Segmentation"

echo ""
echo "All servers started. Monitor progress with:"
echo "  tail -f desktop.log    # Local desktop"
echo "  ssh jperez@bilbo 'tail -f bilbo.log'    # Bilbo"
echo "  ssh jperez@frodo 'tail -f frodo.log'    # Frodo"
echo ""
echo "Ray Tune dashboards:"
echo "  Desktop: http://localhost:8265"
echo "  Bilbo: ssh -L 8266:localhost:8265 jperez@bilbo, then http://localhost:8266"
echo "  Frodo: ssh -L 8267:localhost:8265 jperez@frodo, then http://localhost:8267"
echo ""
echo "MLflow tracking: https://mlflow.developerjose.duckdns.org/"