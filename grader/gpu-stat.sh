#!/bin/bash

tput civis
trap 'tput cnorm; exit' SIGINT

while true; do
    tput cup 0 0
    echo "=== NVIDIA GPU Monitoring - $(date +'%H:%M:%S') ==="
    nvidia-smi
    echo ""
    echo "=== GPU Processes ==="
    nvidia-smi --query-compute-apps=pid,process_name,gpu_name,used_memory --format=csv
    sleep 1
done

tput cnorm
