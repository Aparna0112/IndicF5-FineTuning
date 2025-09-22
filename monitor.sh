#!/bin/bash
# monitor.sh - System monitoring script

echo "=== IndicF5 Malayalam Training Monitor ==="
echo "Current time: $(date)"
echo ""

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | while read line; do
    echo "GPU: $line"
done
echo ""

echo "=== Memory Usage ==="
free -h
echo ""

echo "=== Disk Usage ==="
df -h | grep -E "(Filesystem|/dev/)"
echo ""

echo "=== Process Status ==="
ps aux | grep -E "(python|train)" | grep -v grep
echo ""

echo "=== Training Progress ==="
if [ -f "logs/training.log" ]; then
    echo "Latest training logs:"
    tail -n 10 logs/training.log
else
    echo "No training log found yet."
fi
echo ""

echo "=== Checkpoints ==="
if [ -d "checkpoints/indicf5_malayalam" ]; then
    ls -la checkpoints/indicf5_malayalam/ | head -10
    echo ""
    echo "Checkpoint count: $(ls checkpoints/indicf5_malayalam/*.pt 2>/dev/null | wc -l)"
else
    echo "No checkpoints directory found yet."
fi
echo ""

echo "=== Dataset Status ==="
if [ -f "data/dataset_stats.json" ]; then
    echo "Dataset statistics:"
    cat data/dataset_stats.json
else
    echo "Dataset not prepared yet."
fi
