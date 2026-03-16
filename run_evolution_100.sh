#!/bin/bash

# Run 100-generation evolution test
# This script monitors and logs the evolution process

echo "🧬 Starting 100-Generation Evolution Test"
echo "=========================================="

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Cargo.toml not found. Please run from microgpt-evolve-rust directory."
    exit 1
fi

echo "✅ Rust environment ready"

# Create results directory
mkdir -p results
mkdir -p experiments

# Build the evolution binary
echo ""
echo "🔨 Building evolution binary..."
cargo build --release --bin evolve_loss_config

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build successful"

# Record system info
echo ""
echo "💾 System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4}')"

# Start monitoring in background
echo ""
echo "📊 Starting system monitoring..."
monitor_pid=0
monitor_results() {
    while kill -0 $1 2>/dev/null; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        mem_usage=$(free | grep '^Mem:' | awk '{printf "%.1f", $3/$2 * 100.0}')
        echo "$timestamp,$cpu_usage,$mem_usage" >> results/system_monitor_100.csv
        sleep 30
    done
}

# Start the evolution
echo ""
echo "🚀 Starting 100-generation evolution..."
echo "This may take 30-60 minutes depending on your hardware"
echo ""

# Start monitoring
monitor_results $$ &
monitor_pid=$!

# Run evolution with timestamp
start_time=$(date +%s)
timestamp=$(date '+%Y%m%d_%H%M%S')

cargo run --release --bin evolve_loss_config -- evolution_config_100.json 2>&1 | tee "results/evolution_100_${timestamp}.log"

end_time=$(date +%s)
duration=$((end_time - start_time))

# Stop monitoring
kill $monitor_pid 2>/dev/null

echo ""
echo "🎉 Evolution Complete!"
echo "======================"
echo "Duration: $((duration / 60)) minutes $((duration % 60)) seconds"
echo "Results saved to: results/evolution_100_${timestamp}.log"
echo "System monitoring: results/system_monitor_100.csv"

# Show best result if available
if [ -f "genome.json" ]; then
    echo ""
    echo "🏆 Best Result:"
    cat genome.json | jq '.'
fi

echo ""
echo "📈 To analyze results:"
echo "  cat results/evolution_100_${timestamp}.log | grep 'Best:'"
echo "  tail -n 20 results/evolution_100_${timestamp}.log"
