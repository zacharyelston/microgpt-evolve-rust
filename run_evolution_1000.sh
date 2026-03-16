#!/bin/bash

# Run 1000-generation evolution test
# This is a long-running test (8-20 hours depending on hardware)

echo "🧬 Starting 1000-Generation Evolution Test"
echo "==========================================="
echo "⚠️  WARNING: This is a very long-running test!"
echo "   Expected duration: 8-20 hours"
echo "   Make sure you have stable power and internet"
echo ""

# Confirm before starting
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

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

# Check available disk space (need several GB for logs)
available_space=$(df . | tail -1 | awk '{print $4}')
if [ "$available_space" -lt 5000000 ]; then  # Less than 5GB
    echo "❌ Low disk space: ${available_space}KB available. Need at least 5GB."
    exit 1
fi

echo "✅ Environment checks passed"

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
echo "Start time: $(date)"

# Create checkpoint script
cat > checkpoint_1000.sh << 'EOF'
#!/bin/bash
# Checkpoint script for 1000-generation evolution
echo "📊 Evolution Checkpoint - $(date)"
if [ -f "genome.json" ]; then
    echo "Current best genome:"
    cat genome.json | jq '.'
fi
if pgrep -f "evolve_loss_config" > /dev/null; then
    echo "Evolution process is running"
else
    echo "Evolution process is NOT running"
fi
echo "Recent log entries:"
tail -n 10 results/evolution_1000_*.log 2>/dev/null || echo "No log files found"
EOF
chmod +x checkpoint_1000.sh

# Start comprehensive monitoring
echo ""
echo "📊 Starting comprehensive system monitoring..."
monitor_pid=0
monitor_results() {
    while kill -0 $1 2>/dev/null; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        mem_usage=$(free | grep '^Mem:' | awk '{printf "%.1f", $3/$2 * 100.0}')
        disk_usage=$(df . | tail -1 | awk '{print $5}' | cut -d'%' -f1)
        echo "$timestamp,$cpu_usage,$mem_usage,$disk_usage" >> results/system_monitor_1000.csv
        sleep 60  # Every minute for long run
    done
}

# Start the evolution
echo ""
echo "🚀 Starting 1000-generation evolution..."
echo "This will run for many hours. Use './checkpoint_1000.sh' to check progress."
echo "Logs will be saved to results/evolution_1000_*.log"
echo ""

# Start monitoring
monitor_results $$ &
monitor_pid=$!

# Run evolution with timestamp and nohup for resilience
timestamp=$(date '+%Y%m%d_%H%M%S')
log_file="results/evolution_1000_${timestamp}.log"

# Use nohup to make it resilient to disconnections
nohup cargo run --release --bin evolve_loss_config -- evolution_config_1000.json > "$log_file" 2>&1 &
evolution_pid=$!

echo "Evolution started with PID: $evolution_pid"
echo "Log file: $log_file"
echo "Monitor file: results/system_monitor_1000.csv"
echo ""
echo "Commands to check progress:"
echo "  ./checkpoint_1000.sh"
echo "  tail -f $log_file"
echo "  ps aux | grep evolve_loss_config"
echo ""
echo "To stop the evolution:"
echo "  kill $evolution_pid"

# Wait for completion or interruption
echo "Evolution running... (Press Ctrl+C to stop and show results)"
trap 'echo ""; echo "🛑 Stopping evolution..."; kill $evolution_pid 2>/dev/null; kill $monitor_pid 2>/dev/null; echo "✅ Stopped"; exit 0' INT

while kill -0 $evolution_pid 2>/dev/null; do
    sleep 300  # Check every 5 minutes
    echo "📊 Still running... ($(date))"
done

# Stop monitoring
kill $monitor_pid 2>/dev/null

# Final results
echo ""
echo "🎉 Evolution Complete!"
echo "======================"
echo "End time: $(date)"
echo "Results saved to: $log_file"
echo "System monitoring: results/system_monitor_1000.csv"

# Show best result if available
if [ -f "genome.json" ]; then
    echo ""
    echo "🏆 Best Result:"
    cat genome.json | jq '.'
fi

echo ""
echo "📈 To analyze results:"
echo "  grep 'Best:' $log_file"
echo "  tail -n 50 $log_file"
echo "  ./analyze_results.sh $log_file"
