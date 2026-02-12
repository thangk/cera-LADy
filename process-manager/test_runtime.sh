#!/bin/bash

# Simple test script to demonstrate the runtime counter
status_file="/tmp/test_status.log"

# Create initial status
cat > "$status_file" << EOF
===============================================================================
🧪 TEST EXPERIMENT
===============================================================================
🟡 STATUS: RUNNING
🆔 PID: $$
⏰ START_TIME: $(date)
⏱️  RUNTIME: 000h 00m 00s
===============================================================================
EOF

echo "Status file created: $status_file"
echo "Watch the runtime update: watch -n 1 cat $status_file"

# Function to update runtime
update_runtime() {
    local start_time="$1"
    local status_file="$2"
    
    while true; do
        if ! kill -0 $$ 2>/dev/null; then
            break
        fi
        
        current_time=$(date +%s)
        runtime_seconds=$((current_time - start_time))
        
        hours=$((runtime_seconds / 3600))
        minutes=$(((runtime_seconds % 3600) / 60))
        seconds=$((runtime_seconds % 60))
        
        runtime_formatted=$(printf "%03dh %02dm %02ds" $hours $minutes $seconds)
        
        if [[ -f "$status_file" ]]; then
            sed -i "s/⏱️  RUNTIME: .*/⏱️  RUNTIME: $runtime_formatted/" "$status_file"
        fi
        
        sleep 1
    done
}

# Cleanup function
cleanup() {
    if [[ -n "$runtime_updater_pid" ]]; then
        kill "$runtime_updater_pid" 2>/dev/null
    fi
    
    # Final runtime update
    end_time=$(date +%s)
    total_runtime=$((end_time - start_time))
    hours=$((total_runtime / 3600))
    minutes=$(((total_runtime % 3600) / 60))
    seconds=$((total_runtime % 60))
    
    sed -i "s/⏱️  RUNTIME: .*/⏱️  RUNTIME: $(printf "%03dh %02dm %02ds" $hours $minutes $seconds)/" "$status_file"
    
    cat >> "$status_file" << EOF

===============================================================================
✅ TEST COMPLETED!
===============================================================================
🟢 STATUS: COMPLETED
⏰ END_TIME: $(date)
⏱️  TOTAL_RUNTIME: $(printf "%03dh %02dm %02ds" $hours $minutes $seconds)
===============================================================================
EOF
    echo "Test completed! Final status in: $status_file"
}

trap cleanup EXIT

# Start runtime updater
start_time=$(date +%s)
update_runtime "$start_time" "$status_file" &
runtime_updater_pid=$!

echo "Runtime updater started (PID: $runtime_updater_pid)"
echo "Running test for 10 seconds..."

# Simulate work
sleep 10

echo "Test finished!"