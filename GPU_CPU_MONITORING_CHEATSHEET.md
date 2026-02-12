# GPU/CPU Monitoring Cheatsheet for Training

Quick reference commands to monitor your multi-agent TD3 training processes.

---

## 🎮 GPU Status Commands

### Basic GPU overview
```bash
# Quick GPU status (utilization, memory, temperature)
nvidia-smi

# Continuous monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Compact view with just the essentials
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
```

### Per-process GPU memory usage
```bash
# Show which process is using how much GPU memory
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Detailed per-process view
nvidia-smi pmon -c 1
```

### GPU utilization over time
```bash
# Log GPU stats every 5 seconds to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 5 > gpu_log.csv

# Watch with custom columns
watch -n 2 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
```

---

## 🚀 Process Monitoring Commands

### Find your training processes
```bash
# List all python training processes with PID
ps aux | grep python | grep marl_train

# More detailed view with full command
ps -ef | grep marl_train

# Count how many training processes are running
ps aux | grep marl_train | grep -v grep | wc -l
```

### CPU and Memory usage per process
```bash
# Show specific training processes with resource usage
top -b -n 1 | grep python

# Interactive top filtered for python processes
top -c -p $(pgrep -d',' -f marl_train)

# Detailed process info (replace PID with actual process ID)
top -b -n 1 -p 1042832,1044711,1048084
```

### Process runtime and statistics
```bash
# Get start time and elapsed time for process (replace PID)
ps -p 1042832 -o pid,etime,cmd

# Get all training process PIDs and their runtime
ps -eo pid,etime,cmd | grep marl_train
```

---

## 💾 System Resource Overview

### Memory status
```bash
# Overall memory usage (human-readable)
free -h

# More detailed memory breakdown
cat /proc/meminfo | head -20

# Per-process memory usage (sorted by memory)
ps aux --sort=-%mem | head -20
```

### CPU load and cores
```bash
# CPU load averages (1, 5, 15 minutes)
uptime

# Number of CPU cores
nproc

# Detailed CPU usage per core
mpstat -P ALL 1 1

# Top CPU-consuming processes
ps aux --sort=-%cpu | head -20
```

---

## 🔍 Diagnostic Commands

### Check if processes are stuck
```bash
# Monitor process I/O wait (high 'wa' indicates I/O bottleneck)
iostat -x 1 5

# Check if processes are blocked on I/O
ps aux | grep marl_train | awk '{print $8}'
# Look for 'D' state (uninterruptible sleep = I/O wait)
```

### Memory pressure
```bash
# Check swap usage (high swap = memory pressure)
swapon --show
free -h | grep Swap

# Check for OOM (Out of Memory) killer events
dmesg | grep -i "out of memory"
dmesg | grep -i "killed process"
```

### GPU throttling check
```bash
# Check if GPU is thermal throttling
nvidia-smi --query-gpu=temperature.gpu,clocks.current.graphics,clocks.max.graphics --format=csv

# Detailed GPU clock and throttling reasons
nvidia-smi -q -d CLOCK,PERFORMANCE
```