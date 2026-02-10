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

---

## 📊 One-Liners for Quick Checks

### Combined GPU + Process view
```bash
# Show GPU usage and training processes side-by-side
nvidia-smi && echo "==== Training Processes ====" && ps aux | grep marl_train | grep -v grep
```

### Training throughput estimate
```bash
# Compare runtime between processes to see if new ones are slower
ps -eo pid,etime,cmd | grep marl_train | sort -k2
```

### Quick health check
```bash
# All-in-one system status
echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader && \
echo "=== CPU Load ===" && uptime && \
echo "=== Memory ===" && free -h && \
echo "=== Training Processes ===" && ps aux | grep marl_train | grep -v grep | wc -l
```

---

## 🛑 Process Management

### Kill specific training process
```bash
# Kill by PID (replace with actual PID)
kill 1042832

# Force kill if not responding
kill -9 1042832

# Kill all training processes (USE WITH CAUTION!)
pkill -f marl_train

# Kill all python processes with specific script name
pkill -f marl_train_obstacle_6robots
```

### Check process niceness (priority)
```bash
# View process priority (lower nice = higher priority)
ps -eo pid,ni,cmd | grep marl_train

# Reduce priority of process (make it "nicer" to other processes)
renice +10 -p 1042832
```

---

## 📈 Logging and Monitoring Scripts

### Continuous GPU monitoring to file
```bash
# Log GPU stats every 10 seconds
while true; do 
    echo "$(date) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> gpu_monitor.log
    sleep 10
done
```

### Monitor training progress from logs
```bash
# Watch latest training output (if using tmux/screen)
tail -f /path/to/training.log

# Count epochs completed per process (if logging to file)
grep "Epoch" training.log | tail -20
```

---

## 🔧 Advanced Diagnostics

### Check CPU affinity (which cores process uses)
```bash
# See which CPU cores a process is using
taskset -cp 1042832
```

### Memory leak detection
```bash
# Monitor process memory over time (update every 2 seconds)
watch -n 2 'ps -p 1042832,1044711,1048084 -o pid,vsz,rss,cmd'
```

### Profile GPU usage per second
```bash
# Sample GPU utilization 10 times per second for 30 seconds
nvidia-smi dmon -c 300 -s u
```

---

## 💡 Pro Tips

1. **Use `tmux` or `screen`**: Run training in detached sessions to survive disconnections
   ```bash
   tmux new -s training1
   # Run your training here
   # Ctrl+B then D to detach
   tmux attach -t training1  # to reattach
   ```

2. **Redirect output**: Save training logs
   ```bash
   python -m robot_nav.marl_train_obstacle_6robots > train_6robots.log 2>&1 &
   ```

3. **Background with nohup**: Keep processes running after logout
   ```bash
   nohup python -m robot_nav.marl_train_obstacle_6robots > train.log 2>&1 &
   ```

4. **Check system uptime**: See if restart is needed
   ```bash
   uptime  # Shows load average - if consistently >80% of CPU cores, system is stressed
   ```

---

## 🚨 When to Restart?

Consider restarting if you see:
- GPU memory leak (usage grows over time without plateau)
- System memory < 5% free consistently
- Load average > 2x number of CPU cores for extended period
- Processes in 'D' state (I/O wait) for hours
- OOM killer events in `dmesg`
- Training speed drops significantly without code changes
