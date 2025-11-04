# Hướng dẫn Track CPU Memory khi Load Model

## Vấn đề: OOM (Out of Memory) khi load model

Khi load model SeamlessM4T-v2-large (~2.3B parameters), bạn có thể gặp lỗi OOM do:
1. **Model quá lớn**: ~9-10GB chỉ cho weights
2. **Load pretrained weights**: Cần thêm memory tạm thời khi download và load
3. **Memory overhead**: PyTorch cần buffer thêm cho các operations

## 🔧 Tools đã tạo

### 1. `cpu_memory_tracker.py`
- Track CPU memory (RAM) usage
- Monitor cả system memory và process memory
- Cảnh báo khi memory gần đầy

### 2. `test_cpu_memory_load.py`
- Test script để track memory khi load model
- Phát hiện bottlenecks
- Generate report chi tiết

## 📊 Cách sử dụng

### Test 1: Full Model Loading Test
```bash
python test_cpu_memory_load.py --test full
```

**Output sẽ cho biết:**
- Memory trước và sau mỗi bước
- Bước nào consume nhiều memory nhất
- Có đủ RAM để load model không
- Log file: `cpu_memory_load_test.log`

### Test 2: Checkpoint Loading Test
```bash
python test_cpu_memory_load.py --test checkpoint --checkpoint path/to/checkpoint.pt
```

**Dùng khi:**
- Load từ checkpoint local thay vì HuggingFace
- Muốn test xem checkpoint có vấn đề gì không

### Test 3: Analyze Bottlenecks
```bash
python test_cpu_memory_load.py --test analyze
```

**Phân tích:**
- Memory của từng component (encoder, decoder, etc.)
- Tìm component nào tốn RAM nhất

## 📝 Tích hợp vào code training

### Cách 1: Track trong training script

```python
from cpu_memory_tracker import CPUMemoryTracker, log_cpu_memory

# Initialize tracker
cpu_tracker = CPUMemoryTracker()
cpu_tracker.start_tracking()

# Track model loading
with cpu_tracker.track("load_model"):
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)
    
    # Check memory periodically during loading
    cpu_tracker.check_memory("After model init")
    
    # Load pretrained weights
    model.load_pretrained_weights("facebook/seamless-m4t-v2-large")
    cpu_tracker.check_memory("After loading weights")

# Track data loading
with cpu_tracker.track("load_dataset"):
    dataset = load_dataset(...)
    cpu_tracker.check_memory("After dataset load")

# Track training steps
for epoch in range(num_epochs):
    with cpu_tracker.track(f"epoch_{epoch}"):
        for batch in dataloader:
            # Training step
            ...
            
            # Optional: check memory every N steps
            if step % 100 == 0:
                cpu_tracker.check_memory(f"Step {step}")

# Log final summary
cpu_tracker.log_summary()
```

### Cách 2: Quick checks

```python
from cpu_memory_tracker import log_cpu_memory, check_memory_available

# Quick check current memory
log_cpu_memory("Before model loading")

# Check if enough memory available
if not check_memory_available(required_gb=40.0):
    print("WARNING: Insufficient RAM!")
    # Take action: use smaller model, enable offloading, etc.

# Load model
model = load_model()

log_cpu_memory("After model loading")
```

## 🔍 Đọc kết quả

### Output ví dụ:

```
======================================================================
CPU MEMORY TRACKING STARTED
======================================================================
[Initial]
  System: 12.50GB used / 64.00GB total (19.5% used, 51.50GB available)
  Process: RSS=2.30GB, VMS=3.10GB

======================================================================
Starting: load_pretrained_weights
======================================================================
[load_pretrained_weights - Start]
  System: 15.20GB used / 64.00GB total (23.8% used, 48.80GB available)
  Process: RSS=5.10GB, VMS=6.50GB

[load_pretrained_weights - After download]
  System: 25.30GB used / 64.00GB total (39.5% used, 38.70GB available)
  Process: RSS=15.20GB, VMS=17.80GB

⚠️  Large memory increase: +10.10GB

[load_pretrained_weights - After loading weights]
  System: 28.50GB used / 64.00GB total (44.5% used, 35.50GB available)
  Process: RSS=18.30GB, VMS=21.20GB

======================================================================
Finished: load_pretrained_weights
======================================================================
Memory Changes:
  System Used:  15.20GB → 28.50GB (+13.30GB)
  Process RSS:  5.10GB → 18.30GB (+13.20GB)
  Process VMS:  6.50GB → 21.20GB (+14.70GB)
  System Usage: 23.8% → 44.5%
```

### Các chỉ số quan trọng:

1. **System Used**: Tổng RAM hệ thống đang dùng
2. **Process RSS** (Resident Set Size): RAM thực tế process này đang dùng ⭐ **QUAN TRỌNG NHẤT**
3. **Process VMS** (Virtual Memory Size): Virtual memory (có thể > RAM)
4. **System Usage %**: % RAM hệ thống

### Cảnh báo:

- ⚠️ **80-90%**: Caution - RAM cao
- 🚨 **>90%**: Critical - Sắp OOM!
- ⚠️ **Large increase**: Tăng >5GB trong 1 step

## 🛠️ Giải pháp khi gặp OOM

### 1. Load model từng phần (Recommended)

```python
from cpu_memory_tracker import CPUMemoryTracker
import gc
import torch

tracker = CPUMemoryTracker()

# Option 1: Load with low_cpu_mem_usage
from transformers import SeamlessM4Tv2Model

with tracker.track("load_model_low_mem"):
    model = SeamlessM4Tv2Model.from_pretrained(
        "facebook/seamless-m4t-v2-large",
        low_cpu_mem_usage=True,  # ⭐ Giảm memory usage
        device_map="auto",        # ⭐ Auto distribute to GPU/CPU
    )
```

### 2. Load checkpoint với memory mapping

```python
# Load checkpoint without loading all into RAM
checkpoint = torch.load(
    "checkpoint.pt",
    map_location='cpu',
    mmap=True  # ⭐ Memory-mapped file
)
```

### 3. Giải phóng memory sau mỗi step

```python
with tracker.track("load_model"):
    # Load model
    model = create_model()
    
    # Free memory immediately after each step
    del intermediate_variables
    gc.collect()  # Force garbage collection
    tracker.force_cleanup()
```

### 4. Sử dụng model nhỏ hơn

```python
# Thay vì seamless-m4t-v2-large (2.3B params)
# Dùng seamless-m4t-medium (1.2B params) hoặc small
model_name = "facebook/seamless-m4t-medium"
```

### 5. Enable CPU offloading

```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Create model on meta device (no memory used)
with init_empty_weights():
    model = SeamlessM4Tv2ForSpeechToTextTrain_Pivot(config)

# Load and distribute weights
model = load_checkpoint_and_dispatch(
    model,
    checkpoint="path/to/checkpoint",
    device_map="auto",  # Auto CPU/GPU distribution
    offload_folder="offload",  # Offload to disk if needed
)
```

## 📈 Monitoring trong Production

### Setup continuous monitoring:

```python
import threading
import time
from cpu_memory_tracker import CPUMemoryTracker

class ContinuousMemoryMonitor:
    def __init__(self, interval=10):
        self.interval = interval
        self.tracker = CPUMemoryTracker()
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.tracker.start_tracking()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def _monitor(self):
        while self.running:
            self.tracker.check_memory("Periodic check")
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.tracker.log_summary()

# Usage
monitor = ContinuousMemoryMonitor(interval=30)  # Check every 30s
monitor.start()

# Your training code here
...

monitor.stop()
```

## 🎯 Best Practices

1. **Track early**: Bắt đầu track từ khi import libraries
2. **Track often**: Check memory sau mỗi major step
3. **Clean up**: Del và gc.collect() sau mỗi phase
4. **Monitor logs**: Xem log file để phát hiện pattern
5. **Test trước**: Chạy test_cpu_memory_load.py trước khi training

## ⚙️ System Requirements

### Minimum để load seamless-m4t-v2-large:
- **RAM**: 32GB (recommended 64GB)
- **Free RAM**: Ít nhất 20GB free trước khi load
- **Storage**: 10GB for model cache

### Check requirements:

```python
from cpu_memory_tracker import check_memory_available
import psutil

mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / 1024**3:.1f}GB")
print(f"Available: {mem.available / 1024**3:.1f}GB")

if check_memory_available(20.0):
    print("✅ Sufficient memory")
else:
    print("❌ Insufficient memory - use smaller model or enable offloading")
```

## 📞 Troubleshooting

### Q: Script bị killed mà không có error message
**A**: Đây là OOM killer của OS. Xem log:
```bash
# Linux
dmesg | grep -i "killed process"

# Windows
# Check Event Viewer > Windows Logs > System
```

### Q: Memory không giảm sau khi del object
**A**: 
```python
import gc
del large_object
gc.collect()  # Force collection
torch.cuda.empty_cache()  # If using GPU
```

### Q: Muốn xem memory của specific Python object
**A**:
```python
import sys
obj_size_bytes = sys.getsizeof(obj)
obj_size_gb = obj_size_bytes / 1024**3
print(f"Object size: {obj_size_gb:.2f}GB")

# For tensors
if isinstance(obj, torch.Tensor):
    tensor_size_gb = obj.element_size() * obj.numel() / 1024**3
    print(f"Tensor size: {tensor_size_gb:.2f}GB")
```

---

**Ghi chú**: Sau khi chạy test, check file `cpu_memory_load_test.log` để xem chi tiết.
