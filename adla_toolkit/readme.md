# Amlogic Python Runtime – Quick Start

This guide helps developers quickly run their **first model inference** on an Amlogic NPU using the Python Runtime.

---

## 1. Prerequisites

### Host PC

- OS: **Ubuntu 20.04**
- Python: **3.10**
- Tools: `adb`, `conda` (Anaconda or Miniforge)

### Target Device

- Amlogic SoC with NPU support
- Android / Linux / Yocto system
- Working `nnserver` binary matching the NPU driver version

---

## 2. Python Environment Setup (Host PC)

```bash
# Create and activate Python environment
conda create -n nnserver_310 python=3.10
conda activate nnserver_310
```

Install Python Runtime and dependencies:

```bash
pip install amlnn-<version>.whl
pip install -r requirements.txt
```

---

## 3. Device Connection Check

Verify the device is reachable:

```bash
adb devices -l
adb shell
```

---

## 4. Deploy nnserver to Device

Select the correct `nnserver` binary for your platform.

### Android

```bash
adb root
adb push nnserver /data/local/tmp
adb shell
cd /data/local/tmp
chmod +x nnserver
export LD_LIBRARY_PATH=/vendor/lib64
./nnserver
```

### Linux / Yocto

```bash
adb push nnserver /usr/bin
adb shell
cd /usr/bin
chmod +x nnserver
./nnserver
```

Expected output:

```
nnserver start to work, listening on port 8308
```

---

## 5. Prepare a Model

Supported formats:

- **ADLA** (offline converted, recommended)
- **Quantized TFLite** (online conversion supported)

Example path:

```
example/01_export_model/mobilenet_v2_quant.adla
```

---

## 6. Run Your First Inference (MobileNetV2)

```bash
python mobilenetv2.py \
  --board-work-path /data/local/tmp \
  --model-path ../01_export_model/mobilenet_v2_quant.adla \
  --run-cycles 1 \
  --loglevel INFO
```

On success, logs include:

- Hardware platform information
- Tensor input / output details
- Average inference latency
- FPS
- NPU bandwidth usage

---

## 7. Minimal Python Example

```python
from amlnn import AMLNN
import numpy as np

amlnn = AMLNN()

amlnn.config(
    board_work_path='/data/local/tmp',
    model_path='mobilenet_v2_quant.adla',
    loglevel='INFO'
)

amlnn.init()

inputs = [np.random.rand(1, 224, 224, 3).astype(np.float32)]
outputs = amlnn.inference(inputs)

amlnn.uninit()
```

---

## 8. Visualization (Optional)

Generate layer-level performance reports:

```python
amlnn.visualize()
```

HTML files will be created in the current directory, showing:

- Per-layer latency
- DRAM read/write bandwidth
- Hardware vs software operator distribution

---

## 9. Troubleshooting Tips

- Use unique filenames for different model versions
- Set `loglevel=DEBUG` for detailed logs
- Increase `run_cycles` to better observe NPU utilization

---

You are now ready to evaluate and benchmark models on Amlogic NPU using the Python Runtime.

