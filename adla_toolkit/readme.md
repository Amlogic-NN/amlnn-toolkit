# AMLNN: Amlogic Edge AI Inference Toolkit (Python) - PC Host

![python-runtime-pc](../assets/python-runtime-pc.png)

This repository provides instructions for setting up the **Amlogic Python Runtime** environment on a PC Host and running demo models on Amlogic platforms via ADB. It abstracts complex interactions into a streamlined workflow, enabling rapid deployment and testing from your development machine.

## 🚀 Key Features

- **Developer-Centric API**: Simplified workflow for model loading, configuration, and inference.
- **Remote Execution**: Run Python scripts on your PC that execute inference on the connected Amlogic board via ADB.
- **Deep Profiling**: Built-in visualization tools for layer-wise latency, bandwidth consumption, and NPU utilization.
- **Flexible Data Handling**: Automatic handling of data format conversions (NCHW ↔ NHWC) and dequantization.

---

## 🛠 Supported Examples

Accelerate your development with our curated list of model implementations:

| Model | Repository Link |
| :--- | :--- |
| **MobileNet** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/example/mobilenetv2/02_verify_python) |
| **ResNet** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/resnet/py) |
| **YOLOv11** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolov11/py) |
| **YOLOv8** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolov8/py) |
| **YOLOWorld** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yoloworld/py) |
| **YOLOX** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolox/py) |
| **RetinaFace** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/retinaface/py) |

---

## 💻 Environment Setup

### Prerequisites
- **Host OS**: Ubuntu 20.04
- **Python**: 3.10
- **ADB**: Installed and configured
- **Target Device**: Connected via USB, ADB accessible

> If multiple Python versions are installed on the host system, it is strongly recommended to manage Python environments using **Anaconda** or **Miniforge**.

### Installation

1. **Verify ADB Connection**:
   Connect the host PC to the target device via USB.
   ```bash
   adb devices -l
   adb shell echo "Connection OK"
   ```

2. **Verify NPU Driver** (on Target Device):
   Check the NPU driver version on the target device to determine the correct package.
   The driver version must be 1.7.1.x.x.x or higher.
   ```bash
   # Android
   adb shell "dmesg | grep adla"
   adb shell "strings /vendor/lib64/libadla.so | grep LIBADLA"
   
   # Linux (Buildroot / Yocto)
   adb shell "dmesg | grep adla"
   adb shell "strings /usr/lib/libadla.so | grep LIBADLA"
   ```

4. **Initialize Environment (Recommended: Miniforge)**:
   ```bash
   # Install Miniforge if needed
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh
   
   # Create Environment
   conda create -n nnserver_310 python=3.10 -y
   conda activate nnserver_310
   ```

5. **Install Python Wheel and Dependencies**:
   ```bash
   pip install amlnn-1.0.0-cp310-cp310-linux_x86_64.whl
   ```

6. **Deploy nnserver to Target**:
   Push the `nnserver` executable matching your target platform (Android/Linux, 32/64-bit) to the device.
   
   **Android**:
   
   ```bash
   # Using Android64 bit as an example
   adb root
   adb push Android/arm64-v8a/nnserver /data/nn
   adb shell "chmod +x /data/nn/nnserver"
   ```
   
   **Linux (Buildroot / Yocto) **:
   
   ```bash
   # Using Yocto64 bit as an example
   adb push Linux/aarch64-poky-linux/nnserver /data/nn/nnserver
   adb shell "chmod +x /data/nn/nnserver"
   ```

---

## ⚡ Quick Start

### 1. Start nnserver on Device
Open a **new terminal** and start `nnserver` on the device.

**Android**:
```bash
adb shell
cd /data/nn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vendor/lib64
./nnserver
```

**Linux  (Buildroot / Yocto)**:

```bash
adb shell
cd /data/nn
./nnserver
```

You should see output indicating it is listening on ports (e.g., 8308, 8309, 8310).

### 2. Run Python Inference script (on PC) to delegate infrence task to board
Run the example script on your PC. It will communicate with `nnserver` on the board.

```bash
cd example/mobilenetv2/02_verify_python

python mobilenetv2.py \
    --model-path ../01_export_model/mobilenet_v2_1.0_224_quant_xxx.adla # Choose the correct model path based on the platform type.
```

Upon successful execution, you will see hardware platform info, input/output tensors, NPU latency, and bandwidth usage.

---

## 📖 API Reference Summary

### `AMLNN()`
Initialize the toolkit core engine.

### `config(board_work_path, model_path, run_cycles=1, loglevel="ERROR")`
Configure the runtime environment.
- `board_work_path`: Workspace on the board (where `nnserver` is running or can write temp files).
- `model_path`: Path to `.adla` or quantized `.tflite` (on the PC, will be pushed to board or loaded).
- `run_cycles`: Number of iterations for profiling.

### `init()`
Load the model into the NPU and allocate hardware resources.

### `inference(inputs, inputs_data_format='NHWC', outputs_data_format='NHWC', dequantize_outputs=True)`
Execute synchronous inference.
- Handles padding/strips automatically.
- Supports on-the-fly format conversion and dequantization.

### `visualize()`
Generates comprehensive performance reports (HTML) for the last inference session on the PC.

---

## 🔍 Advanced Features & Insights

### 1. Layer-wise Visualization
Using `amlnn.visualize()`, developers can inspect:
- `hard_op_chart.html`: Hardware-accelerated operators.
- `soft_op_chart.html`: CPU fallback operators.
- `dram_rd/wr_chart.html`: Memory bandwidth analysis.
- `pie_charts_distribution.html`: Overall time distribution.

<div align="center">
  <img src="../assets/image-20251219144855741.png" width="48%" alt="Hard OP Chart" style="border-radius: 8px; margin-right: 2%;">
  <img src="../assets/image-20251219145742364.png" width="48%" alt="Netron OP ID Mapping" style="border-radius: 8px;">
</div>

### 2. NPU Utilization Monitoring
Use the provided `NPU_utilization.py` script to monitor hardware load in real-time during heavy inference tasks.

```bash
python NPU_utilization.py
```

<div align="center">
  <img src="../assets/wps3.jpg" width="80%" alt="NPU Utilization Monitor" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</div>

---

## 🛠 FAQ

- **Data Formats**: Native NPU format is NHWC. Use `inference(..., inputs_data_format='NCHW')` if your pre-processing yields NCHW; the toolkit handles the conversion efficiently.
- **Debugging**: For verbose logs, set `export NN_SERVER_LOG_LEVEL=4` (on board) and `loglevel='DEBUG'` in `config()`.
