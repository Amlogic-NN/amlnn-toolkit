# AMLNNLite & AMLLM: Amlogic Edge AI Inference Toolkits (Python)

This repository provides high-performance, developer-friendly Python toolkits for neural network inference on Amlogic NPU platforms:
- **AMLNNLite**: Specialized for Computer Vision (CV) models.
- **AMLLM**: Specialized for Large Language Models (LLM).

These toolkits abstract complex C-based SDK interfaces into streamlined Pythonic APIs, enabling rapid deployment, performance profiling, and model optimization in edge Ubuntu environments.


## 🚀 Key Features

- **Developer-Centric API**: Simplified workflow for model loading, configuration, and inference.
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
| **LLMs (Qwen/Gemma/Llama)** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |

---

### 💻 Environment Setup

#### 1. Prerequisites
- **OS**: Ubuntu 22.04 (aarch64)
- **Python**: 3.10
- **NPU Driver**: Version 1.7.x or higher

#### 2. Verify NPU Driver
Run the following commands on the board:
```bash
dmesg | grep adla
strings /usr/lib/libadla.so | grep LIBADLA
```
*Note: Driver version must be 1.7.x or higher.*

#### 3. Initialize Python Environment (Recommended: Miniforge)
```bash
# Install Miniforge if needed
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

# Create and activate environment
conda create -n amlnn_dev python=3.10 -y
conda activate amlnn_dev
```

#### 4. Install Toolkits
Depending on your needs, install the appropriate `.whl` package:
```bash
# For Vision models (CV)
pip install amlnnlite-x.x.x-cp310-cp310-linux_aarch64.whl

# For Large Language Models (LLM)
pip install amlllm-x.x.x-cp310-cp310-linux_aarch64.whl
```

---

## ⚡ Quick Start

### 1. Vision Models (AMLNNLite)
Incorporate NPU inference into your Python application in just a few lines:

```python
from amlnnlite.api import AMLNNLite
import numpy as np

# 1. Initialize
amlnn = AMLNNLite()

# 2. Configure (adla or quantized tflite)
amlnn.config(board_work_path="/data/nn", model_path="model.adla", loglevel="INFO")

# 3. Setup Engine
amlnn.init()

# 4. Perform Inference
input_data = [np.random.randn(1, 224, 224, 3).astype(np.uint8)]
outputs = amlnn.inference(input_data)

# 5. Profile (Optional)
amlnn.visualize()

# 6. Cleanup
amlnn.uninit()
```

### 2. Large Language Models (AMLLM)
Sample script for interactive LLM chat:

```python
from amlllm.api import AMLLLM

# 1. Initialize
amlllm = AMLLLM()

# 2. Configure
amlllm.config(
    model_path="qwen2.5_0.5b.adla",
    tokenizer_path="tokenizer.json",
    sampling_mode="argmax",
    loglevel="INFO"
)

# 3. Setup Engine
amlllm.init()

# 4. Perform Inference (Chat)
result = amlllm.run(prompt="Hello, who are you?", run_mode="generate")
print(result)

# 5. Cleanup
amlllm.uninit()
```

---

## 📖 API Reference

### 1. Vision API Summary (`AMLNNLite`)

#### `config(board_work_path, model_path, run_cycles=1, loglevel="ERROR")`
Configure the runtime environment.
- `board_work_path`: Workspace on the board.
- `model_path`: Path to `.adla` or quantized `.tflite`.
- `run_cycles`: Number of iterations for profiling.

#### `inference(inputs, inputs_data_format='NHWC', outputs_data_format='NHWC', dequantize_outputs=True)`
Execute synchronous inference.
- Handles padding/strips automatically.
- Supports on-the-fly format conversion and dequantization.

#### `visualize()`
Generates comprehensive performance reports (HTML) for the last inference session.

### 2. LLM API Summary (`AMLLLM`)

#### `config(model_path, tokenizer_path, sampling_mode='argmax', top_k=50, top_p=0.9, temperature=1.0, repeat_penalty=1.1, loglevel='ERROR', on_token=None)`
Configure the LLM runtime.
- `on_token`: Optional callback function for streaming output.

#### `run(prompt, input_type='prompt', run_mode='generate', retain_history=False, user_data=None)`
Execute LLM inference.
- `retain_history`: Set to `True` for multi-turn conversations.

#### `reset_session()`
Clear conversation history and reset context.

#### `set_chat_template(system_prompt, prompt_prefix, prompt_postfix)`
Configure the chat template (System prompt, prefixes, and suffixes).

---

## 🔍 Advanced Features & Insights

### 1. Vision Layer-wise Visualization
Using `amlnn.visualize()`, developers can inspect:
- Hardware-accelerated operators (`hard_op_chart.html`)
- CPU fallback operators (`soft_op_chart.html`)
- Memory bandwidth and time distribution.

<div align="center">
  <img src="../assets/image-20251219144855741.png" width="48%" alt="Hard OP Chart" style="border-radius: 8px; margin-right: 2%;">
  <img src="../assets/image-20251219145742364.png" width="48%" alt="Netron OP ID Mapping" style="border-radius: 8px;">
</div>

### 2. NPU Utilization Monitoring
Monitor hardware load in real-time during heavy inference tasks:
```bash
python NPU_utilization.py
```

<div align="center">
  <img src="../assets/wps3.jpg" width="80%" alt="NPU Utilization Monitor" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</div>

---

## 🛠 FAQ & Troubleshooting

- **Data Formats (Vision)**: Native NPU format is NHWC. The toolkit handles NCHW ↔ NHWC conversion automatically.
- **Debugging**: 
  - For **Vision**: Set `export NN_SERVER_LOG_LEVEL=4` and `loglevel='DEBUG'`.
  - For **LLM**: Set `export LLM_SDK_LOG_LEVEL=4` and `loglevel='DEBUG'`.
- **Model Conversion**: Please refer to the **LLM Conversion User Guide** for detailed model quantization and deployment steps.
