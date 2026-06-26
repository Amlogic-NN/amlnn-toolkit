# AMLLLMLite: Amlogic Edge LLM Inference Toolkit (Python)

`amlllm_edge_toolkit_lite` provides a lightweight Python SDK for large language model (LLM) inference on Amlogic NPU platforms. It is designed to run **directly on Debian boards** — no host PC or ADB connection required.

## Key Features

- **On-device LLM inference**: Runs entirely on the Amlogic Debian board.
- **Streaming output**: Token-by-token generation via callback.
- **Multi-turn conversation**: Optional history retention across calls.
- **Flexible sampling**: Argmax, Top-P, and Top-K sampling modes.
- **Chat template support**: Built-in templates for Qwen, Llama, DeepSeek, Gemma, and more.

---

## Supported Models

| Model | Repository Link |
| :--- | :--- |
| **Qwen / InternVL** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **DeepSeek** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **Gemma / Gemma3** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **Llama** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **MiniCPM4** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **Phi-1.5 / Phi-2** | [View on GitHub](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |

---

## Environment Setup

### Prerequisites

- **OS**: Debian
- **Python**: 3.10
- **NPU Driver**: 2.0.2 or higher

> If multiple Python versions are installed, use **Miniforge** or **Anaconda** to manage environments.

### 1. Verify NPU Driver

Check the NPU driver version on the target device. Version **2.0.2 or higher** is required.

```bash
dmesg | grep adla
strings /usr/lib/libadla.so | grep LIBADLA
```

Expected output:

```
adla kmd version: 2.0.2.x.x
LIBADLA, v2.0.2.x.x.x, 20xx.xx
```

> **IMPORTANT**: If the driver version is too old, re-flash the device with the latest image. Contact your FAE or after-sales team for the image file.

### 2. Initialize Python Environment (Recommended: Miniforge)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

conda create -n amlllmlite_py310 python=3.10 -y
conda activate amlllmlite_py310
```

### 3. Install the Wheel

```bash
pip install amlnn_toolkit_lite/amlllm_edge_toolkit_lite/whl/aarch64/amlllm_edge_toolkit_lite-1.1.0-cp310-cp310-linux_aarch64.whl
```

---

## Quick Start

```bash
cd amlnn_toolkit_lite/amlllm_edge_toolkit_lite/example/base_api

python base_api_llmlite.py \
  --model /path/to/model.adla \
  --model-type qwen
```

Upon success you will enter an interactive chat interface where you can type prompts directly.

---

## Demo Overview

The `base_api_llmlite.py` demo covers the LLM inference workflow on a pre-compiled `.adla` model:

1. **Config**: Initializes `AMLLLMLite` with model path and sampling parameters.
2. **Init runtime**: Loads the model and initializes the NPU runtime.
3. **Set chat template** (optional): Applies a built-in template based on `--model-type`.
4. **Interactive inference**: Enters an interactive loop, accepting user prompts and running streaming inference with token-by-token output.

### Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--model` | required | Path to the compiled `.adla` LLM model file |
| `--model-type` | `none` | Built-in chat template to apply. See table below. |
| `--sampling-mode` | `argmax` | Sampling mode: `argmax`, `top_p`, or `top_k` |
| `--top-k` | `3` | Top-K sampling parameter |
| `--top-p` | `0.9` | Top-P sampling threshold |
| `--temperature` | `1.0` | Softmax temperature |
| `--repeat-penalty` | `1.1` | Repeat token penalty |
| `--log-level` | `ERROR` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Interactive Commands

| Command | Description |
| :--- | :--- |
| `exit` | Quit the demo |
| `new_talk` | Clear conversation history and start a new session |
| `break` | Interrupt the current generation |

### Built-in Chat Templates

The following model types have built-in templates. Pass `--model-type` when running the demo, or call `set_chat_template()` manually for custom templates.

| Model Type | Supported |
| :--- | :---: |
| `qwen` (Qwen, InternVL) | ✓ |
| `deepseek` | ✓ |
| `gemma` / `gemma3` | ✓ |
| `llama` | ✓ |
| `tiny_llama` / `tiny_llama_v0_4` | ✓ |
| `phi_1_5` / `phi_2` | ✓ |
| `minicpm4` | ✓ |

---

## FAQ

- **Streaming output**: Implement the `on_token` callback to print tokens as they arrive. The `status` field in the token dict indicates `NORMAL`, `FINISH`, or `ERROR`.
- **Multi-turn chat**: Set `retain_history=True` in `run()`. Call `reset_session()` to start a new conversation.
- **Model conversion**: Use `amlnn_toolkit` to convert and compile your LLM to `.adla` format before running inference here.
