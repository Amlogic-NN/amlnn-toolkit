# Amlogic NN Toolkit

Welcome to the **Amlogic Neural Network Toolkit** monorepo. This repository is the central hub for Amlogic NPU development, housing the full software stack needed to deploy AI models on Amlogic platforms — from high-level Python runtimes to low-level C/C++ SDKs and reference examples.

## Repository Structure

| Component | Description | Interface | Target | Model Scale |
| :--- | :--- | :--- | :--- | :--- |
| **[amlnn_toolkit](amlnn_toolkit/readme.md)** | Full toolkit: model conversion, quantization, compilation, and inference. Supports both x86 PC (via nnserver) and Debian board. | Python | x86 PC / Debian board | All |
| **[amlnn_toolkit_lite/amlnn_edge_toolkit_lite](amlnn_toolkit_lite/amlnn_edge_toolkit_lite/readme.md)** | Lightweight Python inference SDK for Debian boards. | Python | Debian board | Small (CNN, etc.) |
| **[amlnn_toolkit_lite/amlllm_edge_toolkit_lite](amlnn_toolkit_lite/amlllm_edge_toolkit_lite/readme.md)** | Lightweight Python LLM inference SDK for Debian boards. | Python | Debian board | Large (Qwen, etc.) |
| **[amlnn_runtime/nn_runtime](amlnn_runtime/nn_runtime/readme.md)** | C/C++ inference SDK for small models. Links against `libnnsdk.so`. | C/C++ | Android / Linux | Small (CNN, etc.) |
| **[amlnn_runtime/llm_runtime](amlnn_runtime/llm_runtime/readme.md)** | C/C++ inference SDK for large language models. Links against `libllmsdk.so`. | C/C++ | Android / Linux | Large (Qwen, etc.) |

## Which Tool Should I Use?

- **"I need to convert, quantize, compile my model, and verify model accuracy."**
  Use **[amlnn_toolkit](amlnn_toolkit/readme.md)**. It provides the full toolchain to transform standard models (PyTorch / TensorFlow / ONNX) into Amlogic executable format (`.adla`). On x86 PC, accuracy is verified via nnserver; on a Debian board, no nnserver is needed.

- **"I am building a Python app on the board for small/vision models."**
  Use **[amlnn_toolkit_lite/amlnn_edge_toolkit_lite](amlnn_toolkit_lite/amlnn_edge_toolkit_lite/readme.md)**. Install the wheel directly on the board.

- **"I am building a Python app on the board for LLMs (Qwen, Llama, etc.)."**
  Use **[amlnn_toolkit_lite/amlllm_edge_toolkit_lite](amlnn_toolkit_lite/amlllm_edge_toolkit_lite/readme.md)**. Install the wheel directly on the board.

- **"I need a C/C++ integration for small/vision models."**
  Use **[amlnn_runtime/nn_runtime](amlnn_runtime/nn_runtime/readme.md)**. Link against `libnnsdk.so` and include `nnsdk2.h`.

- **"I need a C/C++ integration for LLMs."**
  Use **[amlnn_runtime/llm_runtime](amlnn_runtime/llm_runtime/readme.md)**. Link against `libllmsdk.so` and include `llmsdk.h`.

## Supported Platforms

| Platform | Android | Linux Buildroot | Linux Yocto | Linux Debian |
| :--- | :---: | :---: | :---: | :---: |
| amlnn_toolkit (x86 PC ↔ board nnserver) | ✓ | ✓ | ✓ | ✓ |
| amlnn_toolkit (Debian board) | — | — | — | ✓ |
| amlnn_toolkit_lite/amlnn_edge_toolkit_lite | — | — | — | ✓ |
| amlnn_toolkit_lite/amlllm_edge_toolkit_lite | — | — | — | ✓ |
| amlnn_runtime/nn_runtime | ✓ | ✓ | ✓ | ✓ |
| amlnn_runtime/llm_runtime | ✓ | — | ✓ | ✓ |

## Related Resources

- **[Model Playground](https://github.com/Amlogic-NN/amlnn-model-playground)**: Extensive collection of model demos and benchmarks.
- Detailed documentation for each component is in its respective subdirectory.

---

*Copyright (c) 2026 Amlogic, Inc. All rights reserved.*
