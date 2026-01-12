# Amlogic NN Toolkit (Model Zoo & SDKs)

Welcome to the **Amlogic Neural Network Toolkit** monorepo. This repository serves as the central hub for Amlogic NPU development, housing the comprehensive software stack required to deploy AI models on Amlogic platforms—from high-level Python runtimes to low-level C/C++ SDKs and reference examples.

## 📂 Repository Structure

| Component | Description | Target Audience |
| :--- | :--- | :--- |
| **[adla_toolkit](adla_toolkit/readme.md)** | **PC Host Python Runtime**. A toolkit that runs on your Ubuntu PC and executes inference on a connected Amlogic board via ADB. Ideal for rapid development and testing without needing to work directly on the board. | Data Scientists, Algorithm Engineers |
| **[amlnn_edge_toolkit_lite](amlnn_edge_toolkit_lite/readme.md)** | **Edge Board Python Runtime**. A lightweight Python SDK designed to run directly on the Amlogic board (Ubuntu/Debian). Optimized for deployment and production inference. | Embedded Engineers, App Developers |
| **[npu_runtime](npu_runtime/)** | **C/C++ SDKs**. Contains the core libraries for native application development.<br>- `nnsdk`: Standard Neural Network SDK.<br>- `llmsdk`: SDK specialized for Large Language Models. | System Integrators, C++ Developers |
| **[example](example/)** | **Model Zoo & Examples**. A collection of verified model implementations (MobileNet, YOLO, ResNet, etc.) with scripts and pre-trained weights. | All Developers |

## 🚀 Which Tool Should I Use?

- **"I want to test my model quickly from my PC."**  
  👉 Use **[adla_toolkit](adla_toolkit/readme.md)**. You can run Python scripts on your workstation that control the NPU on the board via USB/ADB.

- **"I am building a Python application to run ON the device."**  
  👉 Use **[amlnn_edge_toolkit_lite](amlnn_edge_toolkit_lite/readme.md)**. Install the wheel on your board and import `amlnnlite` to run inference locally.

- **"I need maximum performance and C++ integration."**  
  👉 Refer to **[npu_runtime](npu_runtime/)**. Link against `libnnsdk.so` or `libllmsdk.so` for your native applications.

## 📦 Supported Platforms

The toolkit supports model conversion coverage for:
- **Host OS**: Ubuntu (x86_64) for model conversion tools.
- **Target OS**: Android, Linux (Ubuntu/Debian/Yocto).

## 🔗 Related Resources

- **[Model Playground](https://github.com/Amlogic-NN/amlnn-model-playground)**: Extensive collection of model demos and benchmarks.
- **Docs**: detailed documentation for specific SDKs can be found in their respective subdirectories.

---

*Copyright (c) 2024 Amlogic, Inc. All rights reserved.*
