# amlnn_runtime

`amlnn_runtime` 汇总 Amlogic NPU 的 C/C++ 推理运行时 SDK，用于原生应用集成。该目录按模型类型拆分为小模型运行时和大语言模型运行时，包含头文件、预编译动态库、构建脚本和基础示例。

## 目录结构

| 目录 | 说明 |
| :--- | :--- |
| [nn_runtime](nn_runtime/readme_cn.md) | 面向小模型（CNN模型等）的 C/C++ 推理 SDK，链接 `libnnsdk.so`，使用 `nnsdk2.h`。 |
| [llm_runtime](llm_runtime/readme_cn.md) | 面向大语言模型的 C/C++ 推理 SDK，链接 `libllmsdk.so`，使用 `llmsdk.h`。 |

## 使用场景

- Android / Linux 原生应用需要集成 NPU 推理能力。

详细安装、交叉编译和运行说明请进入对应子目录查看。

