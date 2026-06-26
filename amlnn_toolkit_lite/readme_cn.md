# amlnn_toolkit_lite

`amlnn_toolkit_lite` 汇总面向 Debian 板端的轻量级 Python 推理 SDK。该目录只关注板端本地推理，不依赖宿主机 PC 或 ADB 远程推理流程。

## 目录结构

| 目录                                                         | 说明                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [amlnn_edge_toolkit_lite](amlnn_edge_toolkit_lite/readme_cn.md) | 面向小模型（CNN模型等）的轻量级 Python 推理 SDK，提供 `.adla` 模型加载、推理和性能分析能力。 |
| [amlllm_edge_toolkit_lite](amlllm_edge_toolkit_lite/readme_cn.md) | 面向大语言模型的轻量级 Python 推理 SDK，支持流式输出、采样参数和常见模型对话模板。 |

## 使用场景

- 已有通过 `amlnn_toolkit` 转换生成的 `.adla` 模型，需要在 Debian 板端直接运行。
- Python 应用需要快速集成小模型或 LLM 推理能力。

详细安装命令、示例参数和 API 使用方式请进入对应子目录查看。

