# AMLLLMLite：Amlogic 边缘大语言模型推理工具包（Python）

`amlllm_edge_toolkit_lite` 为 Amlogic NPU 平台提供轻量级大语言模型（LLM）Python 推理 SDK，专为在 Debian 板端直接运行而设计，无需宿主机 PC 或 ADB 连接。

## 主要特性

- **板端本地 LLM 推理**：完全在 Amlogic Debian 板端运行。
- **流式输出**：通过回调函数逐 token 生成输出。
- **多轮对话**：支持跨调用保留历史上下文。
- **灵活采样**：支持 Argmax、Top-P 和 Top-K 采样模式。
- **对话模板支持**：内置 Qwen、Llama、DeepSeek、Gemma 等主流模型模板。

---

## 支持的模型

| 模型 | 仓库链接 |
| :--- | :--- |
| **Qwen / InternVL** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **DeepSeek** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **Gemma / Gemma3** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **Llama** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **MiniCPM4** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |
| **Phi-1.5 / Phi-2** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/LLMs/python) |

---

## 环境配置

### 前置条件

- **操作系统**：Debian
- **Python**：3.10
- **NPU 驱动**：2.0.2 或更高

> 如果系统安装了多个 Python 版本，建议使用 **Miniforge** 或 **Anaconda** 管理环境。

### 1. 验证 NPU 驱动

检查目标设备的 NPU 驱动版本，要求 **2.0.2 或更高**。

```bash
dmesg | grep adla
strings /usr/lib/libadla.so | grep LIBADLA
```

预期输出：

```
adla kmd version: 2.0.2.x.x
LIBADLA, v2.0.2.x.x.x, 20xx.xx
```

> **重要**：如果驱动版本过旧，需要重新烧录最新镜像。请联系 FAE 或售后团队获取镜像文件。

### 2. 初始化 Python 环境（推荐使用 Miniforge）

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

conda create -n amlllmlite_py310 python=3.10 -y
conda activate amlllmlite_py310
```

### 3. 安装 Wheel 包

```bash
pip install amlnn_toolkit_lite/amlllm_edge_toolkit_lite/whl/aarch64/amlllm_edge_toolkit_lite-1.1.0-cp310-cp310-linux_aarch64.whl
```

---

## 快速上手

```bash
cd amlnn_toolkit_lite/amlllm_edge_toolkit_lite/example/base_api

python base_api_llmlite.py \
  --model /path/to/model.adla \
  --model-type qwen
```

执行成功后，将进入交互式对话界面，可直接输入提示词与模型对话。

---

## Demo 简介

`base_api_llmlite.py` 示例覆盖了基于预编译 `.adla` 模型的 LLM 推理流程：

1. **配置**：使用模型路径和采样参数初始化 `AMLLLMLite`。
2. **初始化运行时**：加载模型并初始化 NPU 运行时。
3. **设置对话模板**（可选）：根据 `--model-type` 应用内置模板。
4. **交互式推理**：进入交互循环，接受用户输入并执行流式推理，逐 token 打印输出。

### 参数说明

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--model` | 必填 | 编译后的 `.adla` LLM 模型文件路径 |
| `--model-type` | `none` | 内置对话模板，见下表 |
| `--sampling-mode` | `argmax` | 采样模式：`argmax`、`top_p` 或 `top_k` |
| `--top-k` | `3` | Top-K 采样参数 |
| `--top-p` | `0.9` | Top-P 采样阈值 |
| `--temperature` | `1.0` | Softmax 温度系数 |
| `--repeat-penalty` | `1.1` | 重复 token 惩罚系数 |
| `--log-level` | `ERROR` | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |

### 交互命令

| 命令 | 说明 |
| :--- | :--- |
| `exit` | 退出 demo |
| `new_talk` | 清除对话历史，开始新对话 |
| `break` | 中断当前生成任务 |

### 内置对话模板

以下模型类型具有内置模板。在 CLI demo 中传入 `--model-type` 参数，或手动调用 `set_chat_template()` 使用自定义模板。

| 模型类型 | 是否支持 |
| :--- | :---: |
| `qwen`（Qwen、InternVL） | ✓ |
| `deepseek` | ✓ |
| `gemma` / `gemma3` | ✓ |
| `llama` | ✓ |
| `tiny_llama` / `tiny_llama_v0_4` | ✓ |
| `phi_1_5` / `phi_2` | ✓ |
| `minicpm4` | ✓ |

---

## 常见问题

- **流式输出**：实现 `on_token` 回调以逐 token 打印输出。token 字典中的 `status` 字段表示 `NORMAL`、`FINISH` 或 `ERROR`。
- **多轮对话**：在 `run()` 中设置 `retain_history=True`。调用 `reset_session()` 开始新对话。
- **模型转换**：在此运行推理前，请先使用 `amlnn_toolkit` 将 LLM 转换并编译为 `.adla` 格式。
