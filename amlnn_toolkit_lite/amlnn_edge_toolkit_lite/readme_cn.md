# AMLNNLite：Amlogic 边缘推理工具包（Python）

`amlnn_edge_toolkit_lite` 为 Amlogic NPU 平台提供轻量级 Python 推理 SDK，专为在 Debian 板端直接运行小模型而设计，无需宿主机 PC 或 ADB 连接。

## 主要特性

- **板端本地推理**：完全在 Amlogic Debian 板端运行。
- **开发者友好的 API**：简洁的模型加载、推理和性能分析工作流。
- **深度性能分析**：内置逐层延迟和带宽可视化工具。

---

## 支持的示例

| 模型 | 仓库链接 |
| :--- | :--- |
| **MobileNet** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit_lite/amlnn_edge_toolkit_lite/example/base_api) |
| **ResNet** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/resnet/py) |
| **YOLOv11** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolov11/py) |
| **YOLOv8** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolov8/py) |
| **YOLOWorld** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yoloworld/py) |
| **YOLOX** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolox/py) |
| **RetinaFace** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/retinaface/py) |

---

## 环境配置

### 前置条件

- **操作系统**： Debian
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

conda create -n amlnnlite_py310 python=3.10 -y
conda activate amlnnlite_py310
```

### 3. 安装 Wheel 包

```bash
pip install amlnn_toolkit_lite/amlnn_edge_toolkit_lite/whl/aarch64/amlnn_edge_toolkit_lite-1.0.0-cp310-cp310-linux_aarch64.whl
```

---

## 快速上手

```bash
cd amlnn_toolkit_lite/amlnn_edge_toolkit_lite/example/base_api

python base_api_nnlite.py \
  --model-path /path/to/model.adla \
  --image-path /path/to/image.jpg \
  --labels /path/to/labels.txt
```

执行成功后，将输出 SDK 版本信息、Top-5 分类结果和 NPU 性能指标。

---

## Demo 简介

`base_api_nnlite.py` 示例覆盖了基于预编译 `.adla` 模型的 MobileNetV2 推理流程：

1. **加载**：从指定路径加载 `.adla` 模型。
2. **初始化运行时**：启用性能采集，初始化 NPU 运行时。
3. **推理**：对输入图像进行预处理并执行推理，打印 Top-5 分类结果。
4. **性能分析**：打印性能指标，并生成 HTML 可视化报告。

### 参数说明

| 参数 | 说明 |
| :--- | :--- |
| `--model-path` | 编译后的 `.adla` 模型文件路径，使用 `amlnn_toolkit` 生成。 |
| `--image-path` | 输入图像路径。 |
| `--labels` | 标签文件路径（每行一个标签）。 |

### 可视化输出

`perf_visualize()` 在当前目录生成以下 HTML 报告：

- `hard_op_chart.html`：硬件算子耗时。
- `soft_op_chart.html`：软件算子耗时。
- `dram_rd/wr_chart.html`：内存带宽分析。
- `pie_charts_distribution.html`：整体时间分布。

<div align="center">
  <img src="../../assets/image-20251219144855741.png" width="48%" alt="Hard OP Chart" style="border-radius: 8px; margin-right: 2%;">
  <img src="../../assets/image-20251219145742364.png" width="48%" alt="Netron OP ID Mapping" style="border-radius: 8px;">
</div>

---

## 常见问题

- **模型转换**：在此运行推理前，请先使用 `amlnn_toolkit` 将模型转换并编译为 `.adla` 格式。
