# Amlogic AI Python 工具包（amlnn_toolkit）

本工具包提供 Amlogic NPU 完整工作流的 Python 接口：模型转换、量化、编译为 `.adla` 格式，以及推理。支持两种部署模式：

- **x86 PC 模式**：在宿主机 PC 上运行。推理任务通过 ADB 委托给连接的 Amlogic 板端，由板端的 `nnserver` 作为代理执行。
- **Debian 板端模式**：直接安装在 Amlogic Debian 板端，在板端本地完成完整工作流。

![python-runtime-pc](../assets/host-device2.png)

在 PC 模式下，宿主机作为控制端。Python SDK 负责准备输入数据，通过 ADB 将模型和输入文件推送到设备，并远程触发推理。板端的 `nnserver` 加载 `.adla` 模型并在 NPU 上执行推理，推理结果和性能指标通过 ADB 回传到宿主机进行分析。

## 主要特性

- **完整工作流**：模型导入、量化、编译和推理一体化。
- **双模式支持**：支持 x86 PC（通过 ADB + nnserver）或直接在 Debian 板端运行。
- **深度性能分析**：内置逐层延迟、带宽和 NPU 利用率可视化工具。

---

## 支持的示例

| 模型 | 仓库链接 |
| :--- | :--- |
| **MobileNet** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit/example/mobilenet) |
| **ResNet** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/resnet/py) |
| **YOLOv11** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolov11/py) |
| **YOLOv8** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolov8/py) |
| **YOLOWorld** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yoloworld/py) |
| **YOLOX** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/yolox/py) |
| **RetinaFace** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-model-playground/tree/main/examples/retinaface/py) |
| **Qwen** | [在 GitHub 查看](https://github.com/Amlogic-NN/amlnn-toolkit/tree/main/amlnn_toolkit/example/qwen) |

---

## 环境配置

### 前置条件

- **Python**：3.10
- **ADB**：仅 PC 模式需要
- **NPU 驱动**：2.0.2 或更高

> 如果系统安装了多个 Python 版本，建议使用 **Miniforge** 或 **Anaconda** 管理环境。

### 1. 验证 NPU 驱动

检查目标设备的 NPU 驱动版本，要求 **2.0.2 或更高**。

**Android：**

```bash
adb root
adb shell
dmesg | grep adla
# 64 位
strings /vendor/lib64/libadla.so | grep LIBADLA
# 32 位
# strings /vendor/lib/libadla.so | grep LIBADLA
```

**Linux（Buildroot / Yocto / Debian）：**

```bash
adb shell
dmesg | grep adla
strings /usr/lib/libadla.so | grep LIBADLA
```

预期输出：

```
adla kmd version: 2.0.2.x.x
LIBADLA, v2.0.2.x.x.x, 20xx.xx
```

> **重要**：如果驱动版本过旧，需要重新烧录最新镜像。请联系 FAE 或售后团队获取镜像文件。

### 2. 初始化 Python 环境

```bash
# 如需安装 Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 创建并激活环境
conda create -n amlnn_toolkit_py310 python=3.10 -y
conda activate amlnn_toolkit_py310
```

### 3. 安装 Python Wheel 包

**x86 PC（宿主机模式）：**

```bash
pip install amlnn_toolkit/whl/linux_x86/amlnn_toolkit-1.0.0-cp310-cp310-linux_x86_64.whl
```

**Debian 板端（本地模式）：**

```bash
pip install amlnn_toolkit/whl/aarch64/amlnn_edge_toolkit-1.0.0-cp310-cp310-linux_aarch64.whl
```

预期输出：
```
Successfully installed amlnn_toolkit-1.0.0
```

### 4. 部署 nnserver 到目标设备（仅 PC 模式）

将与目标平台匹配的 `nnserver` 二进制文件推送到设备。

**Android：**

```bash
adb root
adb shell "mkdir -p /data/nn"

# 64 位
adb push amlnn_toolkit/nnserver/android/lib64/nnserver /data/nn/nnserver
# 32 位
# adb push amlnn_toolkit/nnserver/android/lib32/nnserver /data/nn/nnserver

adb shell "chmod +x /data/nn/nnserver"
adb shell "cd /data/nn && export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:$LD_LIBRARY_PATH && ./nnserver &"
adb shell "ps -A | grep nnserver"
```

**Linux（Buildroot）：**

```bash
adb shell "mkdir -p /data/nn"

# 64 位
adb push amlnn_toolkit/nnserver/linux/buildroot/lib64/nnserver /data/nn/nnserver
# 32 位
# adb push amlnn_toolkit/nnserver/linux/buildroot/lib32/nnserver /data/nn/nnserver

adb shell "chmod +x /data/nn/nnserver"
adb shell "/data/nn/nnserver &"
adb shell "ps -A | grep nnserver"
```

**Linux（Yocto / Debian）：**

```bash
adb shell "mkdir -p /data/nn"

# 64 位
adb push amlnn_toolkit/nnserver/linux/yocto/lib64/nnserver /data/nn/nnserver
# 32 位
# adb push amlnn_toolkit/nnserver/linux/yocto/lib32/nnserver /data/nn/nnserver

adb shell "chmod +x /data/nn/nnserver"
adb shell "/data/nn/nnserver &"
adb shell "ps -A | grep nnserver"
```

预期输出：
```
root  2558  1  10834704  2708 futex_wait_queue_me 0 S nnserver
```

> 如果 `nnserver` 已在运行，再次启动会打印 `bind() error`。请先终止已有进程再重新启动。

---

## 快速上手

```bash
cd amlnn_toolkit/example/mobilenet

# 本地模式（直接在 Debian 板端运行）
python mobilenet.py --mode native --target-platform 001

# nnserver 模式（从 x86 PC 运行，板端通过 ADB 连接）
python mobilenet.py --mode nnserver --target-platform 001
```

执行成功后，将输出 SDK 版本信息、Top-5 分类结果和 NPU 性能指标。

---

## Demo 简介

`mobilenet.py` 示例覆盖了 MobileNetV2 模型的完整端到端工作流：

1. **自动下载**：如果本地不存在，自动下载 `mobilenet_v2_1.0_224_quant.tflite` 和 `labels.txt`。
2. **加载**：加载量化 TFLite 模型。
3. **配置**：设置量化类型（`w8a8`）和目标平台。
4. **编译**：使用校准数据集（`datasets.txt`）将模型编译为 `.adla` 格式。
5. **导出**：保存编译后的 `.adla` 文件。
6. **初始化运行时**：以指定模式初始化 NPU 运行时。
7. **推理**：对 `fish_224x224.jpeg` 执行推理，打印 Top-5 分类结果。
8. **性能分析**：打印性能指标，并生成 HTML 可视化报告。

### 参数说明

| 参数 | 可选值 | 说明 |
| :--- | :--- | :--- |
| `--mode` | `native`（默认）、`nnserver` | `native`：直接在 Debian 板端运行。`nnserver`：通过 ADB 将推理委托给连接的板端。 |
| `--target-platform` | 见下表 | 编译时使用的目标平台 ID。 |

**平台 ID 对照表：**

| `--target-platform` | 平台 |
| :--- | :--- |
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 |

### 可视化输出

`perf_visualize()` 在当前目录生成以下 HTML 报告：

- `hard_op_chart.html`：硬件算子耗时。
- `soft_op_chart.html`：软件算子耗时。
- `dram_rd/wr_chart.html`：内存带宽分析。
- `pie_charts_distribution.html`：整体时间分布。

<div align="center">
  <img src="../assets/image-20251219144855741.png" width="48%" alt="Hard OP Chart" style="border-radius: 8px; margin-right: 2%;">
  <img src="../assets/image-20251219145742364.png" width="48%" alt="Netron OP ID Mapping" style="border-radius: 8px;">
</div>

---

## 常见问题

- **nnserver bind 错误**：同一时间只能运行一个 `nnserver` 实例。启动新实例前请先终止已有进程。
