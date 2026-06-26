# llm_runtime：Amlogic 大语言模型 C/C++ SDK

`llm_runtime` 提供用于在 Amlogic NPU 平台部署**大语言模型**的 C/C++ 推理 SDK。链接 `libllmsdk.so` 并包含 `llmsdk.h` 头文件，即可在原生应用中集成 LLM 推理能力。支持 Android 和 Linux（Yocto / Debian）平台。

## 主要特性

- **多平台支持**：Android、Linux Yocto 和 Linux Debian，含 32 位和 64 位库。
- **流式输出**：通过回调逐 token 实时输出。
- **多轮对话**：支持保留对话历史。
- **内置对话模板**：支持 Qwen、Llama、DeepSeek、Gemma 等主流模型。

---

## 支持平台

| 平台 | lib64 | lib32 |
| :--- | :---: | :---: |
| Android | ✓ | ✓ |
| Linux Yocto | ✓ | ✓ |
| Linux Debian | ✓ | ✓ |

---

## 环境配置

### 前置条件

- **NPU 驱动**：2.0.2 或更高

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

**Linux（Yocto / Debian）：**

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

### 2. 安装 CMake

推荐版本 3.16.3 或更高。以 cmake-3.24.0-linux-x86_64 为例：

```bash
# 从 CMake 官网下载 cmake-3.24.0-linux-x86_64.tar.gz

tar -xzf cmake-3.24.0-linux-x86_64.tar.gz -C /opt/

echo 'export PATH=/opt/cmake-3.24.0-linux-x86_64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

cmake --version
```

### 3. 安装交叉编译工具链

根据目标平台选择对应工具链：

**Android**

下载 [android-ndk-r25c](https://github.com/android/ndk/wiki/Unsupported-Downloads)，解压到本地目录（如 `/opt/android-ndk-r25c`），然后修改 `build-android64.sh` 或 `build-android32.sh` 第 4 行的 `ANDROID_NDK_PATH`。

**Linux Yocto / Debian**

- 64 位：下载 [poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/64/poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh)。
- 32 位：下载 [poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/32/poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh)。

运行 Yocto SDK 安装脚本（`.sh`），选择安装路径（如 `/opt/poky/4.0.20`），然后修改 `build-yocto64.sh` 或 `build-yocto32.sh` 第 4 行的 `YOCTO_TOOLCHAIN_PATH`。

### 4. 编译

```bash
cd amlnn_runtime/llm_runtime/example/base_api

# Android 64
./build-android64.sh

# Android 32
./build-android32.sh

# Linux Yocto 64 / Debian 64
./build-yocto64.sh

# Linux Yocto 32 / Debian 32
./build-yocto32.sh
```

编译产物位于 `install/<platform>/base_api_llm`。

---

## 快速上手

编译完成后，将二进制文件、模型推送到板端并运行。

```bash
# 示例：Android 64
adb root
adb push install/android64/base_api_llm /data/local/tmp/

adb push /path/to/model.adla /data/local/tmp/

adb shell

cd /data/local/tmp
./base_api_llm model.adla --model_type qwen
```

运行成功后，将进入交互式对话界面，直接输入问题即可。

---

## Demo 简介

`base_api_llm.cpp` 示例覆盖了 LLM 推理的完整工作流：

1. **初始化**：调用 `aml_llm_init` 加载 `.adla` 模型，注册 token 回调。
2. **设置对话模板**（可选）：根据 `--model_type` 调用 `aml_llm_set_chat_template` 设置内置模板。
3. **交互循环**：等待用户输入，调用 `aml_llm_run` 执行推理，通过回调逐 token 流式输出。
4. **交互命令**：`new_talk` 重置对话历史，`break` 中断当前生成，`exit` 退出程序。
5. **清理**：调用 `aml_llm_uninit` 释放资源。

### 参数说明

```
base_api_llm <model_path> [--model_type <type>]
```

| 参数 | 说明 |
| :--- | :--- |
| `model_path` | 编译后的 `.adla` LLM 模型路径，由 `amlnn_toolkit` 生成 |
| `--model_type` | 内置对话模板类型，见下表。省略则使用模型内置默认模板 |

### 支持的模型类型

| `--model_type` | 适用模型 |
| :--- | :--- |
| `qwen` | Qwen、InternVL |
| `deepseek` | DeepSeek |
| `gemma` | Gemma |
| `gemma3` | Gemma3 |
| `llama` | Llama |
| `tiny_llama` | TinyLlama |
| `tiny_llama_v0_4` | TinyLlama v0.4 |
| `phi_1_5` | Phi-1.5 |
| `phi_2` | Phi-2 |
| `minicpm4` | MiniCPM4 |

---

## 常见问题

- **模型转换**：在此运行推理前，请先使用 `amlnn_toolkit` 将 LLM 转换并编译为 `.adla` 格式。
