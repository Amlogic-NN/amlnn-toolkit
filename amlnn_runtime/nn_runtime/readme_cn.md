# nn_runtime：Amlogic 神经网络 C/C++ SDK

`nn_runtime` 为 Amlogic NPU 平台提供 C/C++ 推理 SDK，用于部署小模型。通过链接 `libnnsdk.so` 并包含 `nnsdk2.h` 头文件即可集成到原生应用中，支持 Android 和 Linux（Buildroot / Yocto / Debian ）平台。

## 主要特性

- **多平台支持**：Android、Linux Buildroot、Linux Yocto、Linux Debian，提供 32 位和 64 位库。
- **简洁 API**：通过 `nnsdk2.h` 完成模型加载、推理和资源释放。

---

## 支持平台

| 平台 | lib64 | lib32 |
| :--- | :---: | :---: |
| Android | ✓ | ✓ |
| Linux Buildroot | ✓ | ✓ |
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

### 2. 安装 CMake

下载并配置 CMake，推荐版本 3.16.3 以上，以 cmake-3.24.0-linux-x86_64 为例：

```bash
# 进入 CMake 官网，找到 cmake-3.24.0-linux-x86_64.tar.gz 并下载到本地

# 解压到指定目录
tar -xzf cmake-3.24.0-linux-x86_64.tar.gz -C /opt/

# 添加到 PATH
echo 'export PATH=/opt/cmake-3.24.0-linux-x86_64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

cmake --version
```

### 3. 安装交叉编译工具链

根据目标平台选择对应工具链：

**Android**

下载 [android-ndk-r25c](https://github.com/android/ndk/wiki/Unsupported-Downloads)，解压到指定目录（如 `/opt/android-ndk-r25c`），修改 `build-android64.sh` 或 `build-android32.sh` 第 4 行的 `ANDROID_NDK_PATH`。

**Linux Buildroot**

- 64 位：下载 [gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu](https://developer.arm.com/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz)，解压到指定目录（如 `/opt/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu`），修改 `build-buildroot64.sh` 第 4 行的 `BUILDROOT_TOOLCHAIN_PATH`。
- 32 位：下载 [gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf](https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz)，解压到指定目录（如 `/opt/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf`），修改 `build-buildroot32.sh` 第 4 行的 `BUILDROOT_TOOLCHAIN_PATH`。

**Linux Yocto / Debian**

- 64位：下载 [poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/64/poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh)
- 32位：下载 [poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/32/poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh)

执行 Yocto SDK 安装包（`.sh`），选择安装路径（如 `/opt/poky/4.0.20`），修改 `build-yocto64.sh` 或 `build-yocto32.sh` 第 4 行的 `YOCTO_TOOLCHAIN_PATH`。

### 4. 编译

```bash
cd amlnn_runtime/nn_runtime/example/base_api

# Android 64
./build-android64.sh

# Android 32
./build-android32.sh

# Linux Buildroot 64
./build-buildroot64.sh

# Linux Buildroot 32
./build-buildroot32.sh

# Linux Yocto 64 / Debian 64
./build-yocto64.sh

# Linux Yocto 32 / Debian 32
./build-yocto32.sh
```

编译产物输出到 `install/<platform>/base_api_nn`。

---

## 快速上手

编译完成后，将二进制文件、模型及输入推送到板端运行。

```bash
# 以 Android 64 为例
adb root
adb push install/android64/base_api_nn /data/local/tmp/

adb push /path/model.adla /data/local/tmp/
adb push /path/input.bin /data/local/tmp/

adb shell

cd /data/local/tmp
./base_api_nn model.adla input.bin
```

执行成功后，将打印 SDK 版本信息、输入输出张量属性和 Top-5 分类结果。

---

## Demo 简介

`base_api_nn.cpp` 示例覆盖了基于预编译 `.adla` 模型的完整推理流程：

1. **查询版本**：调用 `amlnn_query` 打印 SDK、驱动和 delegate 版本。
2. **加载模型**：调用 `amlnn_init` 从文件路径加载 `.adla` 模型。
3. **查询张量信息**：获取输入输出数量及各张量的形状、数据类型、scale/zero-point。
4. **推理**：从文件读取预处理好的 bin 数据，调用 `amlnn_inputs_set` → `amlnn_run` → `amlnn_outputs_get`。
5. **后处理**：对输出结果执行 Top-5 分类并打印。
6. **释放资源**：调用 `amlnn_destroy`。

### 参数说明

```
base_api_nn <model_path> <input_path>
```

| 参数 | 说明 |
| :--- | :--- |
| `model_path` | 编译后的 `.adla` 模型文件路径，使用 `amlnn_toolkit` 生成 |
| `input_path` | 预处理后的输入数据（raw binary），与模型输入张量的形状和数据类型一致 |

---

## 常见问题

- **如何生成 input.bin**：使用 `amlnn_toolkit` 中的 Python 示例对图像进行预处理，将 numpy 数组用 `.tofile()` 保存为 bin 文件。
- **模型转换**：在此运行推理前，请先使用 `amlnn_toolkit` 将模型转换并编译为 `.adla` 格式。
