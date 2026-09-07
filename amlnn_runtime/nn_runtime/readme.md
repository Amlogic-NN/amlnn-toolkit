# nn_runtime: Amlogic Neural Network C/C++ SDK

`nn_runtime` provides the C/C++ inference SDK for deploying small models on Amlogic NPU platforms. Link against `libnnsdk.so` and include `nnsdk2.h` to integrate inference into native applications. Supports Android and Linux (Buildroot / Yocto / Debian) platforms.

## Key Features

- **Multi-platform support**: Android, Linux Buildroot, Linux Yocto, and Linux Debian — 32-bit and 64-bit libraries included.
- **Simple API**: Model loading, inference, and cleanup through `nnsdk2.h`.

---

## Supported Platforms

| Platform | lib64 | lib32 |
| :--- | :---: | :---: |
| Android | ✓ | ✓ |
| Linux Buildroot | ✓ | ✓ |
| Linux Yocto | ✓ | ✓ |
| Linux Debian | ✓ | ✓ |

---

## Environment Setup

### Prerequisites

- **NPU Driver**: 2.0.2 or higher

### 1. Verify NPU Driver

Check the NPU driver version on the target device. Version **2.0.2 or higher** is required.

**Android:**

```bash
adb root
adb shell
dmesg | grep adla
# 64-bit
strings /vendor/lib64/libadla.so | grep LIBADLA
# 32-bit
# strings /vendor/lib/libadla.so | grep LIBADLA
```

**Linux (Buildroot / Yocto / Debian):**

```bash
adb shell
dmesg | grep adla
strings /usr/lib/libadla.so | grep LIBADLA
```

Expected output:

```
adla kmd version: 2.0.2.x.x
LIBADLA, v2.0.2.x.x.x, 20xx.xx
```

> **Important**: If the driver version is too old, you need to reflash the latest image. Contact your FAE or support team to obtain the image file.

### 2. Install CMake

Download and configure CMake. Recommended version 3.16.3 or higher. Example using cmake-3.24.0-linux-x86_64:

```bash
# Download cmake-3.24.0-linux-x86_64.tar.gz from the CMake official website

# Extract to a local directory
tar -xzf cmake-3.24.0-linux-x86_64.tar.gz -C /opt/

# Add to PATH
echo 'export PATH=/opt/cmake-3.24.0-linux-x86_64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

cmake --version
```

### 3. Install Cross-Compilation Toolchain

Choose the toolchain for your target platform:

**Android**

Download [android-ndk-r25c](https://github.com/android/ndk/wiki/Unsupported-Downloads), extract to a local directory (e.g. `/opt/android-ndk-r25c`), then update `ANDROID_NDK_PATH` on line 4 of `build-android64.sh` or `build-android32.sh`.

**Linux Buildroot**

- 64-bit: Download  [gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu](https://developer.arm.com/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz), extract to a local directory (e.g. `/opt/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu`), then update `BUILDROOT_TOOLCHAIN_PATH` on line 4 of `build-buildroot64.sh`.
- 32-bit: Download [gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf](https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf.tar.xz), extract to a local directory (e.g. `/opt/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf`), then update `BUILDROOT_TOOLCHAIN_PATH` on line 4 of `build-buildroot32.sh`.

**Linux Yocto / Debian**

- 64-bit: Download [poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/64/poky-glibc-x86_64-meta-toolchain-armv8a-mesont7-an400-5.15-a64-toolchain-4.0.20.sh)
- 32-bit: Download [poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh](https://pub-8378326bd0fe4b1d9312a3847f6316a2.r2.dev/toolchain/yocto_toolchain/32/poky-glibc-x86_64-amlogic-bsp-armv7at2hf-neon-mesons7-bh201-5.15-a32-toolchain-4.0.20.sh)

Run the Yocto SDK installer (`.sh`), choose an install path (e.g. `/opt/poky/4.0.20`), then update `YOCTO_TOOLCHAIN_PATH` on line 4 of `build-yocto64.sh` or `build-yocto32.sh`.

### 4. Build

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

The compiled binary is placed in `install/<platform>/base_api_nn`.

---

## Quick Start

After building, push the binary, model, and input to the board and run.

```bash
# Example: Android 64
adb root
adb push install/android64/base_api_nn /data/local/tmp/

adb push /path/to/model.adla /data/local/tmp/
adb push /path/to/input.bin /data/local/tmp/

adb shell

cd /data/local/tmp
./base_api_nn model.adla input.bin
```

On success, the demo prints the SDK version, input/output tensor attributes, and Top-5 classification results.

---

## Demo Overview

`base_api_nn.cpp` covers the full inference workflow on a pre-compiled `.adla` model:

1. **Query version**: Calls `amlnn_query` to print SDK, driver, and delegate versions.
2. **Load model**: Calls `amlnn_init` to load the `.adla` model from a file path.
3. **Query tensor info**: Retrieves input/output count and each tensor's shape, dtype, scale, and zero-point.
4. **Inference**: Reads preprocessed binary input data, then calls `amlnn_inputs_set` → `amlnn_run` → `amlnn_outputs_get`.
5. **Post-process**: Runs Top-5 classification on the output and prints results.
6. **Cleanup**: Calls `amlnn_destroy`.

### Parameters

```
base_api_nn <model_path> <input_path>
```

| Parameter | Description |
| :--- | :--- |
| `model_path` | Path to the compiled `.adla` model file, generated by `amlnn_toolkit` |
| `input_path` | Preprocessed input data (raw binary), matching the model's input tensor shape and dtype |

---

## FAQ

- **How to generate input.bin**: Use the Python example in `amlnn_toolkit` to preprocess an image, then save the numpy array as a binary file with `.tofile()`.
- **Model conversion**: Use `amlnn_toolkit` to convert and compile your model to `.adla` format before running inference here.
