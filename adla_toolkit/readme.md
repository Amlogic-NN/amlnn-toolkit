# Amlogic Python Runtime – Quick Start

# Amlogic Python Runtime

This repository provides instructions for setting up the **Amlogic Python Runtime** environment and running demo models on Amlogic platforms.

---

## Supported Environment

- **Host OS**: Ubuntu 20.04  
- **Python Version**: Python 3.10  

> If multiple Python versions are installed on the host system, it is strongly recommended to manage Python environments using **Anaconda** or **Miniforge**.

---

## 1. Environment Setup

### 1.1 Install Anaconda / Miniforge

Miniforge is a lightweight Conda distribution and is recommended.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

---

### 1.2 Create Python Virtual Environment

After installing Anaconda or Miniforge, create and activate a Python 3.10 virtual environment:

```bash
conda create -n nnserver_310 python=3.10
conda activate nnserver_310
```

---

### 1.3 Verify ADB Connection

Connect the host PC to the target device via USB and verify that ADB can detect the device.

```bash
adb devices -l
adb shell
```

The target device should be listed correctly, and you should be able to enter the shell.

---

### 1.4 Verify NPU Driver Version

Check the NPU driver version on the target device.  
The detected driver version determines which Python Runtime release package should be used.

#### Android

```bash
adb root
adb shell
dmesg | grep adla
strings /vendor/lib64/libadla.so | grep LIBADLA
```

> If the system is 32-bit, replace `lib64` with `lib`.

#### Linux / Yocto

```bash
adb shell
dmesg | grep adla
strings /usr/lib/libadla.so | grep LIBADLA
```

---

### 1.5 Install Python Wheel and Dependencies

Install the Python Runtime wheel and its dependencies.

- Python wheel: `aml-nn/python-api/amlnn-xxx.whl`
- Dependency file: `aml-nn/python-api/requirements.txt`

```bash
python -m pip install amlnn-xxx.whl
python -m pip install -r requirements.txt
```

---

### 1.6 Deploy and Configure nnserver

Push the `nnserver` executable to the target device working directory.

Recommended working paths:
- **Android**: `/data/local/tmp`
- **Linux / Yocto**: `/usr/bin`

Available `nnserver` binaries:

```text
nnserver/
├── arm64-v8a               # Android 64-bit
├── armeabi-v7a             # Android 32-bit
├── aarch64-poky-linux      # Yocto 64-bit
├── arm-poky-linux          # Yocto 32-bit
└── arm-none-linux-gnueabihf # Linux 32-bit
```

Select the binary that matches your target platform.

#### Android

```bash
adb root
adb push nnserver /data/local/tmp
```

#### Linux / Yocto

```bash
adb push nnserver /usr/bin
```

---

## Usage Guide

### 2.1 Get ADLA Models

Export the ADLA file in advance using a tool

### 2.2 Start nnserver Service

Open a new terminal and enter the device shell to start `nnserver`.

```bash
adb shell
```

#### Android

```bash
cd /data/local/tmp
export LD_LIBRARY_PATH=/vendor/lib64
chmod +x nnserver
./nnserver
```

> If the system is 32-bit, replace `lib64` with `lib`.

#### Linux / Yocto

```bash
cd /usr/bin
chmod +x nnserver
./nnserver
```

If `nnserver` starts successfully, output similar to the following will be displayed:

```text
NNSERVER, v1.0.0, 2025.08
nnserver start to work, listening on port 8308
nnserver start to work, listening on port 8309
nnserver start to work, listening on port 8310
```

### 2.3 MobileNetV2 Demo

Run the following command in the host PC terminal:

```bash
cd example/mobilenetv2/02_verify_python

python mobilenetv2.py \
    --board-work-path /usr/bin \
    --model-path ../01_export_model/mobilenet_v2_1.0_224_quant.adla \
    --run-cycles 1 \
    --loglevel INFO
```

Upon successful execution, output similar to the following will be displayed, including hardware platform information, input/output basic info, NPU latency, bandwidth consumption, etc.:

```text
......
I Hardware platform: S905X5 (Type: 4)
I Model tensor info - Inputs: 1, Outputs: 1
I Input[0] - Shape: (1, 224, 224, 3), Elements: 150528, Stride: 1, Size: 150528bytes, Format: NHWC, Type: 0, Quantization: scale=0.007812, zp=128
I Output[0] - Shape: (1, 1, 1, 1001), Elements: 1001, Stride: 1, Size: 1001bytes, Format: NHWC, Type: 0, Quantization: scale=0.098893, zp=58
I Average time: 2.396250009536743 ms
I FPS: 417.3187255859375
I Bandwidth: 3.143280029296875 Mbytes
......
```

This demo uses all images in `01_export_model` as test input. To test your own images, place them in `01_export_model` or modify the demo logic.


