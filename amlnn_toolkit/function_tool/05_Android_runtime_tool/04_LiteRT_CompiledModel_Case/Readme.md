CompiledModel API case

Version confirm:
NNSDK Version: v3.0.0, 2026.05
NNSDK2 Version: v1.0.0, 2026.05
Compiler_adla Version: v3.4.4
ADLA Version: 2.0.2.0.0

- Push the existing aml_CompiledModel_benchmark and its required prebuilt libraries to an executable directory on the device (for example /data/local/tmp/ or your designated working directory), and make sure they are executable:
  - First, adb push the files to the target directory
  - Then, on the device, run chmod +x <binary>
- Run command:

```bash
export LD_LIBRARY_PATH=:$(pwd)
./aml_CompiledModel_benchmark mobilenet_v2_1.0_224_quant.tflite
```

- Example output:

```bash
----------------[AML GetInputRequirements] Enter -----------
----------------[AML GetOutputRequirements] Enter -----------
----------------[AML GetOutputRequirements] Enter -----------
----------------[AML GetOutputRequirements] Enter -----------
**************** Create CompiledModel Finished
---
Model signature count: 1
 signature[0] = serving_default
-------------创建输入buffer------
-------------创建输出buffer------
-------------随机填充输入数据------
Running inference...
----------------[AML RegisterTensorBuffer] Enter -----------
----------------[AML RegisterTensorBuffer] Enter -----------
----------------[AML RegisterTensorBuffer] Enter -----------
----------------[AML RegisterTensorBuffer] Enter -----------
LiteRtDispatchInvocationContextT::AttachInputBuffer
----------------[AML GetTensorBuffer] Enter -----------
LiteRtDispatchInvocationContextT::AttachOutputBuffer
----------------[AML GetTensorBuffer] Enter -----------
LiteRtDispatchInvocationContextT::AttachOutputBuffer
----------------[AML GetTensorBuffer] Enter -----------
LiteRtDispatchInvocationContextT::AttachOutputBuffer
----------------[AML GetTensorBuffer] Enter -----------
InvocationContext Execute
-------------推理完成------
 output[0] size=97344
 output[1] size=389376
 output[2] size=24336
----------------[AML GetTensorBuffer] Enter -----------
----------------[AML GetTensorBuffer] Enter -----------
----------------[AML GetTensorBuffer] Enter -----------
----------------[AML GetTensorBuffer] Enter -----------
INFO: [accelerator_registry.cc:41] DestroyAccelerator: ptr=0xb40000772280a6e0, name=CpuAccelerator
INFO: [accelerator_registry.cc:41] DestroyAccelerator: ptr=0xb40000772280a680, name=NpuAccelerator
```
