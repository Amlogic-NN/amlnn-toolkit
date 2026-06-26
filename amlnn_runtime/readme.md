# amlnn_runtime

`amlnn_runtime` collects the C/C++ inference runtime SDKs for Amlogic NPU native application integration. It is split by model type and includes headers, prebuilt shared libraries, build scripts, and basic examples.

## Directory Structure

| Directory | Description |
| :--- | :--- |
| [nn_runtime](nn_runtime/readme.md) | C/C++ inference SDK for small models such as CNN models. Link against `libnnsdk.so` and include `nnsdk2.h`. |
| [llm_runtime](llm_runtime/readme.md) | C/C++ inference SDK for large language models. Link against `libllmsdk.so` and include `llmsdk.h`. |

## Use Cases

- Integrate NPU inference into Android or Linux native applications.

See the corresponding subdirectory for installation, cross-compilation, and runtime instructions.

