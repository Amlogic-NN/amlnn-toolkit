/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * @file    compiler_core_api.h
 * @module  aml_compiler_core
 * @brief   Public C API: serialized AML IR -> ADLA compiled artifact.
 * @note    C / dlopen callers build graphs via compiler_core_context.h then
 *          AmlContextCompile(). Serialized AmlCompilerCoreCompile() remains
 *          available. AML_Model stays internal.
 ******************************************************************************/

#ifndef ODML_LITERT_LITERT_VENDORS_AML_CORE_COMPILER_CORE_API_H_
#define ODML_LITERT_LITERT_VENDORS_AML_CORE_COMPILER_CORE_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AML_COMPILER_API
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef AML_COMPILER_CORE_BUILD
#define AML_COMPILER_API __declspec(dllexport)
#else
#define AML_COMPILER_API __declspec(dllimport)
#endif
#else
#define AML_COMPILER_API __attribute__((visibility("default")))
#endif
#endif

/** @brief aml_compiler_core library version (semantic versioning). */
#define AML_COMPILER_CORE_VERSION_MAJOR 1
#define AML_COMPILER_CORE_VERSION_MINOR 0
#define AML_COMPILER_CORE_VERSION_PATCH 0
#define AML_COMPILER_CORE_VERSION_STRING "1.0.0"

/**
 * @brief Runtime version info returned by AmlCompilerCoreGetVersion().
 * @var major  Major version (AML_COMPILER_CORE_VERSION_MAJOR).
 * @var minor  Minor version (AML_COMPILER_CORE_VERSION_MINOR).
 * @var patch  Patch version (AML_COMPILER_CORE_VERSION_PATCH).
 * @var string Embedded version label, e.g. "AML_COMPILER_CORE/1.0.0".
 */
typedef struct AmlCompilerCoreVersion {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
  const char* string;
} AmlCompilerCoreVersion;

/**
 * @brief Status codes returned by aml_compiler_core C API.
 */
typedef enum AmlCompilerStatus {
  kAmlCompilerStatusOk = 0,                   /**< Success. */
  kAmlCompilerStatusErrorInvalidArgument = 1, /**< Null/empty/invalid argument. */
  kAmlCompilerStatusErrorRuntimeFailure = 2,  /**< Convert/compile runtime failure. */
} AmlCompilerStatus;

/**
 * @brief Serialized AML IR buffer produced by SerializeModel(AML_Model).
 * @var data  Pointer to serialized bytes (caller-owned, valid for the call).
 * @var size  Byte length of @p data.
 */
typedef struct AmlSerializedModelBuffer {
  const void* data;
  size_t size;
} AmlSerializedModelBuffer;

/**
 * @brief Compile input: one or more serialized AML IR models + output naming.
 *
 * Typical CNN usage sets @c num_models = 1.
 *
 * @var models       Array of SerializeModel() buffers (caller-owned).
 * @var num_models   Number of entries in @p models.
 * @var model_name   Base name for .adla / fingerprint files.
 * @var model_path   Output directory for .adla (empty string allowed).
 */
typedef struct AmlCompilerCoreRequest {
  const AmlSerializedModelBuffer* models;
  size_t num_models;
  const char* model_name;
  const char* model_path;
} AmlCompilerCoreRequest;

/**
 * @brief Compile options (behavior knobs + reserved extension slot).
 *
 * Pass NULL to AmlCompilerCoreCompile() for defaults, or fill via
 * AmlCompilerCoreOptionsInitDefaults().
 *
 * @var compile_target        ADLA target string; NULL/empty for default/query.
 * @var compile_subgraphs     Max subgraphs to compile; 0 means all partitions.
 * @var embed_adla_in_result  Non-zero to load .adla bytes into the artifact.
 * @var external_options      Optional @ref AmlCompilerExternalOptions payload.
 * @var external_options_size Byte size of @p external_options; 0 if unused.
 */
typedef struct AmlCompilerCoreOptions {
  const char* compile_target;
  size_t compile_subgraphs;
  int embed_adla_in_result;
  const void* external_options;
  size_t external_options_size;
} AmlCompilerCoreOptions;

/**
 * @brief Extension payload for @c AmlCompilerCoreOptions::external_options.
 *
 * Used with @ref AmlContextCompile() (Context API). Pass via
 * @c options.external_options / @c external_options_size.
 *
 * @var log_level ADLA log level override. 0 = use env/property
 *                (@c vendor.AML_DELEGATE_LOG_LEVEL / @c AML_DELEGATE_LOG_LEVEL).
 *                Non-zero values follow @c ADLA_LogLevel (2=VERBOSE .. 7=FATAL).
 * @var dump_ir   When non-zero, @ref AmlContextCompile() calls
 *                @ref AmlContextPrint() on @p ctx after the caller has fully
 *                filled every graph and immediately before materialize.
 *                0 = no automatic dump (default). Manual @ref AmlContextPrint()
 *                remains available regardless of this flag.
 * @var profile_name ADLA compile profile (e.g. @c "performance"). NULL/empty =
 *                     ADLA default profile from GetProfileDefaultName().
 * @var precision_mode @ref AmlPrecisionMode; @c kAmlPrecisionModeUnset leaves
 *                     ADLA profile precision unchanged.
 * @var metadata     Comma-separated key=value user metadata for
 *                     IContext::SetUserMetaData(), e.g.
 *                     @c normalization_mean="[-128 128 -128]",normalization_std="[128 0.78 128]".
 *                     Pointer must remain valid for the duration of compile.
 * @var dump_ir_tflite When non-zero, @ref AmlContextCompile() writes
 *                     @c <model_path>/<model_name>.aml_ir.tflite from the AML
 *                     IR after materialize. 0 = do not write (default). The
 *                     environment variable @c AML_DUMP_IR_TFLITE=1 enables the
 *                     same dump without this field. Older callers that pass a
 *                     struct ending at @c metadata are still accepted.
 */
typedef enum AmlPrecisionMode {
  kAmlPrecisionModeUnset = 0,
  kAmlPrecisionModePrecise = 1,
  kAmlPrecisionModeNormal = 2
} AmlPrecisionMode;

typedef struct AmlCompilerExternalOptions {
  uint32_t log_level;
  uint32_t dump_ir;
  const char* profile_name;
  uint32_t precision_mode;
  const char* metadata;
  uint32_t dump_ir_tflite;
} AmlCompilerExternalOptions;

/**
 * @brief Opaque handle to a compiled ADLA artifact.
 * @note Destroy with AmlCompilerCoreDestroyCompiledArtifact().
 */
typedef struct AmlCompilerCoreCompiledArtifactT* AmlCompilerCoreCompiledArtifact;

/**
 * @brief Query aml_compiler_core library version.
 * @return Static version struct; do not free. Never NULL.
 * @note The @c string field is embedded in the .so and visible via
 *       @c strings libaml_compiler_core.so | grep AML_COMPILER_CORE .
 */
AML_COMPILER_API const AmlCompilerCoreVersion* AmlCompilerCoreGetVersion(void);

/**
 * @brief Fill @p options with default compile settings.
 * @param[out] options Options struct to initialize; must not be NULL.
 */
AML_COMPILER_API void AmlCompilerCoreOptionsInitDefaults(
    AmlCompilerCoreOptions* options);

/**
 * @brief Fill @p external_options with defaults.
 * @param[out] external_options Extension struct; must not be NULL.
 */
AML_COMPILER_API void AmlCompilerExternalOptionsInitDefaults(
    AmlCompilerExternalOptions* external_options);

/**
 * @brief Compile serialized AML IR into an ADLA artifact.
 * @param request Compile input (models + naming). Must not be NULL.
 * @param options Compile options; NULL means defaults.
 * @param[out] compiled_artifact Newly created artifact handle on success.
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlCompilerCoreCompile(
    const AmlCompilerCoreRequest* request,
    const AmlCompilerCoreOptions* options,
    AmlCompilerCoreCompiledArtifact* compiled_artifact);

/**
 * @brief Release a compiled artifact handle and its owned buffers.
 * @param compiled_artifact Handle from AmlCompilerCoreCompile(); may be NULL.
 */
AML_COMPILER_API void AmlCompilerCoreDestroyCompiledArtifact(
    AmlCompilerCoreCompiledArtifact compiled_artifact);

/**
 * @brief Flush compiler_core process-local state after all handles are
 *        destroyed, before dlclose(libaml_compiler_core.so).
 * @note Call AmlContextDestroy / AmlCompilerCoreDestroyCompiledArtifact first.
 *       Then dlclose core, then compiler_adla. libcompiler_adla may use
 *       OpenMP / worker threads and is not safe to unmap; preload it with
 *       RTLD_NODELETE if the process will dlclose.
 */
AML_COMPILER_API void AmlCompilerCoreShutdown(void);

/**
 * @brief Get embedded ADLA binary bytes from a compiled artifact.
 * @param compiled_artifact Artifact handle.
 * @param[out] adla_bin     Pointer to binary (owned by artifact; do not free).
 * @param[out] adla_bin_size Byte size of @p adla_bin.
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlCompilerCoreGetCompiledArtifactAdlaBinary(
    AmlCompilerCoreCompiledArtifact compiled_artifact, const void** adla_bin,
    size_t* adla_bin_size);

/**
 * @brief Get filesystem path of the generated .adla file.
 * @param compiled_artifact Artifact handle.
 * @param[out] adla_path    NUL-terminated path (owned by artifact).
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlCompilerCoreGetCompiledArtifactAdlaPath(
    AmlCompilerCoreCompiledArtifact compiled_artifact, const char** adla_path);

/**
 * @brief Get fingerprint string used for ADLA cache reuse.
 * @param compiled_artifact Artifact handle.
 * @param[out] fingerprint  NUL-terminated fingerprint (owned by artifact).
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlCompilerCoreGetCompiledArtifactFingerprint(
    AmlCompilerCoreCompiledArtifact compiled_artifact,
    const char** fingerprint);

/**
 * @brief Get resolved ADLA compile target string.
 * @param compiled_artifact Artifact handle.
 * @param[out] target       NUL-terminated target (owned by artifact).
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlCompilerCoreGetCompiledArtifactTarget(
    AmlCompilerCoreCompiledArtifact compiled_artifact, const char** target);

/**
 * @brief Query whether this compile reused a cached .adla.
 * @param compiled_artifact Artifact handle.
 * @param[out] cache_hit    Non-zero if cache hit; zero if recompiled.
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlCompilerCoreGetCompiledArtifactCacheHit(
    AmlCompilerCoreCompiledArtifact compiled_artifact, int* cache_hit);

#ifdef __cplusplus
}
#endif

#endif  // ODML_LITERT_LITERT_VENDORS_AML_CORE_COMPILER_CORE_API_H_
