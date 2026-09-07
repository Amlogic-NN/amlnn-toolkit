/*
* Copyright (C) 2026 Amlogic, Inc. All rights reserved.
*
* This source code is subject to the terms and conditions defined in the
* file 'LICENSE' which is part of this source code package.
*
* Description: aml_llmsdk
*/

#ifndef _LLMSDK_H_
#define _LLMSDK_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLMSDK_VERSION "LLMSDK, v1.2.0, 2026.08" // v<Major>.<Minor>.<Patch>

/**
 * @typedef LLMContext
 * @brief A handle used for managing and interacting with the large language model runtime.
 */
typedef void* LLMContext;

/**
 * @enum AML_LLMRunStatus
 * @brief Represents the possible states during an LLM generation process.
 */
typedef enum {
    AML_LLM_RUN_NORMAL = 0, /**< The LLM is currently processing as expected. */
    AML_LLM_RUN_FINISH = 1, /**< The LLM has completed the task successfully. */
    AML_LLM_RUN_ERROR  = 2, /**< The LLM encountered an error during execution. */
} AML_LLMRunStatus;

/**
 * @enum AML_LLMRetStatus
 * @brief Indicates the return status of an LLM SDK API call.
 */
typedef enum {
    AML_LLM_Status_Success = 0, /**< The API call completed successfully. */
    AML_LLM_Status_Failed  = 1, /**< The API call failed. This is a generic error return value. */
} AML_LLMRetStatus;


#ifndef LLMSDK_ENABLE_EXTSION
/**
 * @enum AML_LLMSamplingMode
 * @brief Defines available sampling strategies for output generation.
 */
typedef enum {
    AML_LLM_ARG_Max,       /**< Argmax sampling. Always selects the token with the highest probability. Deterministic and reproducible. */
    AML_LLM_TOP_P,         /**< Top-P (nucleus) sampling. Samples from the smallest set of tokens whose cumulative probability exceeds threshold p. */
    AML_LLM_TOP_K,         /**< Top-K sampling. Samples from the top k tokens with highest probabilities. Balances diversity and control. */
    AML_LLM_CHAIN_SAMPLER, /**< Configurable sampler chain controlled by sampler_params_json. */
} AML_LLMSamplingMode;

/**
 * @struct AML_LLMInitExtend
 * @brief Additional optional parameters for LLM init (currently reserved for future use).
 */
#pragma pack(push, 1)
typedef struct {
    const char* mmproj_path;                          /**< Optional multimodal projector path. Set NULL for text-only LLM. */
    const char* sampler_params_json;                  /**< JSON string configuring AML_LLM_CHAIN_SAMPLER. Pass NULL or "" for defaults. */
    uint8_t reserved[1024 - sizeof(const char*) * 2]; /**< Reserved for future extension of init configuration. */
} AML_LLMInitExtend;
#pragma pack(pop)

/**
 * @struct AML_LLMRunExtend
 * @brief Additional optional parameters for one LLM inference.
 */
#pragma pack(push, 1)
typedef struct {
    uint32_t sampling_config_valid;               /**< Set to 1 to use all sampling fields below, including sampler_params_json, for this run; set to 0 to use init-time sampling defaults. */
    AML_LLMSamplingMode sampling_mode;            /**< Runtime sampling strategy for this run. */
    int32_t top_k;                                /**< Runtime Top-K sampling parameter for this run. */
    float top_p;                                  /**< Runtime Top-P sampling parameter for this run. */
    float temperature;                            /**< Runtime temperature for this run. */
    float repeat_penalty;                         /**< Runtime repeat penalty for this run. */
    const char* sampler_params_json;              /**< Per-run chain-sampler JSON applied over init defaults when sampling_config_valid is 1. NULL or "" uses init defaults. */
    uint8_t reserved[1024 - sizeof(uint32_t)
                          - sizeof(AML_LLMSamplingMode)
                          - sizeof(int32_t)
                          - sizeof(float) * 3
                          - sizeof(const char*)]; /**< Reserved for future extension of run configuration. */
} AML_LLMRunExtend;
#pragma pack(pop)

/**
 * @enum AML_LLMImageType
 * @brief Specifies the image representation accepted by VLM input.
 */
typedef enum {
    AML_LLM_IMAGE_TYPE_BUFFER = 0, /**< Tightly packed RGB888 buffer. */
} AML_LLMImageType;

/**
 * @enum AML_LLMInputType
 * @brief Specifies the format of input data accepted by the LLM.
 */
typedef enum {
    AML_LLM_INPUT_PROMPT = 0,     /**< Input is a plain text prompt string. The SDK applies chat template and tokenizer before inference. */
    AML_LLM_INPUT_MULTIMODAL = 1, /**< Input contains text plus RGB888 image buffers. The SDK handles preprocessing and multimodal projector inference internally. */
    AML_LLM_INPUT_TOKEN = 2,      /**< Input is final token ids. The SDK skips chat template and tokenizer completely. */
} AML_LLMInputType;

#endif


/**
 * @struct AML_LLMInitConfig
 * @brief Contains configuration parameters for initializing the LLM instance.
 */
typedef struct {
    const char* model_path;            /**< Path to the LLM model file. */
    AML_LLMSamplingMode sampling_mode; /**< Sampling strategy for text generation. */
    int32_t top_k;                     /**< Top-K sampling parameter. */
    float top_p;                       /**< Top-P sampling parameter. */
    float temperature;                 /**< Controls randomness in token selection. */
    float repeat_penalty;              /**< Penalty for repeated tokens. */
    AML_LLMInitExtend init_extend;     /**< Optional initialization extension parameters. */
} AML_LLMInitConfig;

/**
 * @struct AML_LLMImageInput
 * @brief One original RGB888 image supplied to a VLM request.
 *
 * The image buffer must contain width * height * 3 bytes in RGB order, with
 * no row padding. The caller keeps the buffer valid until aml_llm_run()
 * returns. Image preprocessing and multimodal projector inference are performed by
 * the SDK.
 */
typedef struct {
    AML_LLMImageType type; /**< Image representation. Currently only AML_LLM_IMAGE_TYPE_BUFFER is supported. */
    const void* data;      /**< Pointer to a tightly packed RGB888 buffer. */
    uint32_t width;        /**< Original image width in pixels. */
    uint32_t height;       /**< Original image height in pixels. */
    uint64_t timestamp;    /**< Optional frame timestamp in milliseconds. */
} AML_LLMImageInput;

/**
 * @struct AML_LLMMultimodalInput
 * @brief Text prompt and original RGB images for one VLM inference.
 */
typedef struct {
    const char* prompt;                  /**< Prompt text. Use img_content, default <image>, to mark where the image content should appear in the prompt. */
    const AML_LLMImageInput* img_inputs; /**< Array of original RGB888 images. Images correspond to markers in prompt order; one or more images are supported. */
    uint32_t img_count;                  /**< Number of entries in img_inputs. */
    const char* img_start;               /**< Optional text inserted before the image content, e.g. <|vision_start|>. Set NULL when not needed. */
    const char* img_end;                 /**< Optional text inserted after the image content, e.g. <|vision_end|>. Set NULL when not needed. */
    const char* img_content;             /**< Placeholder text in prompt used to locate the image content. Default is <image> when NULL. */
    uint32_t sample_rate_num;            /**< Video sampling-rate numerator. (TODO) */
    uint32_t sample_rate_den;            /**< Video sampling-rate denominator. (TODO) */
} AML_LLMMultimodalInput;

/**
 * @struct AML_LLMTokenInput
 * @brief Final token ids for one LLM inference.
 *
 * The caller is responsible for applying any chat template, system prompt, role
 * formatting, assistant generation prompt, and tokenizer before filling this
 * structure. aml_llm_run() sends these ids directly to the model.
 */
typedef struct {
    const int32_t* input_ids; /**< Final token id array passed directly to the SDK inference path. */
    int32_t n_tokens;         /**< Number of token ids in input_ids. Must be greater than 0. */
} AML_LLMTokenInput;

/**
 * @struct AML_LLMInput
 * @brief Represents input data passed to the LLM, supporting multiple input types.
 */
typedef struct {
    AML_LLMInputType input_type;                 /**< Specifies the type of input. */
    union {
        const char* prompt_input;                /**< Used when input_type is AML_LLM_INPUT_PROMPT. Contains plain text. */
        AML_LLMMultimodalInput multimodal_input; /**< Used when input_type is AML_LLM_INPUT_MULTIMODAL. */
        AML_LLMTokenInput token_input;           /**< Used when input_type is AML_LLM_INPUT_TOKEN. Contains final token ids. */
    };
    const char* role;                            /**< Message role. Default is "user" when NULL. */
} AML_LLMInput;

/**
 * @enum AML_LLMRunMode
 * @brief Describes the available modes for performing LLM inference.
 */
typedef enum {
    AML_LLM_RUN_GENERATE = 0,  /**< Produces generated text from the input. */
    AML_LLM_RUN_EMBEDDING = 1, /**< Produces an embedding vector from the input. (TODO) */
} AML_LLMRunMode;

/**
 * @struct AML_LLMRunConfig
 * @brief Configuration structure to control the behavior of a single inference run.
 */
typedef struct {
    AML_LLMRunMode run_mode;     /**< Inference mode (e.g., text generation). */
    int retain_history;          /**< Whether to retain conversation history: 1 to retain, 0 to reset after each run. */
    int enable_think;            /**< Whether to enable thinking mode: 1 to enable, 0 to disable. */
    AML_LLMRunExtend run_extend; /**< Optional runtime extension parameters for this inference. */
} AML_LLMRunConfig;

/**
 * @struct AML_LLMGenerationResult
 * @brief Represents one incremental result produced during text generation.
 */
typedef struct {
    const char* text; /**< Current generated text fragment. */
    int32_t token_id; /**< Current generated token ID. */
} AML_LLMGenerationResult;

/**
 * @struct AML_LLMEmbeddingResult
 * @brief Represents a semantic vector produced by an embedding model.
 */
typedef struct {
    const float* values; /**< SDK-owned embedding values, valid only during the callback. */
    uint32_t dimension;  /**< Number of float elements in values. */
} AML_LLMEmbeddingResult;

/**
 * @struct AML_LLMResult
 * @brief Represents a result delivered through LLMResultCallback.
 *
 * Select the corresponding union member according to AML_LLMRunConfig::run_mode.
 * Pointers in the result are owned by the SDK and are valid only during the
 * callback. Copy the referenced data if it must be retained afterward.
 */
typedef struct {
    union {
        AML_LLMGenerationResult generation; /**< Valid for AML_LLM_RUN_GENERATE. */
        AML_LLMEmbeddingResult embedding;   /**< Valid for AML_LLM_RUN_EMBEDDING. (TODO) */
    };
} AML_LLMResult;


/**
 * @typedef LLMResultCallback
 * @brief Function signature for handling inference results.
 *
 * @param result Pointer to the result data generated by the LLM.
 * @param userdata Pointer to user-defined data passed to the callback.
 * @param state Current status of the inference process.
 */
typedef void(*LLMResultCallback)(AML_LLMResult* result, void* userdata, AML_LLMRunStatus state);

/**
 * @brief Initialize an LLM instance with the given configuration and callback handler.
 *
 * @param context Pointer to the LLM runtime context.
 * @param init_config Pointer to the configuration settings for initialization.
 * @param callback Function to handle results generated by the LLM.
 * @return Status code (0 for success, non-zero indicates failure).
 */
AML_LLMRetStatus aml_llm_init(LLMContext* context, AML_LLMInitConfig* init_config, LLMResultCallback callback);

/**
 * @brief Uninitialize the LLM instance and release associated resources.
 *
 * @param context The LLM context.
 * @return Status code (0 for success, non-zero indicates failure).
 */
AML_LLMRetStatus aml_llm_uninit(LLMContext context);

/**
 * @brief Execute an inference task.
 *
 * For streaming output, results are delivered incrementally via the registered callback.
 *
 * @param context The LLM context.
 * @param input Pointer to input data for the inference.
 * @param run_config Pointer to inference execution parameters.
 * @param userdata User-defined data that will be passed to the callback.
 * @return Status code (0 for success, non-zero indicates failure).
 */
AML_LLMRetStatus aml_llm_run(LLMContext context, AML_LLMInput* input, AML_LLMRunConfig* run_config, void* userdata);

/**
 * @brief Reset the state of the LLM context, clearing any existing history or state.
 *
 * @param context The LLM context.
 * @return Status code (0 for success, non-zero indicates failure).
 */
AML_LLMRetStatus aml_llm_reset(LLMContext context);

/**
 * @brief Interrupt an active inference process.
 *
 * This function attempts to stop any ongoing generation immediately.
 *
 * @param context The LLM context.
 * @return Status code (0 for success, non-zero indicates failure).
 */
AML_LLMRetStatus aml_llm_break(LLMContext context);

/**
 * @brief Configure the chat formatting template for the LLM, including system prompt, prefix, and postfix.
 *
 * This function lets users define how prompts are constructed during chat interactions.
 * The system prompt typically sets the initial context or role of the assistant.
 * The prefix is prepended to each user input, and the postfix is appended after the input,
 * helping structure the prompt for consistent model behavior.
 *
 * @param context The LLM context.
 * @param system_prompt A string that defines the initial context or role of the LLM.
 * @param prompt_prefix A string inserted before each user input.
 * @param prompt_postfix A string appended after each user input.
 *
 * @return Status code (0 on success; non-zero indicates an error).
 */
AML_LLMRetStatus aml_llm_set_chat_template(LLMContext context, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _LLMSDK_H_
