/*
* Copyright (C) 2026 Amlogic, Inc. All rights reserved.
*
* This source code is subject to the terms and conditions defined in the
* file 'LICENSE' which is part of this source code package.
*
* Description: aml_nnsdk
*/

#ifndef _NN_SDK_H
#define _NN_SDK_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * =============================================================
 *                        Macro Definitions
 * =============================================================
 */
#define ADDRESS_MAX_NUM             64
#define MAX_NAME_LENGTH             64
#define INPUT_CHANNEL               3
#define MAX_TENSOR_NUM_DIMS         6
#define RESERVED_BYTE               256

/**
 * =============================================================
 *                        Type Definitions
 * =============================================================
 */
/**
 * @enum NNSDK_Status
 * @brief Status codes returned by NNSDK APIs
 */
typedef enum _NNSDK_Status
{
    NNsdkStatus_Success         = 0,   /**< Operation completed successfully */
    NNsdkStatus_Failed          = 1,   /**< General failure */
    NNsdkStatus_BadParameter    = 2,   /**< Invalid parameter */
    NNsdkStatus_BackendError    = 3,   /**< Backend execution error */
    NNsdkStatus_HalError        = 4,   /**< Hardware abstraction layer error */
    NNsdkStatus_MallocError     = 5,   /**< Malloc space error */
    NNsdkStatus_Ctx_Invalid     = 6,   /**< Context error */
    NNsdkStatus_MAX             = 10   /**< Maximum enum value */
} NNSDK_Status;

/**
 * @enum aml_hw_flag_t
 * @brief Hardware backend selection
 */
typedef enum _aml_hw_flag_t {
    AML_HW_NPU                  = 1,   /**< NPU */
    AML_HW_GPU                  = 2,   /**< GPU */
    AML_HW_CPU                  = 3,   /**< CPU */
    AML_HW_FLAG_MAX             = 10   /**< Maximum enum value */
} aml_hw_flag_t;

/**
 * @enum amlnn_model_type
 * @brief Neural network model type
 *
 * @note Only ADLA_LOADABLE is actively supported.
 *       Other types are deprecated and should not be used.
 */
typedef enum _amlnn_model_type {
    ADLA_LOADABLE               = 0,   /**< ADLA loadable model */
    CAFFE                       = 1,   /**< Deprecated */
    TENSORFLOW                  = 2,   /**< Deprecated */
    TENSORFLOWLITE              = 3,   /**< Deprecated */
    DARKNET                     = 4,   /**< Deprecated */
    ONNX                        = 5,   /**< Deprecated */
    KERAS                       = 6,   /**< Deprecated */
    PYTORCH                     = 7,   /**< Deprecated */
    AMLNN_MEDEL_TYPE_MAX        = 10   /**< Maximum enum value */
} amlnn_model_type;

/**
 * @enum amlnn_nbg_type
 * @brief Model input source type
 */
typedef enum _amlnn_nbg_type {
    NN_ADLA_FILE                = 0,   /**< Model input source is an ADLA model file */
    NN_ADLA_MEMORY              = 1,   /**< Model input source is a memory buffer containing ADLA model */
    NN_RUNTIME_FILE             = 2,   /**< Deprecated */
    NN_RUNTIME_MEMORY           = 3,   /**< Deprecated */
    NN_NBG_FILE                 = 4,   /**< Deprecated */
    NN_NBG_MEMORY               = 5,   /**< Deprecated */
    AMLNN_NBG_TYPE_MAX          = 10   /**< Maximum enum value */
} amlnn_nbg_type;

/**
 * @enum aml_io_format_t
 *
 * @deprecated This enum is deprecated and should not be used.
 */
typedef enum _aml_io_format_t {
    AML_IO_VIRTUAL              = 0,
    AML_IO_PHYS                 = 1,
    AML_IO_VIRTUAL_SECURE       = 2,
    AML_IO_PHYS_SECURE          = 3,
    AML_IO_FORMAT_MAX           = 10
} aml_io_format_t;

/**
 * @struct assign_user_address_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef struct _assign_address {
    unsigned int    inAddr_size;
    unsigned int    outAddr_size;
    aml_io_format_t io_type;
    unsigned char*  inAddr[ADDRESS_MAX_NUM];
    unsigned char*  outAddr[ADDRESS_MAX_NUM];
} assign_user_address_t;

/**
 * @enum aml_operator_t
 * @brief Software operator type for optimization control
 *
 */
typedef enum _aml_operator_t {
    AML_Add = 0,
    AML_AveragePool2d = 1,
    AML_Concatenation = 2,
    AML_Conv2d = 3,
    AML_DepthwiseConv2d = 4,
    AML_DepthToSpace = 5,
    AML_Dequantize = 6,
    AML_EmbeddingLookup = 7,
    AML_Floor = 8,
    AML_FullyConnected = 9,
    AML_HashtableLookup = 10,
    AML_L2Normalization = 11,
    AML_L2Pool2d = 12,
    AML_LocalResponseNormalization = 13,
    AML_Logistic = 14,
    AML_LshProjection = 15,
    AML_Lstm = 16,
    AML_MaxPool2d = 17,
    AML_Mul = 18,
    AML_Relu = 19,
    AML_ReluN1To1 = 20,
    AML_Relu6 = 21,
    AML_Reshape = 22,
    AML_ResizeBilinear = 23,
    AML_Rnn = 24,
    AML_Softmax = 25,
    AML_SpaceToDepth = 26,
    AML_Svdf = 27,
    AML_Tanh = 28,
    AML_ConcatEmbeddings = 29,
    AML_SkipGram = 30,
    AML_Call = 31,
    AML_Custom = 32,
    AML_EmbeddingLookupSparse = 33,
    AML_Pad = 34,
    AML_UnidirectionalSequenceRnn = 35,
    AML_Gather = 36,
    AML_BatchToSpaceNd = 37,
    AML_SpaceToBatchNd = 38,
    AML_Transpose = 39,
    AML_Mean = 40,
    AML_Sub = 41,
    AML_Div = 42,
    AML_Squeeze = 43,
    AML_UnidirectionalSequenceLstm = 44,
    AML_StridedSlice = 45,
    AML_BidirectionalSequenceRnn = 46,
    AML_Exp = 47,
    AML_TopkV2 = 48,
    AML_Split = 49,
    AML_LogSoftmax = 50,
    AML_Delegate = 51,
    AML_BidirectionalSequenceLstm = 52,
    AML_Cast = 53,
    AML_Prelu = 54,
    AML_Maximum = 55,
    AML_ArgMax = 56,
    AML_Minimum = 57,
    AML_Less = 58,
    AML_Neg = 59,
    AML_PadV2 = 60,
    AML_Greater = 61,
    AML_GreaterEqual = 62,
    AML_LessEqual = 63,
    AML_Select = 64,
    AML_Slice = 65,
    AML_Sin = 66,
    AML_TransposeConv = 67,
    AML_SparseToDense = 68,
    AML_Tile = 69,
    AML_ExpandDims = 70,
    AML_Equal = 71,
    AML_NotEqual = 72,
    AML_Log = 73,
    AML_Sum = 74,
    AML_Sqrt = 75,
    AML_Rsqrt = 76,
    AML_Shape = 77,
    AML_Pow = 78,
    AML_ArgMin = 79,
    AML_FakeQuant = 80,
    AML_ReduceProd = 81,
    AML_ReduceMax = 82,
    AML_Pack = 83,
    AML_LogicalOr = 84,
    AML_OneHot = 85,
    AML_LogicalAnd = 86,
    AML_LogicalNot = 87,
    AML_Unpack = 88,
    AML_ReduceMin = 89,
    AML_FloorDiv = 90,
    AML_ReduceAny = 91,
    AML_Square = 92,
    AML_ZerosLike = 93,
    AML_Fill = 94,
    AML_FloorMod = 95,
    AML_Range = 96,
    AML_ResizeNearestNeighbor = 97,
    AML_LeakyRelu = 98,
    AML_SquaredDifference = 99,
    AML_MirrorPad = 100,
    AML_Abs = 101,
    AML_SplitV = 102,
    AML_Unique = 103,
    AML_Ceil = 104,
    AML_ReverseV2 = 105,
    AML_AddN = 106,
    AML_GatherNd = 107,
    AML_Cos = 108,
    AML_Where = 109,
    AML_Rank = 110,
    AML_Elu = 111,
    AML_ReverseSequence = 112,
    AML_MatrixDiag = 113,
    AML_Quantize = 114,
    AML_MatrixSetDiag = 115,
    AML_Round = 116,
    AML_HardSwish = 117,
    AML_If = 118,
    AML_While = 119,
    AML_NonMaxSuppressionV4 = 120,
    AML_NonMaxSuppressionV5 = 121,
    AML_ScatterNd = 122,
    AML_SelectV2 = 123,
    AML_Densify = 124,
    AML_SegmentSum = 125,
    AML_BatchMatmul = 126,
    AML_PlaceholderForGreaterOpCodes = 127,
    AML_Cumsum = 128,
    AML_CallOnce = 129,
    AML_BroadcastTo = 130,
    AML_Rfft2d = 131,
    AML_Conv3d = 132,
    AML_Imag = 133,
    AML_Real = 134,
    AML_ComplexAbs = 135,
    AML_Hashtable = 136,
    AML_HashtableFind = 137,
    AML_HashtableImport = 138,
    AML_HashtableSize = 139,
    AML_ReduceAll = 140,
    AML_Conv3dTranspose = 141,
    AML_VarHandle = 142,
    AML_ReadVariable = 143,
    AML_AssignVariable = 144,
    AML_BroadcastArgs = 145,
    AML_RandomStandardNormal = 146,
    AML_Bucketize = 147,
    AML_RandomUniform = 148,
    AML_Multinomial = 149,
    AML_Gelu = 150,
    AML_DynamicUpdateSlice = 151,
    AML_Relu0To1 = 152,
    AML_UnsortedSegmentProd = 153,
    AML_UnsortedSegmentMax = 154,
    AML_UnsortedSegmentSum = 155,
    AML_Atan2 = 156,
    AML_UnsortedSegmentMin = 157,
    AML_Sign = 158,
    AML_Bitcast = 159,
    AML_BitwiseXor = 160,
    AML_RightShift = 161,
    AML_DetectionPostProcess = 256,
    AML_Erf = 260,
    AML_Hardware = 511,
    AML_Unknown = 2147483647,
    AML_MIN = AML_Add,
    AML_OPERATOR_MAX = AML_Unknown
} aml_operator_t;

/**
 * @struct aml_openmp_opt_t
 * @brief OpenMP acceleration options for software operators
 */
typedef struct _aml_openmp_opt_t {
    aml_operator_t operator_type;     /**< Operator type */
    bool           enable_openmp;     /**< Enable OpenMP acceleration */
    bool           involve_all_ops;   /**< Enable OpenMP for all operators */
    int8_t         openmp_num;        /**< Number of CPU core */
} aml_openmp_opt_t;

/**
 * @struct aml_neon_opt_t
 * @brief NEON acceleration options for software operators
 */
typedef struct _aml_neon_opt_t {
    aml_operator_t operator_type;     /**< Operator type */
    bool           enable_neon;       /**< Enable NEON acceleration */
    bool           involve_all_ops;   /**< Enable NEON for all operators */
} aml_neon_opt_t;

/**
 * @struct softOpInfo_t
 * @brief Software operator optimization control information
 */
typedef struct _softOpInfo_t {
    bool              set_openmp_opt_flag;   /**< Enable OpenMP option configuration */
    int               openmp_opt_num;        /**< Number of OpenMP option entries */
    aml_openmp_opt_t* openmp_opt;            /**< Pointer to OpenMP option array */

    bool              set_neon_opt_flag;     /**< Enable NEON option configuration */
    int               neon_opt_num;          /**< Number of NEON option entries */
    aml_neon_opt_t*   neon_opt;              /**< Pointer to NEON option array */
} softOpInfo_t;

/**
 * @enum aml_kvcache_type_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef enum _aml_kvcache_type_t {
    KVCompute_Prune             = 1,
    KVTransformer_Accel         = 2,
    AML_KVCACHE_TYPE_MAX        = 10
} aml_kvcache_type_t;

/**
 * @struct aml_kvcache_opt_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef struct _aml_kvcache_opt_t {
    int32_t operator_index;
    bool    enable_kvcache;           /**< enable skipping invalid vector computations outside the range of ADLA_KVCACHE_DYNAMIC_VAL.current_mask. */
    bool    zero_out_invalid_value;   /**< set output tensors partial values to zero outside the range of ADLA_KVCACHE_DYNAMIC_VAL.current_mask, */
                                      /**< When the software operator(enable skip) is followed by a operator(disable skip), it must be set to true to ensure that the result is correct. */
    int8_t  active_axis;
    int32_t active_axis_size;
} aml_kvcache_opt_t;

/**
 * @struct kvCacheInfo_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef struct _kvCacheInfo_t {
    bool               set_kvcache_opt_flag;
    int                kvcache_opt_num;
    aml_kvcache_opt_t* kvcache_opt;
} kvCacheInfo_t;

/**
 * @struct aml_forward_ctrl_t
 * @brief Forward execution control configuration
 */
typedef struct _aml_forward_ctrl_t {
    softOpInfo_t       softop_info;    /**< Software operator optimization info */
    aml_kvcache_type_t kvcache_type;   /**< Deprecated */
    kvCacheInfo_t      kvcache_info;   /**< Deprecated */
} aml_forward_ctrl_t;

/**
 * @struct aml_config_extend
 * @brief Reserved extension field for future expansion
 */
typedef struct _aml_config_extend {
    uint8_t reserved[RESERVED_BYTE];   /**< Reserved bytes for future use */
} aml_config_extend;

/**
 * @struct aml_config
 * @brief Neural network initialization configuration
 */
typedef struct _aml_nn_config {
    int                   typeSize;        /**< Size of this structure */
    const char*           path;            /**< Model file path */
    const char*           pdata;           /**< Pointer to model data in memory */
    int                   length;          /**< Length of model data */
    aml_hw_flag_t         hw_flag;         /**< Hardware backend selection */
    amlnn_model_type      modelType;       /**< Neural network model type */
    amlnn_nbg_type        nbgType;         /**< Model input source type */
    assign_user_address_t inOut;           /**< Deprecated */
    aml_forward_ctrl_t    forward_ctrl;    /**< Forward execution control */
    int                   timeout_ms;      /**< Execution timeout in milliseconds */
    int                   secure_config;   /**< Deprecated */
    const char*           on_path;         /**< Deprecated */
    aml_config_extend     extend;          /**< Reserved extension field */
} aml_config;

/**
 * @enum amlnn_input_type
 * @brief Input data format
 */
typedef enum _amlnn_input_type {
    BINARY_RAW_DATA             = 0,   /**< Raw data */
    TENSOR_RAW_DATA             = 1,   /**< Tensor raw data (float) */
    NV12_RAW_DATA               = 2,   /**< NV12 raw data */
    INPUT_DMA_DATA              = 3,   /**< Deprecated */
    INPUT_DMA_SECURE_DATA       = 4,   /**< Deprecated */
    RGB24_RAW_DATA              = 5,   /**< Deprecated */
    QTENSOR_RAW_DATA            = 6,   /**< Deprecated */
    AMLNN_INPUT_TYPE_MAX        = 10   /**< Maximum enum value */
} amlnn_input_type;

/**
 * @enum aml_input_format_t
 * @brief Input data layout format
 */
typedef enum _aml_input_format_t {
    AML_INPUT_DEFAULT           = 0,   /**< Default format (NHWC) */
    AML_INPUT_MODEL_NHWC        = 1,   /**< NHWC format (batch, height, width, channel) */
    AML_INPUT_MODEL_NCHW        = 2,   /**< NCHW format (batch, channel, height, width) */
    AML_INPUT_FORMAT_MAX        = 10   /**< Maximum enum value */
} aml_input_format_t;

/**
 * @struct input_info
 * @brief Input preprocessing configuration
 */
typedef struct _input_info {
    int                valid;                 /**< Deprecated */
    int                int16_type;            /**< Deprecated */
    int                preprocess_debug;      /**< Deprecated */
    float              mean[INPUT_CHANNEL];   /**< Deprecated */
    float              scale;                 /**< Deprecated */
    aml_input_format_t input_format;          /**< Input data layout format */
} input_info;

/**
 * @struct aml_input_extend
 * @brief Reserved extension field for future expansion
 */
typedef struct _aml_nn_input_extend {
    uint8_t reserved[RESERVED_BYTE];   /**< Reserved bytes for future use */
} aml_input_extend;

/**
 * @struct nn_input
 * @brief Neural network single input configuration
 */
typedef struct _nn_input {
    int               typeSize;             /**< Size of this structure */
    int               input_index;          /**< Input tensor index */
    int               size;                 /**< Input data buffer size (bytes) */
    unsigned char*    input;                /**< Pointer to input data buffer */
    amlnn_input_type  input_type;           /**< Input data format */
    input_info        info;                 /**< Input preprocessing information */
    int               input_valid_length;   /**< Deprecated */
    aml_input_extend  extend;               /**< Reserved extension field */
} nn_input;

/**
 * @struct nn_inputs
 * @brief Neural network multiple input configuration for aml_module_set_multi_input call
 */
typedef struct _nn_inputs {
    unsigned int num;      /**< Number of inputs */
    nn_input*    pInput;   /**< Pointer to input configuration array */
} nn_inputs;

/**
 * @enum aml_perf_mode_t
 * @brief Each stage of the inference API
 *
 * @note
 * This method is generally used to analyze the time consumption of each stage of the inference interface.
 *
 * Configure AML_PERF_OUTPUT_SET, call the inference interface once,
 * then configure AML_PERF_INFERENCE, call the inference interface once more,
 * then configure AML_PERF_OUTPUT_GET, call the inference interface once more,
 * and measure the time consumption three times in sequence. The time consumed during AML_PERF_INFERENCE is the NPU time consumption.
 *
 * If you don't need to focus on precise time consumption,
 * setting it to AML_PERF_ALL or not configuring this parameter at all will result in a single call to the inference interface,
 * and the measured time will be the total process time.
 */
typedef enum _aml_perf_mode_t {
    AML_PERF_ALL                = 0,   /**< entire pipeline */
    AML_PERF_OUTPUT_SET         = 1,   /**< output set stage */
    AML_PERF_INFERENCE          = 2,   /**< inference stage */
    AML_PERF_OUTPUT_GET         = 3,   /**< output get stage */
    AML_PERF_MODE_MAX           = 10   /**< Maximum enum value */
} aml_perf_mode_t;

/**
 * @enum aml_output_format_t
 * @brief Output data format
 */
typedef enum _aml_output_format_t {
    AML_OUTDATA_FLOAT32         = 0,   /**< Float output data */
    AML_OUTDATA_RAW             = 1,   /**< Raw output data */
    AML_OUTDATA_DMA             = 2,   /**< Deprecated */
    AML_OUTDATA_DMA_SECURE      = 3,   /**< Deprecated */
    AML_OUTPUT_FORMAT_MAX       = 10   /**< Maximum enum value */
} aml_output_format_t;

/**
 * @enum aml_output_order_t
 * @brief Output data layout format
 */
typedef enum _aml_output_order_t {
    AML_OUTPUT_ORDER_DEFAULT    = 0,   /**< Default format (NHWC) */
    AML_OUTPUT_ORDER_NHWC       = 1,   /**< NHWC format */
    AML_OUTPUT_ORDER_NCHW       = 2,   /**< NCHW format */
    AML_OUTPUT_ORDER_MAX        = 10   /**< Maximum enum value */
} aml_output_order_t;

/**
 * @enum aml_invoke_type_t
 *
 * @deprecated This enum is deprecated and should not be used.
 */
typedef enum _aml_invoke_type_t {
    AML_INVOKE_NORMAL           = 0,
    AML_INVOKE_NO_WAIT          = 1,
    AML_INVOKE_WAIT_WITHID      = 2,
    AML_INVOKE_TYPE_MAX         = 10
} aml_invoke_type_t;

/**
 * @struct aml_kvcache_dynamic_val_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef struct _aml_kvcache_dynamic_val_t {
    int32_t current_mask;
} aml_kvcache_dynamic_val_t;

/**
 * @struct kvCacheDynamicInfo_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef struct _kvCacheDynamicInfo_t {
    bool                      update_kvcache_info_flag;
    aml_kvcache_dynamic_val_t kvcache_dynamic_val;
} kvCacheDynamicInfo_t;

/**
 * @struct aml_invoke_info_t
 *
 * @deprecated This structure is deprecated and should not be used.
 */
typedef struct _aml_invoke_info_t {
    aml_invoke_type_t    invoke_type;
    int64_t              invoke_id;
    int32_t              timeout_ms;
    kvCacheDynamicInfo_t kvcache_dynamic_info;
} aml_invoke_info_t;

/**
 * @enum aml_module_t
 *
 * @deprecated This enum is deprecated and should not be used.
 */
typedef enum _aml_module_t {
    CUSTOM_NETWORK              = 0,
    AML_MODULE_MAX              = 10
} aml_module_t;

/**
 * @struct aml_output_config_extend
 * @brief Reserved extension field for future expansion
 */
typedef struct _aml_output_config_extend {
    uint8_t reserved[RESERVED_BYTE];   /**< Reserved bytes for future use */
} aml_output_config_extend;

/**
 * @struct aml_output_config_t
 * @brief Neural network output configuration
 */
typedef struct _aml_output_config_t {
    int                      typeSize;   /**< Size of this structure */
    aml_perf_mode_t          perfMode;   /**< Each stage of the inference API */
    aml_output_format_t      format;     /**< Output data format */
    aml_output_order_t       order;      /**< Output data layout format */
    aml_invoke_info_t        invoke;     /**< Deprecated */
    aml_module_t             mdType;     /**< Deprecated */
    aml_output_config_extend extend;     /**< Reserved extension field */
} aml_output_config_t;

/**
 * @enum nn_buffer_format_e
 * @brief Data type of model input and output tensors
 */
typedef enum _nn_buffer_format_e {
    NN_BUFFER_FORMAT_FP32       = 0,    /**< 32-bit float type */
    NN_BUFFER_FORMAT_FP16       = 1,    /**< 16-bit float type (half) */
    NN_BUFFER_FORMAT_UINT8      = 2,    /**< 8-bit unsigned integer type */
    NN_BUFFER_FORMAT_INT8       = 3,    /**< 8-bit signed integer type */
    NN_BUFFER_FORMAT_UINT16     = 4,    /**< 16-bit unsigned integer type */
    NN_BUFFER_FORMAT_INT16      = 5,    /**< 16-bit signed integer type */
    NN_BUFFER_FORMAT_UINT32     = 6,    /**< 32-bit unsigned integer type */
    NN_BUFFER_FORMAT_INT32      = 7,    /**< 32-bit signed integer type */
    NN_BUFFER_FORMAT_UINT64     = 8,    /**< 64-bit unsigned integer type */
    NN_BUFFER_FORMAT_INT64      = 9,    /**< 64-bit signed integer type */
    NN_BUFFER_FORMAT_BOOL       = 10,   /**< Boolean type */
    NN_BUFFER_FORMAT_MAX        = 100   /**< Maximum enum value */
} nn_buffer_format_e;

/**
 * @enum nn_buffer_quantize_format_e
 *
 * @deprecated This enum is deprecated and should not be used.
 */
typedef enum _nn_buffer_quantize_format_e {
    NN_BUFFER_QUANTIZE_NONE                  = 0,
    NN_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT   = 1,
    NN_BUFFER_QUANTIZE_TF_ASYMM              = 2,
    NN_BUFFER_QUANTIZE_FORMAT_MAX            = 10
} nn_buffer_quantize_format_e;

/**
 * @struct nn_buffer_params_t
 * @brief Tensor buffer parameter description
 */
typedef struct _nn_buffer_create_params_t {
    unsigned int                num_of_dims;    /**< Number of tensor dimensions */
    unsigned int                sizes[4];       /**< Size of each dimension */

    nn_buffer_format_e          data_format;    /**< Tensor data type */
    nn_buffer_quantize_format_e quant_format;   /**< Deprecated */

    /**
     * @union quant_data
     * @brief Quantization parameter union
     */
    union {
        /**
         * @struct dfp
         *
         * @deprecated This struct is deprecated and should not be used.
         */
        struct {
            unsigned char       fixed_point_pos;
        } dfp;

        /**
         * @struct affine
         * @brief Affine quantization parameters
         */
        struct {
            float               scale;          /**< Scale factor for quantized values */
            unsigned int        zeroPoint;      /**< Zero point value */
        } affine;
    } quant_data; /**< Quantization information */
} nn_buffer_params_t;

/**
 * @struct outBuf_t
 * @brief Output buffer description
 */
typedef struct _out_buf {
    unsigned int        size;                    /**< Output buffer size in bytes */
    char                name[MAX_NAME_LENGTH];   /**< Output tensor name (reserved for future use) */
    unsigned char*      buf;                     /**< Pointer to output data buffer */
    nn_buffer_params_t* param;                   /**< Pointer to buffer parameter description */
    int                 output_valid_length;     /**< Deprecated */
} outBuf_t;

/**
 * @struct nn_output
 * @brief Neural network output buffers
 */
typedef struct _nnout {
    unsigned int num;   /**< Number of output tensors */
    outBuf_t*    out;   /**< Pointer to output buffer array */
} nn_output;

/**
 * @enum aml_cache_type_t
 * @brief DMA buffer cache policy from CPU perspective
 */
typedef enum _aml_cache_type_t {
    AML_WITH_CACHE              = 0,   /**< Cached buffer (faster CPU access) */
    AML_WITHOUT_CACHE           = 1,   /**< Non-cached buffer */
    AML_CACHE_TYPE_MAX          = 10   /**< Maximum enum value */
} aml_cache_type_t;

/**
 * @enum aml_mem_access_type_t
 * @brief CPU access type for DMA memory
 *
 * @note Currently only read-write access is supported.
 */
typedef enum _aml_mem_access_type_t {
    AML_MEM_ACCESS_READ_WRITE   = 0,   /**< CPU reads and writes the buffer */
    AML_MEM_ACCESS_READ_ONLY    = 1,   /**< CPU read-only access (not implemented) */
    AML_MEM_ACCESS_WRITE_ONLY   = 2,   /**< CPU write-only access (not implemented) */
    AML_MEM_ACCESS_MAX          = 10   /**< Maximum enum value */
} aml_mem_access_type_t;

/**
 * @enum aml_buffer_type_t
 * @brief DMA buffer address type
 */
typedef enum _aml_buffer_type_t {
    AML_VIRTUAL_ADDR            = 0,   /**< Virtual address */
    AML_PHYS_ADDR               = 1,   /**< Physical address */
    AML_PHYS_SECURE_ADDR        = 2,   /**< Deprecated */
    AML_BUFFER_TYPE_MAX         = 10   /**< Maximum enum value */
} aml_buffer_type_t;

/**
 * @struct aml_memory_config_t
 * @brief DMA memory allocation configuration
 */
typedef struct _aml_memory_config_t {
    int                   typeSize;          /**< Size of this structure */
    uint32_t              index;             /**< Memory index identifier */
    uint32_t              mem_size;          /**< Requested memory size (bytes) */
    aml_cache_type_t      cache_type;        /**< Cache policy */
    aml_mem_access_type_t mem_access_type;   /**< CPU access type */
    aml_buffer_type_t     buffer_type;       /**< Buffer address type */
} aml_memory_config_t;

/**
 * @struct aml_memory_data_t
 * @brief DMA memory information
 */
typedef struct _aml_memory_data_t {
    int   typeSize;   /**< Size of this structure */
    void* memory;     /**< Internal memory handle */
    void* viraddr;    /**< Virtual address */
    void* phyaddr;    /**< Physical address */
} aml_memory_data_t;

/**
 * @struct info_t
 * @brief Detailed tensor information
 */
typedef struct _info_t {
    unsigned int dim_count;                           /**< Number of tensor dimensions */
    unsigned int sizes_of_dim[MAX_TENSOR_NUM_DIMS];   /**< Dimension sizes (supports up to 4D) */
    unsigned int data_format;                         /**< Tensor data type (see nn_buffer_format_e) */
    unsigned int data_type;                           /**< Deprecated */
    unsigned int quantization_format;                 /**< Deprecated */
    int   fixed_point_pos;                            /**< Deprecated */
    float TF_scale;                                   /**< scale value */
    int   TF_zeropoint;                               /**< zero point */
    char  name[MAX_NAME_LENGTH];                      /**< Tensor name (reserved for future use) */
} info_t;

/**
 * @struct tensor_info
 * @brief Model input/output tensor information
 */
typedef struct _tensor_info {
    unsigned int valid;   /**< Deprecated */
    unsigned int num;     /**< Number of tensors */
    info_t*      info;    /**< Pointer to tensor information array */
} tensor_info;

/**
 * @struct aml_profiling_ext_data_t
 * @brief Extended profiling information
 */
typedef struct _aml_profiling_ext_data_t {
    uint64_t axi_freq_cur;              /**< Current NPU AXI frequency */
    uint64_t core_freq_cur;             /**< Current NPU core frequency */
    uint64_t mem_alloced_base;          /**< The NN driver occupies memory after insmod */
    uint64_t mem_alloced_umd;           /**< The memory occupied by the current NN model */
    int64_t  mem_pool_size;             /**< The size of the address space supported by the NN driver */
    uint64_t mem_pool_used;             /**< The total memory size occupied by the current NN driver
                                             mem_pool_used = mem_alloced_base +
                                             mem_alloced_umd(modelA) + mem_alloced_umd(modelB) + ... + mem_alloced_umd (modelN) */
    int32_t  us_elapsed_in_fixup_cmq;   /**< Deprecated */
    int32_t  us_elapsed_in_hw_op;       /**< Hardware operator execution time;
                                             includes system overhead and interrupt latency,
                                             therefore >= inference_time_us */
    int32_t  us_elapsed_in_sw_op;       /**< Software operator execution time */
    int32_t invoke_has_error;           /**< record potential kmd warnings during profiling.
                                             these do not affect normal inference. */
} aml_profiling_ext_data_t;

/**
 * @struct aml_profiling_data_t
 * @brief Profiling data for a single inference
 */
typedef struct _aml_profiling_data_t {
    uint64_t inference_time_us;     /**< Theoretical hardware inference time */
    uint64_t memory_usage_bytes;    /**< Theoretical memory usage of the model */
    uint64_t dram_read_bytes;       /**< DDR read traffic per inference (bandwidth) */
    uint64_t dram_write_bytes;      /**< DDR write traffic per inference (bandwidth) */
    uint64_t sram_read_bytes;       /**< SRAM read traffic per inference (bandwidth) */
    uint64_t sram_write_bytes;      /**< SRAM write traffic per inference (bandwidth) */
    aml_profiling_ext_data_t ext;   /**< Extended profiling data */
} aml_profiling_data_t;

/**
 * @enum aml_platform_type_t
 * @brief Platform type
 */
typedef enum _aml_platform_type_t {
    AML_PLATFORM_UNKNOWN        = -1,   /**< Unknown */
    AML_PLATFORM_C308L          = 0,    /**< C308L / C302X platform */
    AML_PLATFORM_S928X          = 1,    /**< S928X platform */
    AML_PLATFORM_A311D2         = 2,    /**< A311D2 platform */
    AML_PLATFORM_T968D4         = 3,    /**< T968D4 platform */
    AML_PLATFORM_S905X5         = 4,    /**< S905X5 / S905D5 platform */
    AML_PLATFORM_C302X2         = 5,    /**< C302X2 platform */
    AML_PLATFORM_C305X2         = 6,    /**< C305X2 platform */
    AML_PLATFORM_A311Y3         = 7,    /**< A311Y3 platform */
    AML_PLATFORM_MAX            = 100   /**< Maximum enum value */
} aml_platform_type_t;

/**
 * @enum aml_npu_type_t
 * @brief NPU type
 */
typedef enum _aml_npu_type_t {
    AML_NPU_INVALID             = -1,   /**< Invalid NPU type */
    AML_NPU_ADLA                = 0,    /**< ADLA */
    AML_NPU_VSI_UNIFY           = 1,    /**< VSI */
    AML_NPU_VIPLITE             = 2,    /**< VIPLite */
    AML_NPU_TYPE_MAX            = 10    /**< Maximum enum value */
} aml_npu_type_t;

/**
 * @struct aml_npu_hw_info_t
 * @brief NPU hardware information
 */
typedef struct _aml_npu_hw_info_t {
    char     hw_version[10];   /**< NPU hardware version string */
    uint32_t i8_mac_cnt;       /**< Number of MAC units */
    uint32_t max_clk;          /**< ADLA maximum number of clocks */
    uint32_t cur_clk;          /**< ADLA current actual clock */
    uint64_t sram_size;        /**< AXI sram size */
    uint32_t Gops;             /**< ADLA computing power */
} aml_npu_hw_info_t;

/**
 * @struct aml_platform_info_t
 * @brief Platform information
 */
typedef struct _aml_platform_info_t {
    char*               sdk_version;     /**< SDK version string */
    aml_platform_type_t platform_type;   /**< Platform type */
    aml_npu_type_t      npu_type;        /**< NPU type */
    aml_npu_hw_info_t   npu_hw_info;     /**< NPU hardware information */
} aml_platform_info_t;

/**
 * @enum aml_npu_status_t
 * @brief NPU status
 */
typedef enum _aml_npu_status_t {
    AML_NPU_STATUS_INIT         = 1,   /**< NPU initialized */
    AML_NPU_STATUS_ERR          = 2,   /**< NPU error state */
    AML_NPU_STATUS_IDLE         = 3,   /**< NPU idle */
    AML_NPU_STATUS_BUSY         = 4,   /**< NPU busy */
    AML_NPU_STATUS_POWEROFF     = 5,   /**< NPU powered off */
    AML_NPU_STATUS_MAX          = 10   /**< Maximum enum value */
} aml_npu_status_t;

/**
 * @enum aml_policy_type_t
 * @brief Power and performance policy
 */
typedef enum _aml_policy_type_t {
    AML_PERFORMANCE_MODE        = 1,   /**< Maximum performance mode */
    AML_POWER_SAVE_MODE         = 2,   /**< Balanced power-saving mode */
    AML_MINIMUM_POWER_MODE      = 3,   /**< Minimum power consumption mode */
    AML_POLICY_TYPE_MAX         = 10   /**< Maximum enum value */
} aml_policy_type_t;

/**
 * @enum aml_flush_type_t
 *
 * @deprecated This enum is deprecated and should not be used.
 */
typedef enum _aml_flush_type_t {
    AML_INPUT_TENSOR            = 0,
    AML_OUTPUT_TENSOR           = 1,
    AML_FLUSH_TYPE_MAX          = 10
} aml_flush_type_t;

/**
 * @enum aml_profile_type_t
 *
 * @deprecated This enum is deprecated and should not be used.
 */
typedef enum _aml_profile_type_t {
    AML_PROFILE_NONE            = 0,
    AML_PROFILE_PERFORMANCE     = 1,
    AML_PROFILE_BANDWIDTH       = 2,
    AML_PROFILE_MEMORY          = 3,
    AML_PERLAYER_RUNTIME        = 4,
    AML_PERLAYER_BANDWIDTH      = 5,
    AML_PERLAYER_OUTPUT         = 6,
    AML_PERLAYER_INPUT          = 7,
    AML_PROFILE_TYPE_MAX        = 10
} aml_profile_type_t;


/**
 * =============================================================
 *                        NNSDK Base API
 * =============================================================
 *  This section defines the primary interfaces for creating,
 *  configuring, running, and destroying an AML neural network
 *  module instance.
 *
 *  Each function operates on a runtime context (`void* context`)
 *  returned by `aml_module_create()`.
 * =============================================================
 */
/**
 * @brief Create aml network module.
 *
 * @param config Initialize configuration parameters pointer.
 * @return Aml runtime context pointer.
 */
void* aml_module_create(aml_config* config);

/**
 * @brief Set network input.
 *
 * @param context Aml runtime context pointer.
 * @param pInput Input configuration parameters pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_module_input_set(void* context, nn_input* pInput);

/**
 * @brief Set multiple inputs for the network.
 *
 * @param context Aml runtime context pointer.
 * @param pInputs Multiple input configuration parameters pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_module_set_multi_input(void* context, nn_inputs* pInputs);

/**
 * @brief Run and get output.
 *
 * @param context Aml runtime context pointer.
 * @param outconfig Output configuration parameters.
 * @return Output information pointer.
 */
void* aml_module_output_get(void* context, aml_output_config_t outconfig);

/**
 * @brief Destroy network environment, free the alloced buffer.
 *
 * @param context Aml runtime context pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_module_destroy(void* context);

/**
 * =============================================================
 *                        NNSDK DMA API
 * =============================================================
 *  This section defines utility interfaces for allocating,
 *  binding, switching, and releasing DMA buffers used by
 *  the AML neural network runtime.
 *
 *  Each function operates on an AML runtime context (`void* context`)
 *  returned by `aml_module_create()`.
 * =============================================================
 */
/**
 * @brief Allocate dma buf.
 *
 * @param context Aml runtime context pointer.
 * @param mem_config Dma configuration parameters pointer.
 * @param mem_data Dma mem info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_mallocBuffer(void* context, aml_memory_config_t* mem_config, aml_memory_data_t* mem_data);

/**
 * @brief Free dma buf.
 *
 * @param context Aml runtime context pointer.
 * @param mem_config Dma configuration parameters pointer.
 * @param mem_data Dma mem info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_freeBuffer(void* context, aml_memory_config_t* mem_config, aml_memory_data_t* mem_data);

/**
 * @brief Bind input dma buf (If buf is a physical address, then buf needs to meet the condition of being 4K aligned to its starting address before it can be called).
 *
 * @param context Aml runtime context pointer.
 * @param mem_config Dma configuration parameters pointer.
 * @param mem_data Dma mem info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_swapExternalInputBuffer(void* context, aml_memory_config_t* mem_config, aml_memory_data_t* mem_data);

/**
 * @brief Bind output dma buf (If buf is a physical address, then buf needs to meet the condition of being 4K aligned to its starting address before it can be called).
 *
 * @param context Aml runtime context pointer.
 * @param mem_config Dma configuration parameters pointer.
 * @param mem_data Dma mem info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_swapExternalOutputBuffer(void* context, aml_memory_config_t* mem_config, aml_memory_data_t* mem_data);

/**
 * @brief Unbind dma buf.
 *
 * @param context Aml runtime context pointer.
 * @param mem_config Dma configuration parameters pointer.
 * @param mem_data Dma mem info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_flushBuffer(void* context, aml_memory_config_t* mem_config, aml_memory_data_t* mem_data);

/**
 * =============================================================
 *                      NNSDK Utility API
 * =============================================================
 *  This section provides utility functions for profiling,
 *  memory management, power policy configuration, NPU status
 *  monitoring in the AML runtime.
 * =============================================================
 */
/**
 * @brief Allocate memory and get tensor info.
 *
 * @param context Aml runtime context pointer.
 * @param model_data Unused, set NULL.
 * @param in_tInfo Input tensor double pointer.
 * @param out_tInfo Output tensor double pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_getTensorInfo(void* context, const char* model_data, tensor_info** in_tInfo, tensor_info** out_tInfo);

/**
 * @brief Free tensor info memory.
 *
 * @param context Aml runtime context pointer.
 * @param tinfo Tensor info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_freeTensorInfo(void* context, tensor_info* tinfo);

/**
 * @brief Enable profile.
 *
 * @param context Aml runtime context pointer.
 * @param profile_data Profile data pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_enableProfile(void* context, aml_profiling_data_t* profile_data);

/**
 * @brief Get profile info.
 *
 * @param context Aml runtime context pointer.
 * @param profile_data Profile data pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_getProfileInfo(void* context, aml_profiling_data_t* profile_data);

/**
 * @brief Didable profile.
 *
 * @param context Aml runtime context pointer.
 * @param profile_data Profile data pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_disableProfile(void* context, aml_profiling_data_t* profile_data);

/**
 * @brief Get platform info.
 *
 * @param platform_info Platform info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 *
 * @note C308L does not support get platform_info->npu_hw_info.
 */
NNSDK_Status aml_util_getPlatformInfo(aml_platform_info_t* platform_info);

/**
 * @brief Get NPU status.
 *
 * @param npu_status NPU status info pointer.
 * @return Status code (0 for success, non-zero indicates failure).
 */
NNSDK_Status aml_util_getNpuStatus(aml_npu_status_t* npu_status);

/**
 * @brief Set power policy.
 *
 * @param type Policy type.
 * @return Status code (0 for success, non-zero indicates failure).
 *
 * @note C308L does not support set power policy.
 */
NNSDK_Status aml_util_setPowerPolicy(aml_policy_type_t type);

/**
 * @brief Set NPU timeout to suspend.
 *
 * @param timeout Timeout.
 * @return Status code (0 for success, non-zero indicates failure).
 *
 * @note C308L does not support set auto suspend.
 */
NNSDK_Status aml_util_setAutoSuspend(int timeout);

/**
 * =============================================================
 *  Some older APIs have been abandoned.
 * =============================================================
 */
unsigned char *aml_util_mallocAlignedBuffer(void* context, int mem_size, aml_memory_config_t* mem_config);
void aml_util_freeAlignedBuffer(void* context, unsigned char* addr);
int aml_util_swapInputBuffer(void* context, void* newBuffer, unsigned int inputId);
int aml_util_swapOutputBuffer(void* context, void* newBuffer, unsigned int outputId);
int aml_util_switchInputBuffer(void* context, void* newBuffer, unsigned int inputId);
int aml_util_switchOutputBuffer(void* context, void* newBuffer, unsigned int outputId);
tensor_info* aml_util_getInputTensorInfo(const char* nbgdata);
tensor_info* aml_util_getOutputTensorInfo(const char* nbgdata);
int aml_util_flushTensorHandle(void* context, aml_flush_type_t type);
int aml_util_setProfile(aml_profile_type_t type, const char* savepath);
int aml_util_getHardwareStatus(int* customID, int* powerStatus, int* version);


#ifdef __cplusplus
} //extern "C"
#endif

#endif // _NN_SDK_H