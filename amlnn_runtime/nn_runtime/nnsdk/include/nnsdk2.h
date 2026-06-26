/*
* Copyright (C) 2026 Amlogic, Inc. All rights reserved.
*
* This source code is subject to the terms and conditions defined in the
* file 'LICENSE' which is part of this source code package.
*
* Description: aml_nnsdk2
*/

#ifndef _NNSDK2_H
#define _NNSDK2_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
    Error code returned by the AMLNN API.
*/
#define AMLNN_SUCCESS                            0       /* succeed. */
#define AMLNN_ERR_FAIL                           -1      /* failed. */
#define AMLNN_ERR_TIMEOUT                        -2      /* timeout. */
#define AMLNN_ERR_DEVICE_UNAVAILABLE             -3      /* device unavailable. */
#define AMLNN_ERR_MALLOC_FAIL                    -4      /* malloc fail. */
#define AMLNN_ERR_PARAM_INVALID                  -5      /* parameter invalid. */
#define AMLNN_ERR_MODEL_INVALID                  -6      /* model invalid. */
#define AMLNN_ERR_CTX_INVALID                    -7      /* context invalid. */
#define AMLNN_ERR_INPUT_INVALID                  -8      /* input invalid. */
#define AMLNN_ERR_OUTPUT_INVALID                 -9      /* output invalid. */
#define AMLNN_ERR_NOT_SUPPORTED                  -10     /* not supported. */
#define AMLNN_ERR_NOT_READY                      -11     /* not ready. */
#define AMLNN_ERR_NOT_MATCH                      -12     /* not match. */

/*
    Tensor limits.
*/
#define AMLNN_MAX_DIMS                           6       /* maximum dimension of tensor. */
#define AMLNN_MAX_NAME_LEN                       256     /* maximum name length of tensor. */
#define AMLNN_MAX_DYNAMIC_SHAPE_NUM              64      /* maximum number of dynamic shape for each input. */


/*
    The query command for amlnn_query.
*/
typedef enum _amlnn_query_cmd
{
    AMLNN_QUERY_SDK_VERSION = 0,          /* query the sdk version */
    AMLNN_QUERY_IN_OUT_NUM = 1,           /* query the number of input & output tensor. */
    AMLNN_QUERY_INPUT_ATTR = 2,           /* query the attribute of input tensor. */
    AMLNN_QUERY_OUTPUT_ATTR = 3,          /* query the attribute of output tensor. */
    AMLNN_QUERY_INPUT_DYNAMIC_RANGE = 4,  /* query the dynamic shape range of input tensor. */
    AMLNN_QUERY_CURRENT_INPUT_ATTR = 5,   /* query the current shape of input tensor, only valid for dynamic model. */
    AMLNN_QUERY_CURRENT_OUTPUT_ATTR = 6,  /* query the current shape of output tensor, only valid for dynamic model. */
    AMLNN_QUERY_PERF_DETAIL = 7,          /* query the profiling data and trigger per-layer txt output,
                                             need to enable the enable_perf in amlnn_init_config and set the perf_detail_path when call amlnn_init,
                                             this query needs to be valid after amlnn_run. (Only supports ADLA NPU) */
    AMLNN_QUERY_PERF_RUN = 8,             /* query the time of run,
                                             this query needs to be valid after amlnn_run. */
    AMLNN_QUERY_MODEL_SOFTOP = 9,         /* query model software operator info. (Only supports ADLA NPU) (TODO) */
    AMLNN_QUERY_NPU_CORE_STATUS = 10,     /* query NPU core status. (Only supports ADLA NPU) */
    AMLNN_QUERY_CMD_MAX
} amlnn_query_cmd;

/*
    The information for AMLNN_QUERY_SDK_VERSION.
*/
typedef struct _amlnn_sdk_version
{
    char sdk2_api_version[64];            /* the version of aml nnsdk2 api. */
    char adla_kmd_version[64];            /* the version of adla driver (NPU). */
    char delegate_version[64];            /* the version of delegate (GPU/CPU). */
} amlnn_sdk_version;

/*
    The information for AMLNN_QUERY_IN_OUT_NUM.
*/
typedef struct _amlnn_input_output_num
{
    uint32_t n_input;
    uint32_t n_output;
} amlnn_input_output_num;

/*
    The tensor data format.
*/
typedef enum _amlnn_tensor_format
{
    AMLNN_TENSOR_NHWC = 0,
    AMLNN_TENSOR_NCHW = 1,
    AMLNN_TENSOR_FORMAT_MAX
} amlnn_tensor_format;

inline static const char* get_format_string(amlnn_tensor_format fmt)
{
    switch(fmt)
    {
    case AMLNN_TENSOR_NHWC: return "NHWC";
    case AMLNN_TENSOR_NCHW: return "NCHW";
    default: return "UNKNOWN";
    }
}

/*
    The tensor data type.
*/
typedef enum _amlnn_tensor_type
{
    AMLNN_TENSOR_FLOAT32 = 0,
    AMLNN_TENSOR_FLOAT16,
    AMLNN_TENSOR_INT8,
    AMLNN_TENSOR_UINT8,
    AMLNN_TENSOR_INT16,
    AMLNN_TENSOR_UINT16,
    AMLNN_TENSOR_INT32,
    AMLNN_TENSOR_UINT32,
    AMLNN_TENSOR_INT64,
    AMLNN_TENSOR_UINT64,
    AMLNN_TENSOR_BOOL,
    AMLNN_TENSOR_INT4,
    AMLNN_TENSOR_BFLOAT16,
    AMLNN_TENSOR_TYPE_MAX
} amlnn_tensor_type;

inline static const char* get_type_string(amlnn_tensor_type type)
{
    switch(type)
    {
    case AMLNN_TENSOR_FLOAT32: return "FP32";
    case AMLNN_TENSOR_FLOAT16: return "FP16";
    case AMLNN_TENSOR_INT8: return "INT8";
    case AMLNN_TENSOR_UINT8: return "UINT8";
    case AMLNN_TENSOR_INT16: return "INT16";
    case AMLNN_TENSOR_UINT16: return "UINT16";
    case AMLNN_TENSOR_INT32: return "INT32";
    case AMLNN_TENSOR_UINT32: return "UINT32";
    case AMLNN_TENSOR_INT64: return "INT64";
    case AMLNN_TENSOR_UINT64: return "UINT64";
    case AMLNN_TENSOR_BOOL: return "BOOL";
    case AMLNN_TENSOR_INT4: return "INT4";
    case AMLNN_TENSOR_BFLOAT16: return "BF16";
    default: return "UNKNOWN";
    }
}

/*
    The information for AMLNN_QUERY_INPUT_ATTR / AMLNN_QUERY_OUTPUT_ATTR.
*/
typedef struct _amlnn_tensor_attr
{
    uint32_t index;                                 /* input parameter, the index of input/output tensor. */
    uint32_t n_dims;                                /* the number of dimensions. */
    uint32_t dims[AMLNN_MAX_DIMS];                  /* the dimensions array. */
    char name[AMLNN_MAX_NAME_LEN];                  /* the name of tensor. */

    uint32_t n_elems;                               /* the number of elements. */
    uint32_t size;                                  /* the bytes size of tensor. */
    uint32_t size_with_stride;                      /* the bytes size of tensor with stride (For DMA mode). */

    amlnn_tensor_format fmt;                        /* the data format of tensor. */
    amlnn_tensor_type type;                         /* the data type of tensor. */

    int32_t zp;                                     /* zero point. */
    float scale;                                    /* scale. */
} amlnn_tensor_attr;

/*
    The information for AMLNN_QUERY_INPUT_DYNAMIC_RANGE.
*/
typedef struct _amlnn_input_range
{
    uint32_t index;                                                   /* input parameter, the index of input tensor. */
    uint32_t shape_number;                                            /* the number of shapes. */
    amlnn_tensor_format fmt;                                          /* the data format of tensor. */
    char name[AMLNN_MAX_NAME_LEN];                                    /* the name of tensor. */
    uint32_t dyn_range[AMLNN_MAX_DYNAMIC_SHAPE_NUM][AMLNN_MAX_DIMS];  /* the dynamic input dimensions range. */
    uint32_t n_dims;                                                  /* the number of dimensions. */
} amlnn_input_range;

/*
    The information for AMLNN_QUERY_PERF_DETAIL. (Only supports ADLA NPU)
*/
typedef struct _amlnn_profiling_data
{
    uint64_t us_elapsed_in_hw_op;                   /* hardware operator execution time. */
    uint64_t us_elapsed_in_sw_op;                   /* software operator execution time. */
    uint64_t memory_usage_bytes;                    /* memory usage. */
    uint64_t dram_read_bytes;                       /* DDR read traffic (Bandwidth). */
    uint64_t dram_write_bytes;                      /* DDR write traffic (Bandwidth). */
    uint64_t sram_read_bytes;                       /* SRAM read traffic (Bandwidth). */
    uint64_t sram_write_bytes;                      /* SRAM write traffic (Bandwidth). */

    uint64_t axi_freq_cur;                          /* current NPU AXI frequency. */
    uint64_t core_freq_cur;                         /* current NPU core frequency. */
    uint64_t mem_alloced_base;                      /* the NN driver occupies memory after insmod. */
    uint64_t mem_alloced_umd;                       /* the memory occupied by the current NN model. */
    uint64_t mem_pool_size;                         /* the size of the address space supported by the NN driver. */
    uint64_t mem_pool_used;                         /* the total memory size occupied by the current NN driver.
                                                       mem_pool_used = mem_alloced_base +
                                                       mem_alloced_umd(modelA) + mem_alloced_umd(modelB) + ... + mem_alloced_umd(modelN) */
    uint8_t reserved[64];                           /* reserved. */
} amlnn_profiling_data;

/*
    The information for AMLNN_QUERY_PERF_RUN.
*/
typedef struct _amlnn_perf_run
{
    int64_t run_duration;                           /* real inference time (us). */
} amlnn_perf_run;

/*
    The information for AMLNN_QUERY_NPU_CORE_STATUS. (Only supports ADLA NPU)
*/
typedef struct _amlnn_npu_core_status
{
    uint32_t core_num;                             /* logical NPU core count. */
    uint64_t core_mask;                            /* all logical NPU cores supported on the current device. */
    uint64_t idle_mask;                            /* NPU cores currently idle. */
    uint64_t busy_mask;                            /* NPU cores currently busy. */
    uint8_t reserved[64];                          /* reserved. */
} amlnn_npu_core_status;

/*
    The backend type for init.
*/
typedef enum _amlnn_backend_type
{
    AMLNN_BACKEND_ADLA_NPU = 0,                     /* run adla model on ADLA NPU. */
    AMLNN_BACKEND_TF_DELEGATE_GPU = 1,              /* run tflite model on GPU. */
    AMLNN_BACKEND_TF_DELEGATE_CPU = 2,              /* run tflite model on CPU. */
    AMLNN_BACKEND_TYPE_MAX
} amlnn_backend_type;

/*
    The model task priority for init.
*/
typedef enum _amlnn_model_task_prior
{
    AMLNN_MODEL_TASK_PRIOR_MEDIUM = 0,              /* medium model task priority */
    AMLNN_MODEL_TASK_PRIOR_LOW = 1,                 /* low model task priority */
    AMLNN_MODEL_TASK_PRIOR_HIGH = 2,                /* high model task priority */
    AMLNN_MODEL_TASK_PRIORITY_MAX
} amlnn_model_task_priority;

/*
    The init config for amlnn_init.
*/
typedef struct _amlnn_init_config
{
    amlnn_backend_type backend_type;                /* backend type. (The following parameters only supports ADLA NPU) */
    uint32_t flags;                                 /* init flags. (TODO) */
    amlnn_model_task_priority task_priority;        /* model task priority. */
    uint32_t timeout_ms;                            /* timeout for invoke. */
    uint32_t enable_perf;                           /* enable to get perf data (1: Enable, 0: Disable). */
    const char* perf_detail_path;                   /* output path for perf detail txt. */
    uint8_t reserved[64];                           /* reserved. */
} amlnn_init_config;

/*
    The input information for amlnn_inputs_set.
*/
typedef struct _amlnn_input
{
    uint32_t index;                                 /* the input index. */
    void* buf;                                      /* the input buffer. */
    uint32_t size;                                  /* the bytes size of input buffer. */
} amlnn_input;

/*
    The run config for amlnn_run.
*/
typedef struct _amlnn_run_config
{
    uint8_t reserved[64];                           /* reserved. */
} amlnn_run_config;

/*
    The output information for amlnn_outputs_get.
*/
typedef struct _amlnn_output
{
    uint32_t is_float;                              /* convert output data to float (1: Return float data, 0: Return raw data). */
    uint32_t index;                                 /* the output index. */
    void* buf;                                      /* the output buffer. */
    uint32_t size;                                  /* the bytes size of output buffer. */
} amlnn_output;

/*
    The information for DMA memory. (Only supports ADLA NPU)
*/
typedef struct _amlnn_dma_memory
{
    void* handle;                                   /* internal memory handle. */
    void* viraddr;                                  /* virtual address. */
    void* phyaddr;                                  /* physical address. */
    uint32_t index;                                 /* memory index identifier. */
    uint32_t size;                                  /* requested memory bytes size. */
} amlnn_dma_memory;

/*
    The access mode for DMA memory. (Only supports ADLA NPU)
*/
typedef enum _amlnn_ext_mem_access_mode
{
    AMLNN_EXT_MEM_ACCESS_READ_WRITE = 0,            /* CPU reads and writes the buffer. */
    AMLNN_EXT_MEM_ACCESS_READ_ONLY = 1,             /* CPU read-only access. (TODO) */
    AMLNN_EXT_MEM_ACCESS_WRITE_ONLY = 2,            /* CPU write-only access. (TODO) */
    AMLNN_EXT_MEM_ACCESS_MODE_MAX
} amlnn_ext_mem_access_mode;

/*
    The buffer address type for DMA memory. (Only supports ADLA NPU)
*/
typedef enum _amlnn_ext_buffer_type
{
    AMLNN_EXT_BUFFER_VIRTUAL_ADDR = 0,              /* virtual address. */
    AMLNN_EXT_BUFFER_PHYS_ADDR = 1,                 /* physical address. */
    AMLNN_EXT_BUFFER_SECURE_PHYS_ADDR = 2,          /* secure physical address. */
    AMLNN_EXT_BUFFER_TYPE_MAX
} amlnn_ext_buffer_type;

/*
    The IO direction for DMA memory. (Only supports ADLA NPU)
*/
typedef enum _amlnn_ext_io_type
{
    AMLNN_EXT_IO_INPUT = 0,                         /* bind as input. */
    AMLNN_EXT_IO_OUTPUT = 1,                        /* bind as output. */
    AMLNN_EXT_IO_TYPE_MAX
} amlnn_ext_io_type;

/*
    The configuration for DMA memory. (Only supports ADLA NPU)
*/
typedef struct _amlnn_dma_config
{
    uint32_t cacheable;                             /* enable caching for faster CPU access (1: Cacheable, 0: Non-cacheable). */
    amlnn_ext_mem_access_mode access_mode;          /* CPU access type. */
    amlnn_ext_buffer_type buffer_type;              /* buffer address type. */
    amlnn_ext_io_type io_type;                      /* input or output. */
} amlnn_dma_config;

/*
    The information for software operator acceleration. (Only supports ADLA NPU)
*/
typedef enum _amlnn_softop_type
{
    AMLNN_Add = 0,
    AMLNN_AveragePool2d = 1,
    AMLNN_Concatenation = 2,
    AMLNN_Conv2d = 3,
    AMLNN_DepthwiseConv2d = 4,
    AMLNN_DepthToSpace = 5,
    AMLNN_Dequantize = 6,
    AMLNN_EmbeddingLookup = 7,
    AMLNN_Floor = 8,
    AMLNN_FullyConnected = 9,
    AMLNN_HashtableLookup = 10,
    AMLNN_L2Normalization = 11,
    AMLNN_L2Pool2d = 12,
    AMLNN_LocalResponseNormalization = 13,
    AMLNN_Logistic = 14,
    AMLNN_LshProjection = 15,
    AMLNN_Lstm = 16,
    AMLNN_MaxPool2d = 17,
    AMLNN_Mul = 18,
    AMLNN_Relu = 19,
    AMLNN_ReluN1To1 = 20,
    AMLNN_Relu6 = 21,
    AMLNN_Reshape = 22,
    AMLNN_ResizeBilinear = 23,
    AMLNN_Rnn = 24,
    AMLNN_Softmax = 25,
    AMLNN_SpaceToDepth = 26,
    AMLNN_Svdf = 27,
    AMLNN_Tanh = 28,
    AMLNN_ConcatEmbeddings = 29,
    AMLNN_SkipGram = 30,
    AMLNN_Call = 31,
    AMLNN_Custom = 32,
    AMLNN_EmbeddingLookupSparse = 33,
    AMLNN_Pad = 34,
    AMLNN_UnidirectionalSequenceRnn = 35,
    AMLNN_Gather = 36,
    AMLNN_BatchToSpaceNd = 37,
    AMLNN_SpaceToBatchNd = 38,
    AMLNN_Transpose = 39,
    AMLNN_Mean = 40,
    AMLNN_Sub = 41,
    AMLNN_Div = 42,
    AMLNN_Squeeze = 43,
    AMLNN_UnidirectionalSequenceLstm = 44,
    AMLNN_StridedSlice = 45,
    AMLNN_BidirectionalSequenceRnn = 46,
    AMLNN_Exp = 47,
    AMLNN_TopkV2 = 48,
    AMLNN_Split = 49,
    AMLNN_LogSoftmax = 50,
    AMLNN_Delegate = 51,
    AMLNN_BidirectionalSequenceLstm = 52,
    AMLNN_Cast = 53,
    AMLNN_Prelu = 54,
    AMLNN_Maximum = 55,
    AMLNN_ArgMax = 56,
    AMLNN_Minimum = 57,
    AMLNN_Less = 58,
    AMLNN_Neg = 59,
    AMLNN_PadV2 = 60,
    AMLNN_Greater = 61,
    AMLNN_GreaterEqual = 62,
    AMLNN_LessEqual = 63,
    AMLNN_Select = 64,
    AMLNN_Slice = 65,
    AMLNN_Sin = 66,
    AMLNN_TransposeConv = 67,
    AMLNN_SparseToDense = 68,
    AMLNN_Tile = 69,
    AMLNN_ExpandDims = 70,
    AMLNN_Equal = 71,
    AMLNN_NotEqual = 72,
    AMLNN_Log = 73,
    AMLNN_Sum = 74,
    AMLNN_Sqrt = 75,
    AMLNN_Rsqrt = 76,
    AMLNN_Shape = 77,
    AMLNN_Pow = 78,
    AMLNN_ArgMin = 79,
    AMLNN_FakeQuant = 80,
    AMLNN_ReduceProd = 81,
    AMLNN_ReduceMax = 82,
    AMLNN_Pack = 83,
    AMLNN_LogicalOr = 84,
    AMLNN_OneHot = 85,
    AMLNN_LogicalAnd = 86,
    AMLNN_LogicalNot = 87,
    AMLNN_Unpack = 88,
    AMLNN_ReduceMin = 89,
    AMLNN_FloorDiv = 90,
    AMLNN_ReduceAny = 91,
    AMLNN_Square = 92,
    AMLNN_ZerosLike = 93,
    AMLNN_Fill = 94,
    AMLNN_FloorMod = 95,
    AMLNN_Range = 96,
    AMLNN_ResizeNearestNeighbor = 97,
    AMLNN_LeakyRelu = 98,
    AMLNN_SquaredDifference = 99,
    AMLNN_MirrorPad = 100,
    AMLNN_Abs = 101,
    AMLNN_SplitV = 102,
    AMLNN_Unique = 103,
    AMLNN_Ceil = 104,
    AMLNN_ReverseV2 = 105,
    AMLNN_AddN = 106,
    AMLNN_GatherNd = 107,
    AMLNN_Cos = 108,
    AMLNN_Where = 109,
    AMLNN_Rank = 110,
    AMLNN_Elu = 111,
    AMLNN_ReverseSequence = 112,
    AMLNN_MatrixDiag = 113,
    AMLNN_Quantize = 114,
    AMLNN_MatrixSetDiag = 115,
    AMLNN_Round = 116,
    AMLNN_HardSwish = 117,
    AMLNN_If = 118,
    AMLNN_While = 119,
    AMLNN_NonMaxSuppressionV4 = 120,
    AMLNN_NonMaxSuppressionV5 = 121,
    AMLNN_ScatterNd = 122,
    AMLNN_SelectV2 = 123,
    AMLNN_Densify = 124,
    AMLNN_SegmentSum = 125,
    AMLNN_BatchMatmul = 126,
    AMLNN_PlaceholderForGreaterOpCodes = 127,
    AMLNN_Cumsum = 128,
    AMLNN_CallOnce = 129,
    AMLNN_BroadcastTo = 130,
    AMLNN_Rfft2d = 131,
    AMLNN_Conv3d = 132,
    AMLNN_Imag = 133,
    AMLNN_Real = 134,
    AMLNN_ComplexAbs = 135,
    AMLNN_Hashtable = 136,
    AMLNN_HashtableFind = 137,
    AMLNN_HashtableImport = 138,
    AMLNN_HashtableSize = 139,
    AMLNN_ReduceAll = 140,
    AMLNN_Conv3dTranspose = 141,
    AMLNN_VarHandle = 142,
    AMLNN_ReadVariable = 143,
    AMLNN_AssignVariable = 144,
    AMLNN_BroadcastArgs = 145,
    AMLNN_RandomStandardNormal = 146,
    AMLNN_Bucketize = 147,
    AMLNN_RandomUniform = 148,
    AMLNN_Multinomial = 149,
    AMLNN_Gelu = 150,
    AMLNN_DynamicUpdateSlice = 151,
    AMLNN_Relu0To1 = 152,
    AMLNN_UnsortedSegmentProd = 153,
    AMLNN_UnsortedSegmentMax = 154,
    AMLNN_UnsortedSegmentSum = 155,
    AMLNN_Atan2 = 156,
    AMLNN_UnsortedSegmentMin = 157,
    AMLNN_Sign = 158,
    AMLNN_Bitcast = 159,
    AMLNN_BitwiseXor = 160,
    AMLNN_RightShift = 161,
    AMLNN_DetectionPostProcess = 256,
    AMLNN_Erf = 260,
    AMLNN_Hardware = 511,
    AMLNN_Unknown = 2147483647,
    AMLNN_MIN = AMLNN_Add,
    AMLNN_OPERATOR_MAX = AMLNN_Unknown
} amlnn_softop_type;

/*
    The acceleration type for software operator. (Only supports ADLA NPU)
*/
typedef enum _amlnn_softop_acc_type
{
    AMLNN_SOFTOP_ACC_OPENMP = 0,                    /* OpenMP acceleration. */
    AMLNN_SOFTOP_ACC_VULKAN = 1,                    /* Vulkan acceleration. (TODO) */
    AMLNN_SOFTOP_ACC_OPENCL = 2,                    /* OpenCL acceleration. (TODO) */
    AMLNN_SOFTOP_ACC_TYPE_MAX
} amlnn_softop_acc_type;

/*
    The request for setting software operator acceleration. (Only supports ADLA NPU)
*/
typedef struct _amlnn_softop_opt_request
{
    amlnn_softop_type softop_type;                  /* software operator type. */
    amlnn_softop_acc_type acc_type;                 /* software operator acceleration type. */
} amlnn_softop_opt_request;

/*
    The policy for setting NPU core mask.
*/
typedef enum _amlnn_core_mask_policy
{
    AMLNN_CORE_MASK_AUTO = 0,                      /* system automatically selects NPU cores. */
    AMLNN_CORE_MASK_BEST_EFFORT = 1,               /* runtime tries to use requested cores, but does not guarantee it. */
    AMLNN_CORE_MASK_POLICY_MAX
} amlnn_core_mask_policy;

/*
    The configuration for amlnn_set_core_mask. (Only supports ADLA NPU)
*/
typedef struct _amlnn_core_mask_config
{
    uint64_t requested_mask;                       /* input: requested NPU core mask, 0 means auto. */
    amlnn_core_mask_policy policy;                 /* input: core mask policy. */
    uint8_t reserved[64];                          /* reserved. */
} amlnn_core_mask_config;

/* Base APIs */
/*  amlnn_init

    initial the context and load the model.

    input:
        void** context                  the double pointer of context handle.
        void* model                     if size > 0, it indicates the model's memory address, where size represents the memory size.
                                        if size = 0, it indicates the model's path.
                                        if the selected backend is not the ADLA NPU, only support model's path.
        uint32_t size                   the size of model.
        amlnn_init_config* init_config  the init configuration.
    return:
        int                             error code.
*/
int amlnn_init(void** context, void* model, uint32_t size, amlnn_init_config* init_config);

/*  amlnn_query

    query the information about model or others. see amlnn_query_cmd.

    input:
        void* context                   the pointer of context handle.
        amlnn_query_cmd query_cmd       the command of query.
        void* info                      the buffer point of information.
        uint32_t size                   the size of information.
    return:
        int                             error code.
*/
int amlnn_query(void* context, amlnn_query_cmd query_cmd, void* info, uint32_t size);

/*  amlnn_inputs_set

    set inputs information by input index of model.
    inputs information see amlnn_input.

    input:
        void* context                   the pointer of context handle.
        uint32_t n_inputs               the number of inputs.
        amlnn_input inputs[]            the arrays of inputs information, see amlnn_input.
    return:
        int                             error code.
*/
int amlnn_inputs_set(void* context, uint32_t n_inputs, amlnn_input inputs[]);

/*  amlnn_run

    run the model.

    input:
        void* context                   the pointer of context handle.
        amlnn_run_config* run_config    the run configuration (can be NULL).
    return:
        int                             error code.
*/
int amlnn_run(void* context, amlnn_run_config* run_config);

/*  amlnn_outputs_get

    get outputs information by output index of model.
    outputs information see amlnn_output.

    input:
        void* context                   the pointer of context handle.
        uint32_t n_outputs              the number of outputs.
        amlnn_output outputs[]          the arrays of outputs information, see amlnn_output.
    return:
        int                             error code.
*/
int amlnn_outputs_get(void* context, uint32_t n_outputs, amlnn_output outputs[]);

/*  amlnn_destroy

    unload the model and destroy the context.

    input:
        void* context                   the pointer of context handle.
    return:
        int                             error code.
*/
int amlnn_destroy(void* context);

/* DMA APIs */
/*  amlnn_alloc_dma_mem

    allocate DMA memory. (Only supports ADLA NPU)

    [IMPORTANT USAGE RULES]:
        1. Virtual Address Mode: Must use this function for allocation.
        Custom user-allocated virtual addresses are NOT supported.
        2. Allocation Size: The 'size' field in amlnn_dma_memory MUST be set based on
        'size_with_stride' from amlnn_tensor_attr to avoid hardware out-of-bounds access.
        3. Logical Usage: While the allocated size follows 'size_with_stride',
        data processing should only operate within the 'size' range.

    input:
        void* context                   the pointer of context handle.
        amlnn_dma_memory* dma_memory    the DMA memory.
        amlnn_dma_config* dma_config    the DMA configuration.
    return:
        int                             error code.
*/
int amlnn_alloc_dma_mem(void* context, amlnn_dma_memory* dma_memory, amlnn_dma_config* dma_config);

/*  amlnn_bind_dma_mem

    bind DMA memory to input or output. (Only supports ADLA NPU)

    input:
        void* context                   the pointer of context handle.
        amlnn_dma_memory* dma_memory    the DMA memory.
        amlnn_dma_config* dma_config    the DMA configuration.
    return:
        int                             error code.
*/
int amlnn_bind_dma_mem(void* context, amlnn_dma_memory* dma_memory, amlnn_dma_config* dma_config);

/*  amlnn_unbind_dma_mem

    unbind all DMA memory. (Only supports ADLA NPU)

    input:
        void* context                   the pointer of context handle.
    return:
        int                             error code.
*/
int amlnn_unbind_dma_mem(void* context);

/*  amlnn_free_dma_mem

    free DMA memory. (Only supports ADLA NPU)

    input:
        void* context                   the pointer of context handle.
        amlnn_dma_memory* dma_memory    the DMA memory.
        amlnn_dma_config* dma_config    the DMA configuration.
    return:
        int                             error code.
*/
int amlnn_free_dma_mem(void* context, amlnn_dma_memory* dma_memory, amlnn_dma_config* dma_config);

/* Software operator acceleration APIs */
/*  amlnn_set_softop_opt

    set software operators acceleration options. (Only supports ADLA NPU)

    input:
        void* context                       the pointer of context handle.
        amlnn_softop_opt_request requests[] the software operator request array.
        uint32_t request_num                the number of requests.
    return:
        int                                 error code.
*/
int amlnn_set_softop_opt(void* context, amlnn_softop_opt_request requests[], uint32_t request_num);

/* Dynamic shape APIs*/
/*  amlnn_set_input_shapes

    set input shapes for dynamic model.

    input:
        void* context                   the pointer of context handle.
        uint32_t n_inputs               the number of inputs.
        amlnn_tensor_attr tensor_attr[] the arrays of tensor attributes.
    return:
        int                             error code.
*/
int amlnn_set_input_shapes(void* context, uint32_t n_inputs, amlnn_tensor_attr tensor_attr[]);

/*  amlnn_set_core_mask

    set NPU core mask for current context. (Only supports ADLA NPU)

    input:
        void* context                               the pointer of context handle.
        amlnn_core_mask_config* core_mask_config    the NPU core mask configuration.
    return:
        int                                         error code.
*/
int amlnn_set_core_mask(void* context, amlnn_core_mask_config* core_mask_config);

#ifdef __cplusplus
} //extern "C"
#endif

#endif // _NNSDK2_H
