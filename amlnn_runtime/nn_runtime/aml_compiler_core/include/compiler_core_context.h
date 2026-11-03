/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * @file    compiler_core_context.h
 * @module  aml_compiler_core
 * @brief   Public C API: opaque Context owns Graph / Node / Tensor.
 *
 * Third parties fill the graph only through functions. No public layout for
 * AML_Model / AML_Node / AML_Tensor, and no AmlTensorDesc snapshot struct.
 *
 * Handles below are incomplete types (pointers only). Callers must not
 * dereference them. Dtype / op / padding enums below are the public schema
 * (numeric values match TFLite), not memory layout.
 *
 * Flow:
 *   ctx   = AmlContextCreate()
 *   graph = AmlGraphCreate(ctx)
 *   t     = AmlTensorCreate(graph); AmlTensorSet*(t, ...)
 *   n     = AmlNodeCreate(graph, op, index); AmlNodeAddInput(n, t); ...
 *   // After every graph / node / tensor is wired:
 *   AmlContextPrint(ctx);   // optional manual IR dump
 *   // or set AmlCompilerExternalOptions::dump_ir = 1 before compile.
 *   AmlContextCompile(ctx, name, path, options, &artifact)
 *   AmlContextDestroy(ctx)   // frees graphs, nodes, tensors
 ******************************************************************************/

#ifndef AML_NPU_COMPILER_CORE_COMPILER_CORE_CONTEXT_H_
#define AML_NPU_COMPILER_CORE_COMPILER_CORE_CONTEXT_H_

#include "compiler_core_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tensor dtypes; numeric values match internal AML_TensorType / TfLiteType.
 */
typedef enum AmlTensorType {
  kAmlTensorTypeNoType = 0,
  kAmlTensorTypeFloat32 = 1,
  kAmlTensorTypeInt32 = 2,
  kAmlTensorTypeUInt8 = 3,
  kAmlTensorTypeInt64 = 4,
  kAmlTensorTypeString = 5,
  kAmlTensorTypeBool = 6,
  kAmlTensorTypeInt16 = 7,
  kAmlTensorTypeComplex64 = 8,
  kAmlTensorTypeInt8 = 9,
  kAmlTensorTypeFloat16 = 10,
  kAmlTensorTypeFloat64 = 11,
  kAmlTensorTypeComplex128 = 12,
  kAmlTensorTypeUInt64 = 13,
  kAmlTensorTypeResource = 14,
  kAmlTensorTypeVariant = 15,
  kAmlTensorTypeUInt32 = 16,
  kAmlTensorTypeUInt16 = 17,
  kAmlTensorTypeInt4 = 18,
  kAmlTensorTypeBFloat16 = 19,
  kAmlTensorTypeInt2 = 20,
  kAmlTensorTypeUInt4 = 21
} AmlTensorType;

/** @brief Fused activations; values match AML_FusedActivation / TfLiteFusedActivation. */
typedef enum AmlFusedActivation {
  kAmlFusedActivationNone = 0,
  kAmlFusedActivationRelu = 1,
  kAmlFusedActivationReluN1To1 = 2,
  kAmlFusedActivationRelu6 = 3,
  kAmlFusedActivationTanh = 4,
  kAmlFusedActivationSignBit = 5,
  kAmlFusedActivationSigmoid = 6
} AmlFusedActivation;

/** @brief Padding for node param "padding"; schema values (SAME=0, VALID=1). */
typedef enum AmlPadding {
  kAmlPaddingSame = 0,
  kAmlPaddingValid = 1
} AmlPadding;

/**
 * @brief Operator types; values match AML_OperationType / TFLite BuiltinOperator.
 */
typedef enum AmlOperationType {
  kAmlOpAdd = 0,
  kAmlOpAveragePool2D = 1,
  kAmlOpConcatenation = 2,
  kAmlOpConv2D = 3,
  kAmlOpDepthwiseConv2D = 4,
  kAmlOpDepthToSpace = 5,
  kAmlOpDequantize = 6,
  kAmlOpEmbeddingLookup = 7,
  kAmlOpFloor = 8,
  kAmlOpFullyConnected = 9,
  kAmlOpHashtableLookup = 10,
  kAmlOpL2Normalization = 11,
  kAmlOpL2Pool2D = 12,
  kAmlOpLocalResponseNormalization = 13,
  kAmlOpLogistic = 14,
  kAmlOpLshProjection = 15,
  kAmlOpLstm = 16,
  kAmlOpMaxPool2D = 17,
  kAmlOpMul = 18,
  kAmlOpRelu = 19,
  kAmlOpReluN1To1 = 20,
  kAmlOpRelu6 = 21,
  kAmlOpReshape = 22,
  kAmlOpResizeBilinear = 23,
  kAmlOpRnn = 24,
  kAmlOpSoftmax = 25,
  kAmlOpSpaceToDepth = 26,
  kAmlOpSvdf = 27,
  kAmlOpTanh = 28,
  kAmlOpConcatEmbeddings = 29,
  kAmlOpSkipGram = 30,
  kAmlOpCall = 31,
  kAmlOpCustom = 32,
  kAmlOpEmbeddingLookupSparse = 33,
  kAmlOpPad = 34,
  kAmlOpUnidirectionalSequenceRnn = 35,
  kAmlOpGather = 36,
  kAmlOpBatchToSpaceNd = 37,
  kAmlOpSpaceToBatchNd = 38,
  kAmlOpTranspose = 39,
  kAmlOpMean = 40,
  kAmlOpSub = 41,
  kAmlOpDiv = 42,
  kAmlOpSqueeze = 43,
  kAmlOpUnidirectionalSequenceLstm = 44,
  kAmlOpStridedSlice = 45,
  kAmlOpBidirectionalSequenceRnn = 46,
  kAmlOpExp = 47,
  kAmlOpTopkV2 = 48,
  kAmlOpSplit = 49,
  kAmlOpLogSoftmax = 50,
  kAmlOpDelegate = 51,
  kAmlOpBidirectionalSequenceLstm = 52,
  kAmlOpCast = 53,
  kAmlOpPrelu = 54,
  kAmlOpMaximum = 55,
  kAmlOpArgMax = 56,
  kAmlOpMinimum = 57,
  kAmlOpLess = 58,
  kAmlOpNeg = 59,
  kAmlOpPadV2 = 60,
  kAmlOpGreater = 61,
  kAmlOpGreaterEqual = 62,
  kAmlOpLessEqual = 63,
  kAmlOpSelect = 64,
  kAmlOpSlice = 65,
  kAmlOpSin = 66,
  kAmlOpTransposeConv = 67,
  kAmlOpSparseToDense = 68,
  kAmlOpTile = 69,
  kAmlOpExpandDims = 70,
  kAmlOpEqual = 71,
  kAmlOpNotEqual = 72,
  kAmlOpLog = 73,
  kAmlOpSum = 74,
  kAmlOpSqrt = 75,
  kAmlOpRsqrt = 76,
  kAmlOpShape = 77,
  kAmlOpPow = 78,
  kAmlOpArgMin = 79,
  kAmlOpFakeQuant = 80,
  kAmlOpReduceProd = 81,
  kAmlOpReduceMax = 82,
  kAmlOpPack = 83,
  kAmlOpLogicalOr = 84,
  kAmlOpOneHot = 85,
  kAmlOpLogicalAnd = 86,
  kAmlOpLogicalNot = 87,
  kAmlOpUnpack = 88,
  kAmlOpReduceMin = 89,
  kAmlOpFloorDiv = 90,
  kAmlOpReduceAny = 91,
  kAmlOpSquare = 92,
  kAmlOpZerosLike = 93,
  kAmlOpFill = 94,
  kAmlOpFloorMod = 95,
  kAmlOpRange = 96,
  kAmlOpResizeNearestNeighbor = 97,
  kAmlOpLeakyRelu = 98,
  kAmlOpSquaredDifference = 99,
  kAmlOpMirrorPad = 100,
  kAmlOpAbs = 101,
  kAmlOpSplitV = 102,
  kAmlOpUnique = 103,
  kAmlOpCeil = 104,
  kAmlOpReverseV2 = 105,
  kAmlOpAddN = 106,
  kAmlOpGatherNd = 107,
  kAmlOpCos = 108,
  kAmlOpWhere = 109,
  kAmlOpRank = 110,
  kAmlOpElu = 111,
  kAmlOpReverseSequence = 112,
  kAmlOpMatrixDiag = 113,
  kAmlOpQuantize = 114,
  kAmlOpMatrixSetDiag = 115,
  kAmlOpRound = 116,
  kAmlOpHardSwish = 117,
  kAmlOpIf = 118,
  kAmlOpWhile = 119,
  kAmlOpNonMaxSuppressionV4 = 120,
  kAmlOpNonMaxSuppressionV5 = 121,
  kAmlOpScatterNd = 122,
  kAmlOpSelectV2 = 123,
  kAmlOpDensify = 124,
  kAmlOpSegmentSum = 125,
  kAmlOpBatchMatmul = 126,
  kAmlOpPlaceholderForGreaterOpCodes = 127,
  kAmlOpCumsum = 128,
  kAmlOpCallOnce = 129,
  kAmlOpBroadcastTo = 130,
  kAmlOpRfft2D = 131,
  kAmlOpConv3D = 132,
  kAmlOpImag = 133,
  kAmlOpReal = 134,
  kAmlOpComplexAbs = 135,
  kAmlOpHashtable = 136,
  kAmlOpHashtableFind = 137,
  kAmlOpHashtableImport = 138,
  kAmlOpHashtableSize = 139,
  kAmlOpReduceAll = 140,
  kAmlOpConv3DTranspose = 141,
  kAmlOpVarHandle = 142,
  kAmlOpReadVariable = 143,
  kAmlOpAssignVariable = 144,
  kAmlOpBroadcastArgs = 145,
  kAmlOpRandomStandardNormal = 146,
  kAmlOpBucketize = 147,
  kAmlOpRandomUniform = 148,
  kAmlOpMultinomial = 149,
  kAmlOpGelu = 150,
  kAmlOpDynamicUpdateSlice = 151,
  kAmlOpRelu0To1 = 152,
  kAmlOpUnsortedSegmentProd = 153,
  kAmlOpUnsortedSegmentMax = 154,
  kAmlOpUnsortedSegmentSum = 155,
  kAmlOpAtan2 = 156,
  kAmlOpUnsortedSegmentMin = 157,
  kAmlOpSign = 158
} AmlOperationType;

/**
 * @brief String keys for AmlNodeSetParamI / AmlNodeSetParamF.
 *
 * A dash means the converter does not read that node’s params / fparams.
 * Keep in sync with tflite_to_convert.cc (and context_binary_info.h).
 * Public copy under LiteRT is refreshed by ./install_to_litert.sh.
 *
 * Value conventions:
 * - activation: AmlFusedActivation numeric value (AmlNodeSetParamI).
 * - padding: AmlPadding numeric value (AmlNodeSetParamI).
 * - pot_scale_int16, keep_num_dims, keep_dims, align_corners,
 *   half_pixel_centers, offset, adj_x, adj_y, asymmetric_quantize_inputs,
 *   exclusive, reverse, approximate and time_major: boolean encoded as 0 or 1
 *   (AmlNodeSetParamI).
 * - mode: TFLite MirrorPadMode (REFLECT=0, SYMMETRIC=1).
 * - stride_*, dilation_*, filter_*, axis, batch_dims, multiplier, block_size,
 *   masks, num_split, values_count and num: integer (AmlNodeSetParamI).
 * - alpha, beta, cell_clip and proj_clip: float (AmlNodeSetParamF).
 *
 * @code
 * Operator                         params strings                         fparams strings
 * ADD                              activation, pot_scale_int16            -
 * AVERAGE_POOL_2D                  activation, filter_h, filter_w,         -
 *                                  padding, stride_h, stride_w
 * CONCATENATION                    activation, axis                       -
 * CONV_2D                          activation, dilation_h, dilation_w,     -
 *                                  padding, stride_h, stride_w
 * DEPTHWISE_CONV_2D                activation, dilation_h, dilation_w,     -
 *                                  multiplier, padding, stride_h, stride_w
 * DEPTH_TO_SPACE                   block_size                             -
 * DEQUANTIZE                       -                                      -
 * EMBEDDING_LOOKUP                 -                                      -
 * FLOOR                            -                                      -
 * FULLY_CONNECTED                  activation, keep_num_dims              -
 * HASHTABLE_LOOKUP                 -                                      -
 * L2_NORMALIZATION                 activation                             -
 * L2_POOL_2D                       -                                      -
 * LOCAL_RESPONSE_NORMALIZATION     -                                      -
 * LOGISTIC                         -                                      -
 * LSH_PROJECTION                   -                                      -
 * LSTM                             -                                      -
 * MAX_POOL_2D                      activation, filter_h, filter_w,         -
 *                                  padding, stride_h, stride_w
 * MUL                              activation                             -
 * RELU                             -                                      -
 * RELU_N1_TO_1                     -                                      -
 * RELU6                            -                                      -
 * RESHAPE                          -                                      -
 * RESIZE_BILINEAR                  align_corners, half_pixel_centers       -
 * RNN                              -                                      -
 * SOFTMAX                          -                                      beta
 * SPACE_TO_DEPTH                   block_size                             -
 * SVDF                             rank, activation                       -
 * TANH                             -                                      -
 * CONCAT_EMBEDDINGS                -                                      -
 * SKIP_GRAM                        -                                      -
 * CALL                             -                                      -
 * CUSTOM                           -                                      -
 * EMBEDDING_LOOKUP_SPARSE          -                                      -
 * PAD                              -                                      -
 * UNIDIRECTIONAL_SEQUENCE_RNN      -                                      -
 * GATHER                           axis, batch_dims                       -
 * BATCH_TO_SPACE_ND                -                                      -
 * SPACE_TO_BATCH_ND                -                                      -
 * TRANSPOSE                        -                                      -
 * MEAN                             keep_dims                              -
 * SUB                              activation, pot_scale_int16            -
 * DIV                              activation                             -
 * SQUEEZE                          -                                      -
 * UNIDIRECTIONAL_SEQUENCE_LSTM     activation, asymmetric_quantize_inputs, cell_clip, proj_clip
 *                                  time_major
 * STRIDED_SLICE                    begin_mask, ellipsis_mask, end_mask,    -
 *                                  new_axis_mask, offset, shrink_axis_mask
 * BIDIRECTIONAL_SEQUENCE_RNN       -                                      -
 * EXP                              -                                      -
 * TOPK_V2                          -                                      -
 * SPLIT                            num_split                              -
 * LOG_SOFTMAX                      -                                      -
 * DELEGATE                         -                                      -
 * BIDIRECTIONAL_SEQUENCE_LSTM      -                                      -
 * CAST                             -                                      -
 * PRELU                            -                                      -
 * MAXIMUM                          -                                      -
 * ARGMAX                           -                                      -
 * MINIMUM                          -                                      -
 * LESS                             -                                      -
 * NEG                              -                                      -
 * PADV2                            -                                      -
 * GREATER                          -                                      -
 * GREATER_EQUAL                    -                                      -
 * LESS_EQUAL                       -                                      -
 * SELECT                           -                                      -
 * SLICE                            -                                      -
 * SIN                              -                                      -
 * TRANSPOSE_CONV                   activation, padding, stride_h, stride_w -
 * SPARSE_TO_DENSE                  -                                      -
 * TILE                             -                                      -
 * EXPAND_DIMS                      -                                      -
 * EQUAL                            -                                      -
 * NOT_EQUAL                        -                                      -
 * LOG                              -                                      -
 * SUM                              keep_dims                              -
 * SQRT                             -                                      -
 * RSQRT                            -                                      -
 * SHAPE                            -                                      -
 * POW                              -                                      -
 * ARG_MIN                          -                                      -
 * FAKE_QUANT                       -                                      -
 * REDUCE_PROD                      -                                      -
 * REDUCE_MAX                       keep_dims                              -
 * PACK                             axis, values_count                     -
 * LOGICAL_OR                       -                                      -
 * ONE_HOT                          -                                      -
 * LOGICAL_AND                      -                                      -
 * LOGICAL_NOT                      -                                      -
 * UNPACK                           axis, num                              -
 * REDUCE_MIN                       keep_dims                              -
 * FLOOR_DIV                        -                                      -
 * REDUCE_ANY                       keep_dims                              -
 * SQUARE                           -                                      -
 * ZEROS_LIKE                       -                                      -
 * FILL                             -                                      -
 * FLOOR_MOD                        -                                      -
 * RANGE                            -                                      -
 * RESIZE_NEAREST_NEIGHBOR          align_corners, half_pixel_centers       -
 * LEAKY_RELU                       -                                      alpha
 * SQUARED_DIFFERENCE               -                                      -
 * MIRROR_PAD                       mode                                   -
 * ABS                              -                                      -
 * SPLIT_V                          -                                      -
 * UNIQUE                           -                                      -
 * CEIL                             -                                      -
 * REVERSE_V2                       -                                      -
 * ADD_N                            -                                      -
 * GATHER_ND                        -                                      -
 * COS                              -                                      -
 * WHERE                            -                                      -
 * RANK                             -                                      -
 * ELU                              -                                      -
 * REVERSE_SEQUENCE                 -                                      -
 * MATRIX_DIAG                      -                                      -
 * QUANTIZE                         -                                      -
 * MATRIX_SET_DIAG                  -                                      -
 * ROUND                            -                                      -
 * HARD_SWISH                       -                                      -
 * IF                               -                                      -
 * WHILE                            -                                      -
 * NON_MAX_SUPPRESSION_V4           -                                      -
 * NON_MAX_SUPPRESSION_V5           -                                      -
 * SCATTER_ND                       -                                      -
 * SELECT_V2                        -                                      -
 * DENSIFY                          -                                      -
 * SEGMENT_SUM                      -                                      -
 * BATCH_MATMUL                     adj_x, adj_y, asymmetric_quantize_inputs -
 * PLACEHOLDER_FOR_GREATER_OP_CODES -                                      -
 * CUMSUM                           exclusive, reverse                     -
 * CALL_ONCE                        -                                      -
 * BROADCAST_TO                     -                                      -
 * RFFT2D                           -                                      -
 * CONV_3D                          -                                      -
 * IMAG                             -                                      -
 * REAL                             -                                      -
 * COMPLEX_ABS                      -                                      -
 * HASHTABLE                        -                                      -
 * HASHTABLE_FIND                   -                                      -
 * HASHTABLE_IMPORT                 -                                      -
 * HASHTABLE_SIZE                   -                                      -
 * REDUCE_ALL                       keep_dims                              -
 * CONV_3D_TRANSPOSE                -                                      -
 * VAR_HANDLE                       -                                      -
 * READ_VARIABLE                    -                                      -
 * ASSIGN_VARIABLE                  -                                      -
 * BROADCAST_ARGS                   -                                      -
 * RANDOM_STANDARD_NORMAL           -                                      -
 * BUCKETIZE                        -                                      -
 * RANDOM_UNIFORM                   -                                      -
 * MULTINOMIAL                      -                                      -
 * GELU                             approximate                            -
 * DYNAMIC_UPDATE_SLICE             -                                      -
 * RELU_0_TO_1                      -                                      -
 * UNSORTED_SEGMENT_PROD            -                                      -
 * UNSORTED_SEGMENT_MAX             -                                      -
 * UNSORTED_SEGMENT_SUM             -                                      -
 * ATAN2                            -                                      -
 * UNSORTED_SEGMENT_MIN             -                                      -
 * SIGN                             -                                      -
 * @endcode
 */

/**
 * @brief Session that owns every graph / node / tensor created under it.
 * @note Destroy with AmlContextDestroy(). Invalidates all child handles.
 */
typedef struct AmlContextT* AmlContext;

/**
 * @brief One partition / subgraph. Owned by @ref AmlContext.
 * @note Maps internally to one AML_Model.
 */
typedef struct AmlGraphT* AmlGraph;

/**
 * @brief Operator node. Owned by its @ref AmlGraph.
 * @note Maps internally to one AML_Node.
 */
typedef struct AmlNodeT* AmlNode;

/**
 * @brief Tensor. Owned by its @ref AmlGraph.
 * @note Maps internally to one AML_Tensor. The same handle can be a graph
 *       I/O and a node I/O (identity is the handle, not a copied snapshot).
 */
typedef struct AmlTensorT* AmlTensor;

/* -------------------------------------------------------------------------- */
/* Context                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create an empty compilation session.
 * @return New context handle, or NULL on allocation failure.
 * @note Pair with AmlContextDestroy(). All graphs created under this context
 *       are owned by it.
 */
AML_COMPILER_API AmlContext AmlContextCreate(void);

/**
 * @brief Destroy a context and every graph / node / tensor it owns.
 * @param ctx Context from AmlContextCreate(). No-op if NULL.
 * @note Invalidates all handles obtained from this context.
 */
AML_COMPILER_API void AmlContextDestroy(AmlContext ctx);

/**
 * @brief Compile every graph in @p ctx (order of AmlGraphCreate).
 * @param ctx                 Session that owns the graph(s) to compile.
 * @param model_name          Base name for outputs / logging (non-NULL).
 * @param model_path          Directory or path prefix for artifacts (non-NULL).
 * @param options             Compile options; may be NULL for defaults.
 * @param compiled_artifact   Out: compiled result; caller frees via
 *                            AmlCompilerCoreDestroyCompiledArtifact().
 * @return kAmlCompilerStatusOk on success.
 * @note Call only after all graphs are fully filled (tensors created, nodes
 *       wired, graph I/O registered). Internally materializes each graph into
 *       AML_Model then runs convert.
 * @note When @c options->external_options points to @ref AmlCompilerExternalOptions
 *       with @c dump_ir != 0, dumps every graph via AmlContextPrint() immediately
 *       before materialize (same output as a manual AmlContextPrint() call).
 *       Set @c log_level to @c ADLA_LOG_LEVEL_INFO (4) or lower for output.
 */
AML_COMPILER_API AmlCompilerStatus AmlContextCompile(
    AmlContext ctx, const char* model_name, const char* model_path,
    const AmlCompilerCoreOptions* options,
    AmlCompilerCoreCompiledArtifact* compiled_artifact);

/**
 * @brief Dump every graph in @p ctx to log (node/tensor metadata only).
 * @param ctx Context to inspect. No-op if NULL.
 * @note Call after the graph is fully filled (all AmlTensorSet*, AmlNodeCreate,
 *       AmlNodeAddInput/Output, AmlGraphAddInput/Output complete) and before
 *       AmlContextCompile(). Does not materialize or dump weight bytes.
 * @note Alternatively set @ref AmlCompilerExternalOptions::dump_ir before
 *       AmlContextCompile() to dump automatically at compile time.
 */
AML_COMPILER_API void AmlContextPrint(AmlContext ctx);

/* -------------------------------------------------------------------------- */
/* Graph                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create a new subgraph / partition under @p ctx.
 * @param ctx   Owning context (non-NULL).
 * @param graph Out: new graph handle.
 * @return kAmlCompilerStatusOk on success.
 */
AML_COMPILER_API AmlCompilerStatus AmlGraphCreate(AmlContext ctx,
                                                  AmlGraph* graph);

/**
 * @brief Set the ADLA compile target hint for this graph.
 * @param graph  Target graph (non-NULL).
 * @param target Target string, e.g. platform or SoC id (non-NULL).
 */
AML_COMPILER_API AmlCompilerStatus AmlGraphSetTarget(AmlGraph graph,
                                                     const char* target);

/**
 * @brief Set the graph / subgraph display name.
 * @param graph Graph handle (non-NULL).
 * @param name  Name string (non-NULL). Replaces any previous name.
 */
AML_COMPILER_API AmlCompilerStatus AmlGraphSetName(AmlGraph graph,
                                                   const char* name);

/**
 * @brief Append an entry to the optional op-code name table.
 * @param graph    Graph handle (non-NULL).
 * @param op_code  Op-code label string (non-NULL).
 */
AML_COMPILER_API AmlCompilerStatus AmlGraphAddOpCode(AmlGraph graph,
                                                     const char* op_code);

/**
 * @brief Register a tensor as a graph-level input.
 * @param graph  Graph handle (non-NULL).
 * @param tensor Tensor on the same graph (non-NULL).
 * @return kAmlCompilerStatusErrorInvalidArgument if @p tensor belongs to
 *         another graph.
 */
AML_COMPILER_API AmlCompilerStatus AmlGraphAddInput(AmlGraph graph,
                                                    AmlTensor tensor);

/**
 * @brief Register a tensor as a graph-level output.
 * @param graph  Graph handle (non-NULL).
 * @param tensor Tensor on the same graph (non-NULL).
 */
AML_COMPILER_API AmlCompilerStatus AmlGraphAddOutput(AmlGraph graph,
                                                     AmlTensor tensor);

/**
 * @brief Dump one filled graph to log (node/tensor metadata only).
 * @param graph Graph to inspect. No-op if NULL.
 * @note Does not materialize or dump weight bytes.
 */
AML_COMPILER_API void AmlGraphPrint(AmlGraph graph);

/* -------------------------------------------------------------------------- */
/* Tensor  (create empty, then fill with setters; library copies payloads)     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create an empty tensor on @p graph.
 * @param graph  Owning graph (non-NULL).
 * @param tensor Out: new tensor handle. Fields are unset until AmlTensorSet*.
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorCreate(AmlGraph graph,
                                                   AmlTensor* tensor);

/**
 * @brief Set the tensor name.
 * @param name NUL-terminated string; copied into internal storage.
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorSetName(AmlTensor tensor,
                                                    const char* name);

/**
 * @brief Set tensor element type.
 * @param dtype AmlTensorType numeric value (or compatible TfLite type id).
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorSetDtype(AmlTensor tensor,
                                                     int dtype);

/**
 * @brief Set the original tensor index from the source model.
 * @param tensor_index Index in the imported / delegate graph.
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorSetIndex(AmlTensor tensor,
                                                     int tensor_index);

/**
 * @brief Set tensor shape.
 * @param dims        Dimension array (may be NULL if @p dims_count == 0).
 * @param dims_count  Number of dimensions.
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorSetDims(AmlTensor tensor,
                                                    const int* dims,
                                                    uint32_t dims_count);

/**
 * @brief Set per-tensor or per-channel quantization parameters.
 * @param zp          Zero-point array (may be NULL if @p zp_count == 0).
 * @param zp_count    Length of @p zp.
 * @param scale       Scale array (may be NULL if @p scale_count == 0).
 * @param scale_count Length of @p scale.
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorSetQuant(AmlTensor tensor,
                                                     const int* zp,
                                                     uint32_t zp_count,
                                                     const float* scale,
                                                     uint32_t scale_count);

/**
 * @brief Copy const / weight bytes into the tensor (is_const = 1).
 * @param data       May be NULL iff @p data_size == 0.
 * @param data_size  Byte size. Pointer need only live for this call.
 */
AML_COMPILER_API AmlCompilerStatus AmlTensorSetConstData(AmlTensor tensor,
                                                         const void* data,
                                                         size_t data_size);

/* -------------------------------------------------------------------------- */
/* Node                                                                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create an operator node on @p graph.
 * @param graph       Owning graph (non-NULL).
 * @param op_type     AmlOperationType numeric value.
 * @param node_index  Node index within the partition / subgraph.
 * @param node        Out: new node handle. Wire I/O with AmlNodeAddInput/Output.
 */
AML_COMPILER_API AmlCompilerStatus AmlNodeCreate(AmlGraph graph, int op_type,
                                                 int node_index,
                                                 AmlNode* node);

/**
 * @brief Append an input tensor to @p node.
 * @param tensor Same graph as @p node, or NULL for an absent optional input.
 *               NULL keeps the slot so later inputs stay at their TFLite indices.
 */
AML_COMPILER_API AmlCompilerStatus AmlNodeAddInput(AmlNode node,
                                                   AmlTensor tensor);

/**
 * @brief Append an output tensor to @p node.
 * @param tensor Must belong to the same graph as @p node.
 */
AML_COMPILER_API AmlCompilerStatus AmlNodeAddOutput(AmlNode node,
                                                    AmlTensor tensor);

/**
 * @brief Append an intermediate (scratch) tensor to @p node.
 * @param tensor Same graph as @p node, or NULL for an absent slot.
 *               Order matches TFLite Operator.intermediates.
 */
AML_COMPILER_API AmlCompilerStatus AmlNodeAddIntermediate(AmlNode node,
                                                          AmlTensor tensor);

/**
 * @brief Set an integer node attribute.
 * @param key   Param name; see operator table above (e.g. "padding", "axis").
 * @param value Integer value (AmlPadding, AmlFusedActivation, or 0/1 bool).
 */
AML_COMPILER_API AmlCompilerStatus AmlNodeSetParamI(AmlNode node,
                                                    const char* key, int value);

/**
 * @brief Set a floating-point node attribute.
 * @param key   Param name (e.g. "alpha", "beta").
 * @param value Float value.
 */
AML_COMPILER_API AmlCompilerStatus AmlNodeSetParamF(AmlNode node,
                                                    const char* key,
                                                    float value);

#ifdef __cplusplus
}
#endif

#endif  // AML_NPU_COMPILER_CORE_COMPILER_CORE_CONTEXT_H_
