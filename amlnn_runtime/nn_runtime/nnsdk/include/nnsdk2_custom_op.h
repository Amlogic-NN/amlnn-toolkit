/*
* Copyright (C) 2026 Amlogic, Inc. All rights reserved.
*
* This source code is subject to the terms and conditions defined in the
* file 'LICENSE' which is part of this source code package.
*
* Description: aml_nnsdk2_custom_operator
*/

#ifndef _NNSDK2_CUSTOM_OP_H
#define _NNSDK2_CUSTOM_OP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nnsdk2.h"

#include <stdint.h>
#include <stddef.h>

/*
    The data type of a custom operator option.
    The value passed to amlnn_custom_op_get_option must match the selected type.
*/
typedef enum _amlnn_custom_op_option_type
{
    AMLNN_CUSTOM_OP_OPTION_FLOAT32 = 0,       /* 32-bit floating-point option. */
    AMLNN_CUSTOM_OP_OPTION_FLOAT64,           /* 64-bit floating-point option. */
    AMLNN_CUSTOM_OP_OPTION_INT8,              /* signed 8-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_INT16,             /* signed 16-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_INT32,             /* signed 32-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_INT64,             /* signed 64-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_UINT8,             /* unsigned 8-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_UINT16,            /* unsigned 16-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_UINT32,            /* unsigned 32-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_UINT64,            /* unsigned 64-bit integer option. */
    AMLNN_CUSTOM_OP_OPTION_BOOL,              /* boolean option (stored as int32_t). */
    AMLNN_CUSTOM_OP_OPTION_STRING,            /* string option. */
    AMLNN_CUSTOM_OP_OPTION_BLOB               /* binary blob option. */
} amlnn_custom_op_option_type;

/*
    The arguments passed to a custom operator callback. Tensor attributes use
    the same format as amlnn_query(). All pointers are valid only during the
    callback.
*/
typedef struct _amlnn_custom_op_callback_args
{
    const uint8_t *options;                         /* serialized operator options. */
    size_t options_size;                            /* options size in bytes. */
    int32_t num_inputs;                             /* number of input tensors. */
    int32_t num_outputs;                            /* number of output tensors. */
    const amlnn_tensor_attr *input_descs;           /* input tensor attributes. */
    const amlnn_tensor_attr *output_descs;          /* output tensor attributes. */
    const void *const *input_data;                  /* input tensor data pointers. */
    void *const *output_data;                       /* output tensor data pointers. */
    void *user_data;                                /* user data from registration. */
    uint8_t reserved[64];                           /* reserved for future use. */
} amlnn_custom_op_callback_args;

/*
    The custom operator callback function. The callback reads input_data,
    writes output_data, and returns AMLNN_SUCCESS on success or an AMLNN error
    code on failure.
*/
typedef int (*amlnn_custom_op_callback)(const amlnn_custom_op_callback_args *args);

/*
 * Custom operator registration entry. custom_name must match the name in the model.
 * The array must remain valid until amlnn_destroy returns.
 */
typedef struct _amlnn_custom_op
{
    char custom_name[AMLNN_MAX_NAME_LEN];            /* custom operator name in the model. */
    amlnn_custom_op_callback callback;               /* custom operator implementation. */
    void *user_data;                                 /* opaque data passed to callback. */
} amlnn_custom_op;

/*
    Register custom operators for an initialized context. custom_ops must remain
    valid until amlnn_destroy() returns, and registration must be completed
    before amlnn_run().
*/
int amlnn_register_custom_ops(void *context, amlnn_custom_op custom_ops[], uint32_t custom_op_num);
/* Read one named option from the options buffer supplied to a callback. */
int amlnn_custom_op_get_option(const uint8_t *options, size_t options_size, const char *name, amlnn_custom_op_option_type type, void *value);

#ifdef __cplusplus
} //extern "C"
#endif

#endif // _NNSDK2_CUSTOM_OP_H
