/****************************************************************************
 *
 *    Copyright (C) 2026 Amlogic, Inc. All rights reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 ***************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

#include "nnsdk2.h"

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(amlnn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, size_with_stride=%d, fmt=%s, type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
           attr->zp, attr->scale);
}

static void *read_file(const char *file_path, uint32_t *file_size)
{
    FILE *fp = NULL;
    uint32_t size = 0;
    void *buf = NULL;
    fp = fopen(file_path, "rb");
    if (NULL == fp)
    {
        printf("open file fail!\n");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    rewind(fp);

    buf = malloc(sizeof(uint8_t) * size);

    fread(buf, 1, size, fp);

    fclose(fp);

    *file_size = size;
    return buf;
}

static void process_top5_f32(float *buf, uint32_t num)
{
    uint32_t i, j, k;
    uint32_t MaxClass[5] = {0};
    float fMaxProb[5] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

    for (i = 0; i < num; i++)
    {
        for (j = 0; j < 5; j++)
        {
            if (buf[i] > fMaxProb[j])
            {
                for (k = 4; k > j; k--)
                {
                    fMaxProb[k] = fMaxProb[k - 1];
                    MaxClass[k] = MaxClass[k - 1];
                }
                fMaxProb[j] = buf[i];
                MaxClass[j] = i;
                break;
            }
        }
    }

    printf(" --- Top5 ---\n");
    for (i = 0; i < 5; i++)
    {
        printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int base_api(int argc, char **argv)
{
    int ret = 0;

    const char *model_path = argv[1];
    const char *input_path = argv[2];

    if (argc != 3)
    {
        printf("Usage: %s <adla model_path> <input_path> \n", argv[0]);
        return -1;
    }

    // Get sdk version
    amlnn_sdk_version sdk_ver;
    ret = amlnn_query(NULL, AMLNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("version info: sdk2-[%s], adla-[%s], delegate-[%s]\n", sdk_ver.sdk2_api_version, sdk_ver.adla_kmd_version, sdk_ver.delegate_version);

    // Load ADLA Model
    void *ctx = NULL;
    amlnn_init_config init_config;
    memset(&init_config, 0, sizeof(amlnn_init_config));
    init_config.backend_type = AMLNN_BACKEND_ADLA_NPU;
    ret = amlnn_init(&ctx, (void *)model_path, 0, &init_config);
    if (ret < 0)
    {
        printf("amlnn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
    amlnn_input_output_num io_num;
    ret = amlnn_query(ctx, AMLNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != AMLNN_SUCCESS)
    {
        printf("amlnn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    amlnn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = amlnn_query(ctx, AMLNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(amlnn_tensor_attr));
        if (ret != AMLNN_SUCCESS)
        {
            printf("amlnn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    amlnn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = amlnn_query(ctx, AMLNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(amlnn_tensor_attr));
        if (ret != AMLNN_SUCCESS)
        {
            printf("amlnn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Load input data
    void *input_data;
    uint32_t input_size;
    uint32_t file_size;
    input_data = read_file(input_path, &file_size);
    input_size = input_attrs[0].n_elems * sizeof(uint8_t);

    // Set input
    printf("amlnn_inputs_set\n");
    amlnn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = input_data;
    inputs[0].size = input_size;
    ret = amlnn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("amlnn_inputs_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    printf("amlnn_run\n");
    ret = amlnn_run(ctx, NULL);
    if (ret < 0)
    {
        printf("amlnn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    printf("amlnn_outputs_get\n");
    amlnn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].is_float = 1;
    outputs[0].index = 0;
    ret = amlnn_outputs_get(ctx, 1, outputs);
    if (ret < 0)
    {
        printf("amlnn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // Post Process
    for (uint32_t i = 0; i < io_num.n_output; i++)
    {
        float *buffer = (float *)outputs[i].buf;
        uint32_t sz = outputs[i].size / sizeof(float);

        process_top5_f32(buffer, sz);
    }

    // Destroy
    ret = amlnn_destroy(ctx);
    if (ret < 0)
    {
        printf("amlnn_destroy fail! ret=%d\n", ret);
        return -1;
    }

    if (input_data)
    {
        free(input_data);
    }

    return 0;
}


int main(int argc, char **argv)
{
    int ret = 0;

    ret = base_api(argc, argv);

    return ret == 0 ? 0 : 1;
}
