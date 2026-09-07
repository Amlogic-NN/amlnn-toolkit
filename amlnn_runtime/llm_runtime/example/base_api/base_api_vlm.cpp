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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

#include "llmsdk.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void print_usage(const char* exe)
{
    printf("Usage:\n");
    printf("  %s --model_path <llm.adla> --mmproj_path <mmproj.adla> [--sampling_mode 0]\n", exe);
    printf("\n");
    printf("After model loading, input image path first, then prompt.\n");
    printf("Leave image path empty to run normal LLM chat.\n");
    printf("When an image is loaded, '<image>' in prompt controls image position.\n");
}

static const char* get_arg(int argc, char** argv, const char* name)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return NULL;
}

static void callback(AML_LLMResult* result, void* userdata, AML_LLMRunStatus status)
{
    (void)userdata;
    if (status == AML_LLM_RUN_NORMAL)
    {
        printf("%s", result->generation.text);
        fflush(stdout);
    }
    else if (status == AML_LLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (status == AML_LLM_RUN_ERROR)
    {
        printf("run error\n");
    }
}

static std::string trim_copy(const std::string& text)
{
    size_t begin = 0;
    while (begin < text.size() && isspace((unsigned char)text[begin])) ++begin;
    size_t end = text.size();
    while (end > begin && isspace((unsigned char)text[end - 1])) --end;
    return text.substr(begin, end - begin);
}

static bool load_rgb_image(const char* path, std::vector<uint8_t>& rgb,
                           uint32_t* width, uint32_t* height)
{
    int w = 0, h = 0, channels = 0;
    unsigned char* data = stbi_load(path, &w, &h, &channels, 3);
    if (!data || w <= 0 || h <= 0)
    {
        if (data) stbi_image_free(data);
        printf("stbi_load failed: %s\n", path);
        return false;
    }

    rgb.assign(data, data + (size_t)w * h * 3);
    stbi_image_free(data);
    *width = (uint32_t)w;
    *height = (uint32_t)h;
    printf("image loaded: %s, original=%dx%d\n", path, w, h);
    return true;
}

int main(int argc, char** argv)
{
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))
    {
        print_usage(argv[0]);
        return 0;
    }

    const char* model_path = get_arg(argc, argv, "--model_path");
    const char* mmproj_path = get_arg(argc, argv, "--mmproj_path");
    const char* sampling_text = get_arg(argc, argv, "--sampling_mode");
    int sampling_mode = sampling_text ? atoi(sampling_text) : AML_LLM_ARG_Max;
    if (!model_path || !mmproj_path)
    {
        print_usage(argv[0]);
        return -1;
    }

    AML_LLMInitConfig init_config;
    memset(&init_config, 0, sizeof(init_config));
    init_config.model_path = model_path;
    init_config.sampling_mode = (AML_LLMSamplingMode)sampling_mode;
    init_config.top_k = 3;
    init_config.top_p = 0.9f;
    init_config.temperature = 1.0f;
    init_config.repeat_penalty = 1.1f;
    init_config.init_extend.mmproj_path = mmproj_path;

    LLMContext context = NULL;
    if (aml_llm_init(&context, &init_config, callback) != AML_LLM_Status_Success)
    {
        printf("aml_llm_init failed\n");
        return -1;
    }

    AML_LLMRunConfig run_config;
    memset(&run_config, 0, sizeof(run_config));
    run_config.run_mode = AML_LLM_RUN_GENERATE;

    const char* img_content = "<image>";
    const char* img_start = "<|vision_start|>";
    const char* img_end = "<|vision_end|>";
    // const char* img_start = "<img>";
    // const char* img_end = "</img>";

    printf("\nVLM model loaded. Use /image <path> to append images, then enter a prompt.\n");
    printf("Leave image path empty, or use an invalid path, to run normal LLM chat.\n");
    printf("Commands: exit, new_talk, break\n");
    printf("img_start: %s\n", img_start);
    printf("img_end: %s\n", img_end);
    printf("img_content: %s\n", img_content);

    std::vector<AML_LLMImageInput> images;
    std::vector<std::vector<uint8_t>> image_buffers;

    while (true)
    {
        std::string line;
        printf("\nVLM>>> ");
        fflush(stdout);
        if (!std::getline(std::cin, line))
        {
            break;
        }
        line = trim_copy(line);
        if (line.empty())
        {
            printf("Please enter your question!\n");
            continue;
        }
        if (line == "exit")
        {
            break;
        }
        if (line == "new_talk")
        {
            aml_llm_reset(context);
            images.clear();
            image_buffers.clear();
            continue;
        }
        if (line == "break")
        {
            aml_llm_break(context);
            continue;
        }
        if (line.rfind("/image ", 0) == 0)
        {
            std::string path = trim_copy(line.substr(7));
            std::vector<uint8_t> rgb;
            uint32_t width = 0;
            uint32_t height = 0;
            if (path.empty() || !load_rgb_image(path.c_str(), rgb, &width, &height))
            {
                continue;
            }
            image_buffers.push_back(std::move(rgb));
            AML_LLMImageInput image;
            memset(&image, 0, sizeof(image));
            image.type = AML_LLM_IMAGE_TYPE_BUFFER;
            image.data = image_buffers.back().data();
            image.width = width;
            image.height = height;
            images.push_back(image);
            for (size_t i = 0; i < images.size(); ++i)
            {
                images[i].data = image_buffers[i].data();
            }
            printf("image appended, total=%zu\n", images.size());
            continue;
        }

        std::string prompt = line;
        bool use_vlm = !images.empty();
        if (use_vlm && prompt.find(img_content) == std::string::npos)
        {
            std::string markers;
            for (size_t i = 0; i < images.size(); ++i)
            {
                markers += img_content;
            }
            prompt = markers + prompt;
            printf("prompt has no %s, prepend %zu marker(s)\n", img_content, images.size());
        }

        AML_LLMInput input;
        memset(&input, 0, sizeof(input));
        input.role = "user";

        if (use_vlm)
        {
            input.input_type = AML_LLM_INPUT_MULTIMODAL;
            input.multimodal_input.prompt = prompt.c_str();
            input.multimodal_input.img_inputs = images.data();
            input.multimodal_input.img_count = (uint32_t)images.size();
            input.multimodal_input.img_content = img_content;
            input.multimodal_input.img_start = img_start;
            input.multimodal_input.img_end = img_end;
        }
        else
        {
            input.input_type = AML_LLM_INPUT_PROMPT;
            input.prompt_input = prompt.c_str();
        }

        printf("robot: ");
        aml_llm_run(context, &input, &run_config, NULL);
        images.clear();
        image_buffers.clear();
    }

    aml_llm_uninit(context);
    return 0;
}
