#pragma once

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <string>

namespace s2 {

struct PipelineParams {
    std::string model_path;
    std::string tokenizer_path;
    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;
    GenerateParams gen;
    int32_t gpu_device = -1;   // -1 = CPU only
    int32_t backend_type = -1; //0 = Vulkan; 1 = Cuda;
    bool trim_silence = true;
    bool normalize_output = true;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);
    bool synthesize(const PipelineParams & params);

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    bool initialized_ = false;
};

}
