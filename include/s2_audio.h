#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct AudioData {
    std::vector<float> samples;
    int32_t            sample_rate = 0;
};

bool audio_read(const std::string & path, AudioData & out);
bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate);
std::vector<float> audio_resample(const float * data, size_t n_samples, int32_t src_rate, int32_t dst_rate);
std::vector<float> audio_trim_trailing_silence(const float * data, size_t n_samples,
                                              int32_t sample_rate,
                                              float threshold = 0.01f,
                                              float min_silence_duration = 0.1f);

bool load_audio(const std::string & path, AudioData & out, int32_t target_sample_rate = 0);
bool save_audio(const std::string & path,
                const std::vector<float> & data,
                int32_t sample_rate,
                bool trim_silence = true,
                bool normalize_peak = true);

}
