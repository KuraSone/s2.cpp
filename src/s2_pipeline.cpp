#include "../include/s2_pipeline.h"
#include <cstdio>

namespace s2 {

static void safe_print_ln(const std::string& msg) {
    fputs(msg.c_str(), stdout);
    fputc('\n', stdout);
}

static void safe_print_error_ln(const std::string& msg) {
    fputs(msg.c_str(), stderr);
    fputc('\n', stderr);
}

Pipeline::Pipeline() {}
Pipeline::~Pipeline() {}

bool Pipeline::init(const PipelineParams & params) {
    safe_print_ln("--- Pipeline Init ---");

    if (!tokenizer_.load(params.tokenizer_path)) {
        safe_print_error_ln("Pipeline error: could not load tokenizer from " + params.tokenizer_path);
        return false;
    }

    if (!model_.load(params.model_path, params.gpu_device, params.backend_type)) {
        safe_print_error_ln("Pipeline error: could not load model from " + params.model_path);
        return false;
    }

    if (!codec_.load(params.model_path, -1, -1)) {
        safe_print_error_ln("Pipeline error: could not load codec from " + params.model_path);
        return false;
    }

    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig & tc    = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    initialized_ = true;
    return true;
}

bool Pipeline::synthesize(const PipelineParams & params) {
    if (!initialized_) {
        safe_print_error_ln("Pipeline not initialized.");
        return false;
    }

    safe_print_ln("--- Pipeline Synthesize ---");

    std::string text_msg = "Text: " + params.text;
    safe_print_ln(text_msg);

    const int32_t num_codebooks = model_.hparams().num_codebooks;

    std::vector<int32_t> ref_codes;
    int32_t T_prompt = 0;
    if (!params.prompt_audio_path.empty()) {
        std::string loading_msg = "Loading reference audio: " + params.prompt_audio_path;
        safe_print_ln(loading_msg);

        AudioData ref_audio;
        if (load_audio(params.prompt_audio_path, ref_audio, codec_.sample_rate())) {
            if (!codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
                               params.gen.n_threads, ref_codes, T_prompt)) {
                safe_print_error_ln("Pipeline warning: encode failed, running without reference audio.");
                ref_codes.clear();
                T_prompt = 0;
            }
        } else {
            safe_print_error_ln("Pipeline warning: load_audio failed, running without reference audio.");
        }
    }

    PromptTensor prompt = build_prompt(
        tokenizer_, params.text, params.prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_codebooks, T_prompt);

    int32_t max_seq_len = prompt.cols + params.gen.max_new_tokens;
    if (!model_.init_kv_cache(max_seq_len)) {
        safe_print_error_ln("Pipeline error: init_kv_cache failed.");
        return false;
    }

    GenerateResult res = generate(model_, tokenizer_.config(), prompt, params.gen);
    if (res.n_frames == 0) {
        safe_print_error_ln("Pipeline error: generation produced no frames.");
        return false;
    }

    std::vector<float> audio_out;
    if (!codec_.decode(res.codes.data(), res.n_frames, params.gen.n_threads, audio_out)) {
        safe_print_error_ln("Pipeline error: decode failed.");
        return false;
    }

    if (!save_audio(params.output_path,
                    audio_out,
                    codec_.sample_rate(),
                    params.trim_silence,
                    params.normalize_output)) {
        safe_print_error_ln("Pipeline error: save_audio failed to " + params.output_path);
        return false;
    }

    std::string saved_msg = "Saved audio to: " + params.output_path;
    safe_print_ln(saved_msg);
    return true;
}

}
