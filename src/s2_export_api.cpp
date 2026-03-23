#include "../include/s2_export_api.h"

s2::Pipeline* AllocS2Pipeline()
{
	return new s2::Pipeline();
}
void ReleaseS2Pipeline(s2::Pipeline* Pipeline)
{
	delete Pipeline;
}
void SyncS2TokenizerConfigFromS2Model(s2::SlowARModel* Model, s2::Tokenizer* Tokenizer)
{
	const s2::ModelHParams & hp = Model->hparams();
	s2::TokenizerConfig & tc    = Tokenizer->config();
	if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
	if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
	if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
	if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
	if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
}
int InitializeS2Pipeline(s2::Pipeline* Pipeline, s2::Tokenizer* Tokenizer, s2::SlowARModel* Model, s2::AudioCodec* AudioCodec)
{
	if(!Pipeline->initialized_)
	{
		Pipeline->tokenizer_ = *Tokenizer;
		Pipeline->model_ = *Model;
		Pipeline->codec_ = *AudioCodec;
		Pipeline->initialized_ = true;
		return true;
	}
	return false;
}

s2::GenerateParams* AllocS2GenerateParams()
{
	return new s2::GenerateParams();
}
void ReleaseS2GenerateParams(s2::GenerateParams* GenerateParams)
{
	delete GenerateParams;
}
int InitializeS2GenerateParams(s2::GenerateParams* GenerateParams, int32_t max_new_tokens, float temperature, float top_p, int32_t top_k, int32_t min_tokens_before_end, int32_t n_threads, int verbose)
{
	GenerateParams->max_new_tokens = max_new_tokens >= 0 ? max_new_tokens : GenerateParams->max_new_tokens;
	GenerateParams->temperature = temperature >= 0 ? temperature : GenerateParams->temperature;
	GenerateParams->top_p = top_p >= 0 ? top_p : GenerateParams->top_p;
	GenerateParams->top_k = top_k >= 0 ? top_k : GenerateParams->top_k;
	GenerateParams->min_tokens_before_end = min_tokens_before_end >= 0 ? min_tokens_before_end : GenerateParams->min_tokens_before_end;
	GenerateParams->n_threads = n_threads >= 0 ? n_threads : GenerateParams->n_threads;
	GenerateParams->verbose = verbose >= 0 ? verbose : GenerateParams->verbose;
	return true;
}

s2::SlowARModel* AllocS2Model()
{
	return new s2::SlowARModel();
}
void ReleaseS2Model(s2::SlowARModel* Model)
{
	delete Model;
}
int InitializeS2Model(s2::SlowARModel* Model, const char* gguf_path, int32_t gpu_device, int32_t backend_type)
{
	return Model->load(std::string(gguf_path), gpu_device, backend_type);
}

s2::Tokenizer* AllocS2Tokenizer()
{
	return new s2::Tokenizer();
}
void ReleaseS2Tokenizer(s2::Tokenizer* Tokenizer)
{
	delete Tokenizer;
}
int InitializeS2Tokenizer(s2::Tokenizer* Tokenizer, const char* path)
{
	return Tokenizer->load(std::string(path));
}

s2::AudioCodec* AllocS2AudioCodec()
{
	return new s2::AudioCodec();
}
void ReleaseS2AudioCodec(s2::AudioCodec* AudioCodec)
{
	delete AudioCodec;
}
int InitializeS2AudioCodec(s2::AudioCodec* AudioCodec, const char* gguf_path, int32_t gpu_device, int32_t backend_type)
{
	return AudioCodec->load(std::string(gguf_path), gpu_device, backend_type);
}

std::vector<int32_t>* AllocS2AudioPromptCodes()
{
	return new std::vector<int32_t>();
}
void ReleaseS2AudioPromptCodes(std::vector<int32_t>* AudioPromptCodes)
{
	delete AudioPromptCodes;
}
int InitializeAudioPromptCodes(s2::Pipeline* Pipeline, int32_t ThreadCount, const char* ReferenceAudioPath, std::vector<int32_t>* AudioPromptCodes, int* TPrompt)
{
	int ReturnCode = 1;
	if(AudioPromptCodes->size() == 0)
	{
		if (ReferenceAudioPath != NULL) {
			s2::AudioData ref_audio;
			if (load_audio(std::string(ReferenceAudioPath), ref_audio, Pipeline->codec_.sample_rate())) {
				if (!Pipeline->codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
					ThreadCount, *AudioPromptCodes, *TPrompt)) {
					ReturnCode = -1; //Pipeline warning: encode failed, running without reference audio.
					AudioPromptCodes->clear();
					*TPrompt = 0;
				}
			} else {
				ReturnCode = -2; //Pipeline warning: load_audio failed, running without reference audio.
			}
		}
	}
	return ReturnCode;
}

std::vector<float>* AllocS2AudioBuffer(int InitialSize)
{
	return InitialSize > 0 ? new std::vector<float>(InitialSize) : new std::vector<float>();
}
void ReleaseS2AudioBuffer(std::vector<float>* AudioBuffer)
{
	delete AudioBuffer;
}
float* GetS2AudioBufferDataPointer(std::vector<float>* AudioBuffer)
{
	return AudioBuffer->data();
}

int S2Synthesize(s2::Pipeline* Pipeline, const s2::GenerateParams* GenerateParams, std::vector<float>* AudioBuffer, std::vector<int32_t>* ReferenceAudioPromptCodes, int32_t* ReferenceAudioTPrompt, const char* ReferenceAudioPath, const char* ReferenceAudioTranscript, const char* TextToInfer, const char* OutputAudioPath, int32_t* AudioBufferOutputLength)
{
	if(Pipeline->initialized_)
	{
		const int32_t num_codebooks = Pipeline->model_.hparams().num_codebooks; int ReturnCode = true;

		// 1. Audio Prompt Loading
		// encode() returns codes in row-major (num_codebooks, T_prompt) format,
		// matching the layout expected by build_prompt() (prompt_codes[c*T+t]).
		if(ReferenceAudioPromptCodes->size() == 0)
		{
			if (ReferenceAudioPath != NULL) {
				s2::AudioData ref_audio;
				if (load_audio(std::string(ReferenceAudioPath), ref_audio, Pipeline->codec_.sample_rate())) {
					if (!Pipeline->codec_.encode(ref_audio.samples.data(), (int32_t)ref_audio.samples.size(),
						GenerateParams->n_threads, *ReferenceAudioPromptCodes, *ReferenceAudioTPrompt)) {
						ReturnCode = -1; //Pipeline warning: encode failed, running without reference audio.
						ReferenceAudioPromptCodes->clear();
						*ReferenceAudioTPrompt = 0;
					}
				} else {
					ReturnCode = -2; //Pipeline warning: load_audio failed, running without reference audio.
				}
			}
		}

		// 2. Build Prompt Tensor
		// build_prompt expects prompt_codes as (num_codebooks, T_prompt) row-major,
		// which is exactly the format produced by encode() above.
		s2::PromptTensor prompt = s2::build_prompt(
			Pipeline->tokenizer_, std::string(TextToInfer), std::string(ReferenceAudioTranscript),
			ReferenceAudioPromptCodes->empty() ? nullptr : ReferenceAudioPromptCodes->data(),
			num_codebooks, *ReferenceAudioTPrompt);

		// 3. Setup KV Cache
		int32_t max_seq_len = prompt.cols + GenerateParams->max_new_tokens;
		if (!Pipeline->model_.init_kv_cache(max_seq_len)) {
			return -3; //Pipeline error: init_kv_cache failed.
		}

		// 4. Generate
		// generate() returns GenerateResult.codes in row-major (num_codebooks, n_frames).
		s2::GenerateResult res = s2::generate(Pipeline->model_, Pipeline->tokenizer_.config(), prompt, *GenerateParams);
		if (res.n_frames == 0) {
			return -4; //Pipeline error: generation produced no frames.
		}

		// 5. Decode
		// codec_.decode() receives codes in row-major (num_codebooks, n_frames),
		// which matches GenerateResult.codes layout.
		std::vector<float>* audio_out = AudioBuffer == NULL ? new std::vector<float>() : AudioBuffer; int32_t audio_n_frames_out = 0;
		if (!Pipeline->codec_.decode(res.codes.data(), res.n_frames, GenerateParams->n_threads, *audio_out, audio_n_frames_out)) {
			return -5; //Pipeline error: decode failed.
		}
		if(AudioBufferOutputLength != NULL) { *AudioBufferOutputLength = audio_n_frames_out; }

		if(OutputAudioPath != NULL)
		{
			// 6. Save
			if (!s2::save_audio(std::string(OutputAudioPath), *audio_out, Pipeline->codec_.sample_rate())) {
				return -6; //Pipeline error: save_audio failed to + (params.output_path).
			}
		}
		return ReturnCode;
	}
	return false;
}