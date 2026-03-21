#pragma once
#define S2_Export __declspec(dllexport)

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"
#include "s2_pipeline.h"
#include "s2_prompt.h"


extern "C" 
{
	S2_Export s2::Pipeline* AllocS2Pipeline();
	S2_Export void ReleaseS2Pipeline(s2::Pipeline* Pipeline);
	S2_Export void SyncS2TokenizerConfigFromS2Model(s2::SlowARModel* Model, s2::Tokenizer* Tokenizer);
	S2_Export int InitializeS2Pipeline(s2::Pipeline* Pipeline, s2::Tokenizer* Tokenizer, s2::SlowARModel* Model, s2::AudioCodec* AudioCodec);

	S2_Export s2::GenerateParams* AllocS2GenerateParams();
	S2_Export void ReleaseS2GenerateParams(s2::GenerateParams* GenerateParams);
	S2_Export int InitializeS2GenerateParams(s2::GenerateParams* GenerateParams, int32_t max_new_tokens = -1, float temperature = -1, float top_p = -1, int32_t top_k = -1, int32_t min_tokens_before_end = -1, int32_t n_threads = -1, int verbose = -1);

	S2_Export s2::SlowARModel* AllocS2Model();
	S2_Export void ReleaseS2Model(s2::SlowARModel* Model);
	S2_Export int InitializeS2Model(s2::SlowARModel* Model, const char* gguf_path, int32_t gpu_device, int32_t backend_type);

	S2_Export s2::Tokenizer* AllocS2Tokenizer();
	S2_Export void ReleaseS2Tokenizer(s2::Tokenizer* Tokenizer);
	S2_Export int InitializeS2Tokenizer(s2::Tokenizer* Tokenizer, const char* path);

	S2_Export s2::AudioCodec* AllocS2AudioCodec();
	S2_Export void ReleaseS2AudioCodec(s2::AudioCodec* AudioCodec);
	S2_Export int InitializeS2AudioCodec(s2::AudioCodec* AudioCodec, const char* gguf_path, int32_t gpu_device, int32_t backend_type);

	S2_Export std::vector<int32_t>* AllocS2AudioPromptCodes();
	S2_Export void ReleaseS2AudioPromptCodes(std::vector<int32_t>* AudioPromptCodes);
	S2_Export int InitializeAudioPromptCodes(s2::Pipeline* Pipeline, int32_t ThreadCount, const char* ReferenceAudioPath, std::vector<int32_t>* AudioPromptCodes, int* TPrompt);

	S2_Export std::vector<float>* AllocS2AudioBuffer(int InitialSize);
	S2_Export void ReleaseS2AudioBuffer(std::vector<float>* AudioBuffer);
	S2_Export float* GetS2AudioBufferDataPointer(std::vector<float>* AudioBuffer);

	S2_Export int S2Synthesize(s2::Pipeline* Pipeline, const s2::GenerateParams* GenerateParams, std::vector<float>* AudioBuffer, std::vector<int32_t>* ReferenceAudioPromptCodes, int32_t* ReferenceAudioTPrompt, const char* ReferenceAudioPath, const char* ReferenceAudioTranscript, const char* TextToInfer, const char* OutputAudioPath, int32_t* AudioBufferOutputLength);
}