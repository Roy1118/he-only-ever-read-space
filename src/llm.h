// ============================================================================
//  llm.h -- Q8 transformer inference for the "imprisoned AI" installation.
//
//  Reads the interleaved-layout model produced by tools/convert_esp.py straight
//  out of its flash partition via esp_partition_mmap(). Nothing is copied into
//  RAM: the weights are read in place, sequentially, from the memory-mapped
//  window. Measured on this board: ~25 MB/s, so ~0.36 s per generated token
//  (one token touches all 8.94 MB of weights, and the 32 KB data cache cannot
//  hold a meaningful fraction of that across steps).
//
//  On-disk layout (see tools/convert_esp.py):
//    [64B header]  magic "LLME", version, then dim/hidden/layers/heads/kv/vocab/
//                  seq/gs/shared as int32
//    [fp32]        rms_att  (n_layers * dim)
//    [fp32]        rms_ffn  (n_layers * dim)
//    [fp32]        rms_final(dim)
//    per layer:    wq, wk, wv, wo, w1, w2, w3
//    [Q8]          lm_head  (== the tied token embedding, so also used for the
//                            embedding lookup at position 0 of each step)
//
//  Every Q8 tensor is stored as repeating groups of [group_size int8][1 float32
//  scale]. This is the whole point of the conversion: runq.c's native layout
//  keeps all int8 first and all scales after, which forces a random jump to the
//  scale array on every group during a row-wise matmul.
// ============================================================================
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace llm {

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int group_size;   // quantization group size (GS)
    // derived
    int head_size;
    int kv_dim;       // dim * n_kv_heads / n_heads
    int kv_mul;       // n_heads / n_kv_heads
};

// A quantized tensor as it appears on flash. `layers` worth of tensors are
// stored back to back, so layer l starts at base + l * stride.
struct Q8 {
    const uint8_t *base;
    size_t stride;    // bytes per layer
    int n;            // elements per layer
    int n_groups;     // groups per layer
};

struct Weights {
    const float *rms_att;
    const float *rms_ffn;
    const float *rms_final;
    Q8 wq, wk, wv, wo, w1, w2, w3;
    Q8 lm_head;
};

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

// Map the "model" partition and validate the header. Logs a one-line summary.
// Returns false on failure (see lastError()).
bool load();
const char *lastError();

const Config &config();
const Weights &weights();
size_t modelBytes();

// ---------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------

// Clear the KV cache and rewind to position 0. Call before each generation
// cycle -- the cache is the only carried-over state.
void resetState();

// Run one step of the network. Returns the logits for the next token, and
// remembers token/pos internally for the KV cache write.
float *forward(int token, int pos);

struct SamplerCfg {
    float temperature = 0.85f;
    float top_p = 0.90f;
    int   top_k = 0;          // 0 = disabled
    float repeat_penalty = 1.0f;  // 1.0 = off
};

void configureSampler(const SamplerCfg &cfg);
void seedSampler(uint64_t seed);
int  sample(float *logits);

// ---------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------

struct MemoryReport {
    size_t kv_bytes;
    size_t activation_bytes;
    size_t psram_free;
    size_t internal_free;
};
MemoryReport memoryReport();

}  // namespace llm
