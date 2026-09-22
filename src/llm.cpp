// ============================================================================
//  llm.cpp -- Q8 transformer inference reading weights straight from flash.
//
//  Structure follows karpathy/llama2.c's runq.c so the numerics match the
//  training-time implementation exactly, with two deliberate differences:
//
//    1. Weights are read in the interleaved group layout (see llm.h), which
//       turns every matmul into a purely sequential flash stream.
//
//    2. No per-tensor random access: the token-embedding lookup dequantizes a
//       single row on the fly (dim = 288 elements = 9 groups = 324 bytes)
//       instead of materialising the whole 2.2 MB table into RAM.
//
//  Only the KV cache lives in PSRAM; activations stay in internal RAM because
//  they are touched thousands of times per token and internal SRAM is much
//  faster. Weights are never copied at all -- they are used in place, through
//  the mmap window.
// ============================================================================

#include "llm.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "log.h"          // must come after esp_log.h -- see the comment there
#include "esp_partition.h"
#include "esp_spi_flash.h"

namespace llm {

static const char *TAG = "llm";

// ---------------------------------------------------------------- module state
static const char *s_err = nullptr;
static Config s_cfg = {};
static Weights s_w = {};
static const esp_partition_t *s_part = nullptr;
static const void *s_map = nullptr;
static spi_flash_mmap_handle_t s_map_handle = 0;
static size_t s_model_bytes = 0;

// activation buffers (internal RAM)
static float *s_x, *s_xb, *s_xb2, *s_hb, *s_hb2;
static float *s_q, *s_k, *s_v, *s_att, *s_logits;
static float *s_key_cache, *s_value_cache;   // PSRAM
static int8_t *s_xq_q, *s_hq_q;
static float *s_xq_s, *s_hq_s;

struct Grouped { int8_t *q; float *s; };
static Grouped s_xq, s_hq;

// sampler
static SamplerCfg s_scfg;
static uint64_t s_rng = 0x9E3779B97F4A7C15ULL;
struct ProbIndex { float prob; int index; };
static ProbIndex *s_probindex = nullptr;   // PSRAM, vocab_size entries

// short history for the repetition penalty
static const int HIST = 64;
static int s_hist[HIST];
static int s_hist_n = 0;
static int s_hist_pos = 0;

// ---------------------------------------------------------------- small helpers

static inline size_t tensor_bytes(int n, int gs) {
    return (size_t)(n / gs) * (size_t)(gs + 4);
}

static inline const uint8_t *layer_ptr(const Q8 &t, int l) {
    return t.base + (size_t)l * t.stride;
}

static void set_err(const char *msg) {
    s_err = msg;
    ESP_LOGE(TAG, "%s", msg);
}

static void *alloc(size_t bytes, uint32_t caps, const char *what) {
    void *p = heap_caps_malloc(bytes, caps);
    if (!p) {
        ESP_LOGE(TAG, "alloc %s failed: %u bytes", what, (unsigned)bytes);
        return nullptr;
    }
    ESP_LOGI(TAG, "  %-14s %8u bytes", what, (unsigned)bytes);
    return p;
}

// ------------------------------------------------------------------ load / map

bool load() {
    s_err = nullptr;

    s_part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA,
                                      ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!s_part) { set_err("partition \"model\" not found"); return false; }
    ESP_LOGI(TAG, "model partition at 0x%06X, %u bytes",
             (unsigned)s_part->address, (unsigned)s_part->size);

    // --- read just the header first to learn the real weight size ---
    uint8_t hdr[64];
    if (esp_partition_read(s_part, 0, hdr, sizeof(hdr)) != ESP_OK) {
        set_err("cannot read model header"); return false;
    }
    uint32_t magic; memcpy(&magic, hdr + 0, 4);
    uint32_t version; memcpy(&version, hdr + 4, 4);
    if (magic != 0x4C4C4D45) { set_err("bad magic (want \"LLME\")"); return false; }
    if (version != 1) { set_err("unsupported model version"); return false; }

    memcpy(&s_cfg.dim,        hdr + 8,  4);
    memcpy(&s_cfg.hidden_dim, hdr + 12, 4);
    memcpy(&s_cfg.n_layers,   hdr + 16, 4);
    memcpy(&s_cfg.n_heads,    hdr + 20, 4);
    memcpy(&s_cfg.n_kv_heads, hdr + 24, 4);
    memcpy(&s_cfg.vocab_size, hdr + 28, 4);
    memcpy(&s_cfg.seq_len,    hdr + 32, 4);
    memcpy(&s_cfg.group_size, hdr + 36, 4);

    int shared = 0;
    memcpy(&shared, hdr + 40, 4);
    if (!shared) { set_err("model must use tied embeddings"); return false; }

    const int dim = s_cfg.dim, gs = s_cfg.group_size;
    if (gs <= 0 || (gs % 4) != 0) { set_err("bad group size"); return false; }
    if (dim % gs || s_cfg.hidden_dim % gs) { set_err("dim not divisible by group size"); return false; }
    if (s_cfg.n_heads <= 0 || s_cfg.n_kv_heads <= 0 || s_cfg.n_heads % s_cfg.n_kv_heads) {
        set_err("bad head configuration"); return false;
    }

    s_cfg.head_size = dim / s_cfg.n_heads;
    s_cfg.kv_mul    = s_cfg.n_heads / s_cfg.n_kv_heads;
    s_cfg.kv_dim    = dim * s_cfg.n_kv_heads / s_cfg.n_heads;

    // --- expected size, so a truncated flash fails loudly instead of silently ---
    const int head_size = s_cfg.head_size;
    size_t expect = 64;
    expect += (size_t)s_cfg.n_layers * dim * 4 * 2;   // rms_att + rms_ffn
    expect += (size_t)dim * 4;                        // rms_final
    size_t per_layer = 0;
    per_layer += tensor_bytes(dim * (s_cfg.n_heads * head_size), gs);   // wq
    per_layer += tensor_bytes(dim * s_cfg.kv_dim, gs);                  // wk
    per_layer += tensor_bytes(dim * s_cfg.kv_dim, gs);                  // wv
    per_layer += tensor_bytes((s_cfg.n_heads * head_size) * dim, gs);   // wo
    per_layer += tensor_bytes(dim * s_cfg.hidden_dim, gs);              // w1
    per_layer += tensor_bytes(s_cfg.hidden_dim * dim, gs);              // w2
    per_layer += tensor_bytes(dim * s_cfg.hidden_dim, gs);              // w3
    expect += per_layer * s_cfg.n_layers;
    expect += tensor_bytes(s_cfg.vocab_size * dim, gs);                 // lm_head

    ESP_LOGI(TAG, "config: dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d gs=%d",
             dim, s_cfg.hidden_dim, s_cfg.n_layers, s_cfg.n_heads, s_cfg.n_kv_heads,
             s_cfg.vocab_size, s_cfg.seq_len, gs);
    ESP_LOGI(TAG, "expect %u bytes of weights, partition holds %u",
             (unsigned)expect, (unsigned)s_part->size);
    if (expect > s_part->size) { set_err("model larger than its partition"); return false; }

    // --- map the whole thing in one shot (measured: the MMU gives us 9 MB) ---
    esp_err_t e = esp_partition_mmap(s_part, 0, expect, SPI_FLASH_MMAP_DATA,
                                     &s_map, &s_map_handle);
    if (e != ESP_OK || !s_map) { set_err("esp_partition_mmap failed"); return false; }
    s_model_bytes = expect;
    ESP_LOGI(TAG, "mapped %u bytes at %p (no copy)", (unsigned)expect, s_map);

    // --- walk the pointers ---
    const uint8_t *p = (const uint8_t *)s_map;
    s_w.rms_att   = (const float *)(p + 64);
    s_w.rms_ffn   = s_w.rms_att + (size_t)s_cfg.n_layers * dim;
    s_w.rms_final = s_w.rms_ffn + (size_t)s_cfg.n_layers * dim;
    p = (const uint8_t *)(s_w.rms_final + dim);

    const int n_layers = s_cfg.n_layers;

    // Per-layer tensors are stored LAYER-MAJOR: all seven tensors of layer 0,
    // then all seven of layer 1, and so on (see the loop in convert_esp.py).
    // So every tensor shares the same stride -- the sum of the seven -- and its
    // base is offset by the sizes of the tensors before it within a layer.
    //
    // Getting this wrong is invisible to a total-bytes check (the sum is
    // identical either way), so the layout assertion below cannot catch it; only
    // comparing logits against a reference implementation can.
    const size_t sz_wq = tensor_bytes(dim * (s_cfg.n_heads * head_size), gs);
    const size_t sz_wk = tensor_bytes(dim * s_cfg.kv_dim, gs);
    const size_t sz_wv = sz_wk;
    const size_t sz_wo = tensor_bytes((s_cfg.n_heads * head_size) * dim, gs);
    const size_t sz_w1 = tensor_bytes(dim * s_cfg.hidden_dim, gs);
    const size_t sz_w2 = tensor_bytes(s_cfg.hidden_dim * dim, gs);
    const size_t sz_w3 = sz_w1;
    const size_t layer_stride = sz_wq + sz_wk + sz_wv + sz_wo + sz_w1 + sz_w2 + sz_w3;

    const uint8_t *layer0 = p;
    auto mk = [&](Q8 &t, size_t offset, int n) {
        t.base     = layer0 + offset;
        t.stride   = layer_stride;
        t.n        = n;
        t.n_groups = n / gs;
    };
    size_t off = 0;
    mk(s_w.wq, off, dim * (s_cfg.n_heads * head_size));      off += sz_wq;
    mk(s_w.wk, off, dim * s_cfg.kv_dim);                     off += sz_wk;
    mk(s_w.wv, off, dim * s_cfg.kv_dim);                     off += sz_wv;
    mk(s_w.wo, off, (s_cfg.n_heads * head_size) * dim);      off += sz_wo;
    mk(s_w.w1, off, dim * s_cfg.hidden_dim);                 off += sz_w1;
    mk(s_w.w2, off, s_cfg.hidden_dim * dim);                 off += sz_w2;
    mk(s_w.w3, off, dim * s_cfg.hidden_dim);                 off += sz_w3;

    p += layer_stride * (size_t)n_layers;

    // lm_head is a single tensor (tied to the embedding table)
    s_w.lm_head.base     = p;
    s_w.lm_head.n        = s_cfg.vocab_size * dim;
    s_w.lm_head.n_groups = s_w.lm_head.n / gs;
    s_w.lm_head.stride   = tensor_bytes(s_w.lm_head.n, gs);
    p += s_w.lm_head.stride;

    const size_t consumed = (size_t)(p - (const uint8_t *)s_map);
    ESP_LOGI(TAG, "layout walk consumed %u / %u bytes %s",
             (unsigned)consumed, (unsigned)expect, consumed == expect ? "OK" : "MISMATCH");
    if (consumed != expect) { set_err("layout walk mismatch"); return false; }

    // ---------------------------------------------------------------- buffers
    const int kv_dim = s_cfg.kv_dim;
    const int hidden = s_cfg.hidden_dim;
    const int vocab  = s_cfg.vocab_size;

    ESP_LOGI(TAG, "allocating run state");
    s_x     = (float *)alloc(dim * 4, MALLOC_CAP_INTERNAL, "x");
    s_xb    = (float *)alloc(dim * 4, MALLOC_CAP_INTERNAL, "xb");
    s_xb2   = (float *)alloc(dim * 4, MALLOC_CAP_INTERNAL, "xb2");
    s_hb    = (float *)alloc(hidden * 4, MALLOC_CAP_INTERNAL, "hb");
    s_hb2   = (float *)alloc(hidden * 4, MALLOC_CAP_INTERNAL, "hb2");
    s_q     = (float *)alloc(dim * 4, MALLOC_CAP_INTERNAL, "q");
    s_k     = (float *)alloc(kv_dim * 4, MALLOC_CAP_INTERNAL, "k");
    s_v     = (float *)alloc(kv_dim * 4, MALLOC_CAP_INTERNAL, "v");
    s_att   = (float *)alloc((size_t)s_cfg.n_heads * s_cfg.seq_len * 4,
                             MALLOC_CAP_INTERNAL, "att");
    s_logits = (float *)alloc(vocab * 4, MALLOC_CAP_INTERNAL, "logits");
    s_xq_q  = (int8_t *)alloc(dim, MALLOC_CAP_INTERNAL, "xq.q");
    s_xq_s  = (float *)alloc((dim / gs) * 4, MALLOC_CAP_INTERNAL, "xq.s");
    s_hq_q  = (int8_t *)alloc(hidden, MALLOC_CAP_INTERNAL, "hq.q");
    s_hq_s  = (float *)alloc((hidden / gs) * 4, MALLOC_CAP_INTERNAL, "hq.s");

    if (!s_x || !s_xb || !s_xb2 || !s_hb || !s_hb2 || !s_q || !s_k || !s_v ||
        !s_att || !s_logits || !s_xq_q || !s_xq_s || !s_hq_q || !s_hq_s) {
        set_err("activation allocation failed"); return false;
    }
    s_xq = {s_xq_q, s_xq_s};
    s_hq = {s_hq_q, s_hq_s};

    // KV cache goes to PSRAM: it is large and only touched once per step
    const size_t kv_elems = (size_t)s_cfg.n_layers * s_cfg.seq_len * kv_dim;
    s_key_cache   = (float *)alloc(kv_elems * 4, MALLOC_CAP_SPIRAM, "key_cache");
    s_value_cache = (float *)alloc(kv_elems * 4, MALLOC_CAP_SPIRAM, "value_cache");
    if (!s_key_cache || !s_value_cache) { set_err("KV cache allocation failed"); return false; }

    // top-p scratch (vocab * 8 bytes -- PSRAM, we do not want this in SRAM)
    s_probindex = (ProbIndex *)alloc((size_t)vocab * sizeof(ProbIndex),
                                     MALLOC_CAP_SPIRAM, "probindex");
    if (!s_probindex) { set_err("sampler scratch allocation failed"); return false; }

    resetState();

    ESP_LOGI(TAG, "model ready: %u bytes weights, internal free %u, psram free %u",
             (unsigned)s_model_bytes,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    return true;
}

const char *lastError() { return s_err; }
const Config &config() { return s_cfg; }
const Weights &weights() { return s_w; }
size_t modelBytes() { return s_model_bytes; }

void resetState() {
    const size_t kv_elems = (size_t)s_cfg.n_layers * s_cfg.seq_len * s_cfg.kv_dim;
    if (s_key_cache)   memset(s_key_cache, 0, kv_elems * 4);
    if (s_value_cache) memset(s_value_cache, 0, kv_elems * 4);
    if (s_x)           memset(s_x, 0, s_cfg.dim * 4);
    s_hist_n = 0;
    s_hist_pos = 0;
}

MemoryReport memoryReport() {
    MemoryReport r = {};
    r.kv_bytes = (size_t)s_cfg.n_layers * s_cfg.seq_len * s_cfg.kv_dim * 4 * 2;
    r.activation_bytes = 0;
    r.psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    r.internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    return r;
}

// ------------------------------------------------------------- math primitives

static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (x[j] * ss);
}

static void softmax(float *x, int size) {
    float maxv = x[0];
    for (int i = 1; i < size; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { float e = expf(x[i] - maxv); x[i] = e; sum += e; }
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < size; i++) x[i] *= inv; }
}

static void quantize(Grouped *a, const float *x, int n) {
    const int gs = s_cfg.group_size;
    const int ng = n / gs;
    for (int g = 0; g < ng; g++) {
        const float *xg = x + g * gs;
        float wmax = 0.0f;
        for (int i = 0; i < gs; i++) { float v = fabsf(xg[i]); if (v > wmax) wmax = v; }
        float scale = wmax / 127.0f;
        a->s[g] = scale;
        float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        int8_t *q = a->q + g * gs;
        for (int i = 0; i < gs; i++) {
            int v = (int)lrintf(xg[i] * inv);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            q[i] = (int8_t)v;
        }
    }
}

// W (d,n) @ x (n,) -> xout (d,), both sides quantized.
// `w` points at the layer's tensor in the interleaved layout, so row i is a
// contiguous run of n/gs groups: reading the whole tensor is sequential.
static void matmul(float *xout, const Grouped *x, const uint8_t *w, int n, int d) {
    const int gs = s_cfg.group_size;
    const int per_group = gs + 4;
    const int ngr = n / gs;
    const size_t row_bytes = (size_t)ngr * per_group;

    for (int i = 0; i < d; i++) {
        const uint8_t *row = w + (size_t)i * row_bytes;
        float val = 0.0f;
        for (int g = 0; g < ngr; g++) {
            const uint8_t *grp = row + (size_t)g * per_group;
            const int8_t *wq = (const int8_t *)grp;
            float ws;
            memcpy(&ws, grp + gs, sizeof(float));
            const int8_t *xq = x->q + g * gs;

            // 4 partial sums for instruction-level parallelism
            int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            int k = 0;
            for (; k + 3 < gs; k += 4) {
                a0 += (int32_t)xq[k    ] * (int32_t)wq[k    ];
                a1 += (int32_t)xq[k + 1] * (int32_t)wq[k + 1];
                a2 += (int32_t)xq[k + 2] * (int32_t)wq[k + 2];
                a3 += (int32_t)xq[k + 3] * (int32_t)wq[k + 3];
            }
            for (; k < gs; k++) a0 += (int32_t)xq[k] * (int32_t)wq[k];

            val += (float)(a0 + a1 + a2 + a3) * ws * x->s[g];
        }
        xout[i] = val;
    }
}

// Dequantize one row of the interleaved embedding table (no full copy).
static void dequant_row(float *out, const uint8_t *table, int row, int dim) {
    const int gs = s_cfg.group_size;
    const int per_group = gs + 4;
    const int gpr = dim / gs;                       // groups per row (9)
    const uint8_t *r = table + (size_t)row * gpr * per_group;
    for (int g = 0; g < gpr; g++) {
        const uint8_t *grp = r + (size_t)g * per_group;
        const int8_t *q = (const int8_t *)grp;
        float s;
        memcpy(&s, grp + gs, sizeof(float));
        float *o = out + g * gs;
        for (int k = 0; k < gs; k++) o[k] = (float)q[k] * s;
    }
}

// ------------------------------------------------------------------- forward

float *forward(int token, int pos) {
    const Config &p = s_cfg;
    const Weights &w = s_w;
    const int dim = p.dim, kv_dim = p.kv_dim, hidden = p.hidden_dim;
    const int kv_mul = p.kv_mul, head_size = p.head_size;
    float *x = s_x;

    if (token < 0 || token >= p.vocab_size) token = 0;
    dequant_row(x, w.lm_head.base, token, dim);

    for (int l = 0; l < p.n_layers; l++) {
        // ---- attention block ----
        rmsnorm(s_xb, x, w.rms_att + (size_t)l * dim, dim);
        quantize(&s_xq, s_xb, dim);
        matmul(s_q, &s_xq, layer_ptr(w.wq, l), dim, dim);
        matmul(s_k, &s_xq, layer_ptr(w.wk, l), dim, kv_dim);
        matmul(s_v, &s_xq, layer_ptr(w.wv, l), dim, kv_dim);

        // RoPE (adjacent-pair rotation, matching train.py's apply_rotary_emb)
        for (int i = 0; i < dim; i += 2) {
            const int hd = i % head_size;
            const float freq = 1.0f / powf(10000.0f, (float)hd / (float)head_size);
            const float ang = (float)pos * freq;
            const float fcr = cosf(ang), fci = sinf(ang);
            const int rotn = (i < kv_dim) ? 2 : 1;   // 2 = rotate q and k
            for (int vi = 0; vi < rotn; vi++) {
                float *vec = (vi == 0) ? s_q : s_k;
                const float v0 = vec[i], v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // write this step's key/value into the cache
        const size_t loff = (size_t)l * p.seq_len * kv_dim;
        float *krow = s_key_cache + loff + (size_t)pos * kv_dim;
        float *vrow = s_value_cache + loff + (size_t)pos * kv_dim;
        memcpy(krow, s_k, kv_dim * sizeof(float));
        memcpy(vrow, s_v, kv_dim * sizeof(float));

        // ---- multi-head attention over 0..pos ----
        for (int h = 0; h < p.n_heads; h++) {
            const float *qh = s_q + h * head_size;
            float *att = s_att + (size_t)h * p.seq_len;
            const size_t kbase = loff + (size_t)(h / kv_mul) * head_size;

            for (int t = 0; t <= pos; t++) {
                const float *kh = s_key_cache + kbase + (size_t)t * kv_dim;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += qh[i] * kh[i];
                att[t] = score / sqrtf((float)head_size);
            }
            softmax(att, pos + 1);

            float *xbh = s_xb + h * head_size;
            memset(xbh, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *vh = s_value_cache + kbase + (size_t)t * kv_dim;
                const float a = att[t];
                for (int i = 0; i < head_size; i++) xbh[i] += a * vh[i];
            }
        }

        // ---- attention output projection + residual ----
        quantize(&s_xq, s_xb, dim);
        matmul(s_xb2, &s_xq, layer_ptr(w.wo, l), dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s_xb2[i];

        // ---- feed-forward (SwiGLU) ----
        rmsnorm(s_xb, x, w.rms_ffn + (size_t)l * dim, dim);
        quantize(&s_xq, s_xb, dim);
        matmul(s_hb,  &s_xq, layer_ptr(w.w1, l), dim, hidden);
        matmul(s_hb2, &s_xq, layer_ptr(w.w3, l), dim, hidden);
        for (int i = 0; i < hidden; i++) {
            float val = s_hb[i];
            val *= 1.0f / (1.0f + expf(-val));   // silu
            s_hb[i] = val * s_hb2[i];
        }
        quantize(&s_hq, s_hb, hidden);
        matmul(s_xb, &s_hq, layer_ptr(w.w2, l), hidden, dim);
        for (int i = 0; i < dim; i++) x[i] += s_xb[i];
    }

    // final norm + classifier (shares the embedding table)
    rmsnorm(x, x, w.rms_final, dim);
    quantize(&s_xq, x, dim);
    matmul(s_logits, &s_xq, w.lm_head.base, dim, p.vocab_size);
    return s_logits;
}

// -------------------------------------------------------------------- sampler

void configureSampler(const SamplerCfg &cfg) { s_scfg = cfg; }

void seedSampler(uint64_t seed) {
    s_rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    // xorshift64* needs a non-zero state
    if (s_rng == 0) s_rng = 1;
}

static inline uint32_t random_u32() {
    s_rng ^= s_rng >> 12;
    s_rng ^= s_rng << 25;
    s_rng ^= s_rng >> 27;
    return (uint32_t)((s_rng * 0x2545F4914F6CDD1DULL) >> 32);
}

static inline float random_f32() {
    return (float)(random_u32() >> 8) / 16777216.0f;
}

static void hist_push(int t) {
    s_hist[s_hist_pos] = t;
    s_hist_pos = (s_hist_pos + 1) % HIST;
    if (s_hist_n < HIST) s_hist_n++;
}

static int sample_argmax(const float *p, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (p[i] > p[best]) best = i;
    return best;
}

static int cmp_prob_desc(const void *a, const void *b) {
    const ProbIndex *x = (const ProbIndex *)a, *y = (const ProbIndex *)b;
    if (x->prob > y->prob) return -1;
    if (x->prob < y->prob) return 1;
    return 0;
}

static int sample_mult(const float *p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += p[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

int sample(float *logits) {
    const int n = s_cfg.vocab_size;

    // repetition penalty: discourage the last few tokens repeating verbatim,
    // which small models fall into easily (and is very visible on the OLED)
    if (s_scfg.repeat_penalty > 1.0f && s_hist_n > 0) {
        for (int i = 0; i < s_hist_n; i++) {
            const int t = s_hist[i];
            if (t < 0 || t >= n) continue;
            if (logits[t] > 0.0f) logits[t] /= s_scfg.repeat_penalty;
            else                  logits[t] *= s_scfg.repeat_penalty;
        }
    }

    int next;
    if (s_scfg.temperature <= 0.0f) {
        next = sample_argmax(logits, n);
    } else {
        const float inv_t = 1.0f / s_scfg.temperature;
        for (int i = 0; i < n; i++) logits[i] *= inv_t;

        // top-k: keep only the k best by zeroing the rest
        if (s_scfg.top_k > 0 && s_scfg.top_k < n) {
            // partial selection is enough: we only need the k-th largest value
            // as a threshold. Reuse probindex as scratch.
            for (int i = 0; i < n; i++) { s_probindex[i].prob = logits[i]; s_probindex[i].index = i; }
            qsort(s_probindex, n, sizeof(ProbIndex), cmp_prob_desc);
            const float thr = s_probindex[s_scfg.top_k - 1].prob;
            for (int i = 0; i < n; i++) if (logits[i] < thr) logits[i] = -INFINITY;
        }

        softmax(logits, n);
        const float coin = random_f32();

        if (s_scfg.top_p <= 0.0f || s_scfg.top_p >= 1.0f) {
            next = sample_mult(logits, n, coin);
        } else {
            // nucleus sampling
            int n0 = 0;
            const float cutoff = (1.0f - s_scfg.top_p) / (float)(n - 1);
            for (int i = 0; i < n; i++) {
                if (logits[i] >= cutoff) {
                    s_probindex[n0].index = i;
                    s_probindex[n0].prob = logits[i];
                    n0++;
                }
            }
            qsort(s_probindex, n0, sizeof(ProbIndex), cmp_prob_desc);

            float cum = 0.0f;
            int last = n0 - 1;
            for (int i = 0; i < n0; i++) {
                cum += s_probindex[i].prob;
                if (cum > s_scfg.top_p) { last = i; break; }
            }
            const float r = coin * cum;
            float cdf = 0.0f;
            next = s_probindex[last].index;
            for (int i = 0; i <= last; i++) {
                cdf += s_probindex[i].prob;
                if (r < cdf) { next = s_probindex[i].index; break; }
            }
        }
    }

    hist_push(next);
    return next;
}

}  // namespace llm
