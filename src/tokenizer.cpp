// ============================================================================
//  tokenizer.cpp -- see tokenizer.h
// ============================================================================

#include "tokenizer.h"

#include <string.h>
#include <stdio.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "log.h"          // must come after esp_log.h -- see the comment there
#include "esp_partition.h"

namespace tok {

static const char *TAG = "tok";
static const char *s_err = nullptr;

// 104,637 bytes of pieces + one NUL each. PSRAM: it is rarely touched compared
// to the weight stream, and internal SRAM is needed for the index and buffers.
static char *s_blob = nullptr;
static size_t s_blob_bytes = 0;

static char **s_pieces = nullptr;      // vocab_size pointers into s_blob
static int s_vocab = 0;
static int s_max_piece = 0;

// NOTE: 0xFF must not double as "not a byte token" -- the byte-fallback entry
// for 0xFF is legitimate, and using 0xFF as the sentinel silently dropped
// exactly that one token. A separate flag array removes the collision.
static uint8_t *s_byteval = nullptr;   // raw byte value, meaningful only if s_isbyte
static uint8_t *s_isbyte = nullptr;    // 1 = this token is a byte-fallback spelling
static uint8_t s_byte_pieces[512];     // the 256 single-byte strings

bool load() {
    s_err = nullptr;

    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "tok");
    if (!part) { s_err = "partition \"tok\" not found"; ESP_LOGE(TAG, "%s", s_err); return false; }

    // We do not know the vocab size from the file (llama2.c leaves it out and
    // passes it in from the model config), so read max_token_length and then
    // size the blob from the *largest possible* content, capped by the partition.
    int32_t mtl = 0;
    if (esp_partition_read(part, 0, &mtl, 4) != ESP_OK) {
        s_err = "cannot read tok header"; ESP_LOGE(TAG, "%s", s_err); return false;
    }
    if (mtl <= 0 || mtl > 4096) { s_err = "implausible max_token_length";
        ESP_LOGE(TAG, "%s (%d)", s_err, (int)mtl); return false; }
    s_max_piece = mtl;

    // Read the whole partition content we could possibly need, then parse it in
    // RAM. The partition is much larger than the file (1.69 MB vs 0.10 MB), so
    // stop as soon as the stream runs into erased flash.
    const size_t want = part->size < (1u << 20) ? part->size : (1u << 20);
    uint8_t *raw = (uint8_t *)heap_caps_malloc(want, MALLOC_CAP_SPIRAM);
    if (!raw) { s_err = "tok staging alloc failed"; ESP_LOGE(TAG, "%s", s_err); return false; }
    if (esp_partition_read(part, 0, raw, want) != ESP_OK) {
        heap_caps_free(raw);
        s_err = "tok read failed"; ESP_LOGE(TAG, "%s", s_err); return false;
    }

    // ---- first pass: count entries until the stream goes blank ----
    size_t off = 4;
    int count = 0;
    while (off + 8 <= want) {
        int32_t len;
        memcpy(&len, raw + off + 4, 4);
        if (len == 0) break;                     // 0xFFFFFFFF (blank) or empty tail
        if (len < 0 || len > s_max_piece || off + 8 + (size_t)len > want) break;
        off += 8 + (size_t)len;
        count++;
    }
    if (count < 256) {
        heap_caps_free(raw);
        s_err = "tok partition does not look like a tokenizer";
        ESP_LOGE(TAG, "%s (parsed %d entries)", s_err, count);
        return false;
    }
    s_vocab = count;
    ESP_LOGI(TAG, "tokenizer: %d entries, max_token_length=%d, %u bytes used",
             s_vocab, s_max_piece, (unsigned)off);

    // ---- second pass: copy pieces into a NUL-terminated blob ----
    s_blob_bytes = off + (size_t)s_vocab;        // +1 NUL per piece
    s_blob = (char *)heap_caps_malloc(s_blob_bytes, MALLOC_CAP_SPIRAM);
    s_pieces = (char **)heap_caps_malloc((size_t)s_vocab * sizeof(char *), MALLOC_CAP_INTERNAL);
    s_byteval = (uint8_t *)heap_caps_malloc((size_t)s_vocab, MALLOC_CAP_INTERNAL);
    s_isbyte = (uint8_t *)heap_caps_malloc((size_t)s_vocab, MALLOC_CAP_INTERNAL);
    if (!s_blob || !s_pieces || !s_byteval || !s_isbyte) {
        heap_caps_free(raw);
        s_err = "tok index alloc failed"; ESP_LOGE(TAG, "%s", s_err); return false;
    }

    memset(s_isbyte, 0, (size_t)s_vocab);
    memset(s_byteval, 0, (size_t)s_vocab);
    for (int i = 0; i < 256; i++) {
        s_byte_pieces[i * 2] = (uint8_t)i;
        s_byte_pieces[i * 2 + 1] = 0;
    }

    size_t src = 4, dst = 0;
    for (int i = 0; i < s_vocab; i++) {
        int32_t len;
        memcpy(&len, raw + src + 4, 4);
        char *out = s_blob + dst;
        memcpy(out, raw + src + 8, (size_t)len);
        out[len] = '\0';
        dst += (size_t)len + 1;
        s_pieces[i] = out;

        // detect the byte-fallback spelling, e.g. "<0x7F>"
        if (len == 6 && out[0] == '<' && out[1] == '0' && out[2] == 'x' && out[5] == '>') {
            auto hexv = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int hi = hexv(out[3]), lo = hexv(out[4]);
            if (hi >= 0 && lo >= 0) {
                s_byteval[i] = (uint8_t)(hi * 16 + lo);
                s_isbyte[i] = 1;
            }
        }
        src += 8 + (size_t)len;
    }
    heap_caps_free(raw);

    int nbytes = 0;
    for (int i = 0; i < s_vocab; i++) if (s_isbyte[i]) nbytes++;
    ESP_LOGI(TAG, "indexed %d pieces (%d byte-fallback), blob %u bytes",
             s_vocab, nbytes, (unsigned)s_blob_bytes);
    return true;
}

const char *lastError() { return s_err; }
int vocabSize() { return s_vocab; }
int maxPieceBytes() { return s_max_piece; }
bool isByteFallback(int token) {
    return token >= 0 && token < s_vocab && s_isbyte[token] != 0;
}
uint8_t byteValue(int token) {
    return (token >= 0 && token < s_vocab) ? s_byteval[token] : 0;
}

// Static: single-threaded by design, and callers use the result immediately.
static char s_scratch[256];

const char *piece(int prev_token, int token) {
    if (!s_pieces || token < 0 || token >= s_vocab) return "";
    if (s_isbyte[token]) {
        return (const char *)s_byte_pieces + (size_t)s_byteval[token] * 2;
    }
    const char *p = s_pieces[token];
    int len = (int)strlen(p);
    // a piece right after BOS loses one leading space (see llama2.c decode())
    if (prev_token == 1 && len > 0 && p[0] == ' ') { p++; len--; }
    if (len >= (int)sizeof(s_scratch)) len = (int)sizeof(s_scratch) - 1;
    memcpy(s_scratch, p, (size_t)len);
    s_scratch[len] = '\0';
    return s_scratch;
}

int utf8Feed(Utf8Assembler *a, uint8_t b) {
    if (a->have == 0) {
        if (b < 0x80) { a->buf[0] = b; return 1; }
        if ((b & 0xE0) == 0xC0) { a->buf[0] = b; a->have = 1; a->need = 2; return 0; }
        if ((b & 0xF0) == 0xE0) { a->buf[0] = b; a->have = 1; a->need = 3; return 0; }
        if ((b & 0xF8) == 0xF0) { a->buf[0] = b; a->have = 1; a->need = 4; return 0; }
        // lone continuation byte: not recoverable, drop it
        return 0;
    }
    a->buf[a->have++] = b;
    if (a->have == a->need) {
        const int n = a->have;
        a->have = 0;
        a->need = 0;
        return n;
    }
    return 0;
}

}  // namespace tok
