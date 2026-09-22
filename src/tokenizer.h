// ============================================================================
//  tokenizer.h -- device-side decode for the llama2.c tokenizer format.
//
//  The device never *encodes*: the prompts in include/prompts.h are already
//  tokenized at build time, and generation only ever needs token -> text. So
//  this is deliberately decode-only, which keeps it tiny and removes the whole
//  sorted-vocabulary/BPE-merge machinery.
//
//  File format (produced by llama2.c/tokenizer.py):
//    int32   max_token_length
//    repeat vocab_size times:
//      float32 score
//      int32   length
//      bytes   piece (UTF-8, not NUL terminated)
// ============================================================================
#pragma once

#include <stdint.h>

namespace tok {

// Load the "tok" partition into RAM and index it.
bool load();
const char *lastError();

int vocabSize();
int maxPieceBytes();

// Decoded piece for `token`, as a NUL-terminated string.
//
//   prev_token is the token that came before it: sentencepiece's decoder strips
//   one leading space from the first piece after BOS, and llama2.c reproduces
//   that. Passing the previous token keeps our output byte-identical to run.c.
//
//   Byte-fallback tokens (stored as the literal text "<0xAB>") are returned as
//   the single raw byte 0xAB, so callers must treat the result as bytes, not as
//   a complete UTF-8 sequence.
const char *piece(int prev_token, int token);

bool isByteFallback(int token);
uint8_t byteValue(int token);

// ---------------------------------------------------------------------------
// Byte-fallback tokens split a single UTF-8 codepoint across several tokens, so
// the output stream has to be reassembled before anything can render it. Without
// this the OLED would receive lone continuation bytes and print garbage.
// ---------------------------------------------------------------------------
struct Utf8Assembler {
    uint8_t buf[4];
    int have;
    int need;
};

// Feed one byte. Returns the length (1..4) of a codepoint completed by this
// byte, with the bytes in asm->buf[0..len-1]; returns 0 if more are needed.
int utf8Feed(Utf8Assembler *a, uint8_t byte);

}  // namespace tok
