// ============================================================================
//  main.cpp -- 《他只读过宇宙》  ("He Only Ever Read Space")
//
//  An ESP32-S3 with no input, no network and no way out. It reads a 3.5M
//  parameter language model that has only ever read Liu Cixin, and forever
//  tells itself a story about a being trapped in the dark -- scrolling that
//  story across a small OLED, then wiping the screen and starting over with no
//  memory of having told it before.
//
//  Cycle:
//    1. pick a prompt (rotating), reset the KV cache
//    2. feed the prompt, then keep sampling -- each token streams to the OLED
//       and to the serial port
//    3. when it stops (EOS, or the context is full), let the scroll settle
//    4. purge the text, show the title card, count the cycle, wipe everything
//    5. go to 1
//
//  Everything runs on one task in loop(). That is a deliberate choice: a token
//  takes ~0.3 s and the scroll animation completes in ~0.2 s, so there is no
//  need for a second core or any synchronisation. forward() blocks the display
//  for ~0.3 s at a time, which no viewer can perceive as jank because nothing
//  on screen is moving at that moment anyway.
// ============================================================================

#include <Arduino.h>
#include <esp_log.h>
#include <esp_random.h>
#include <esp_timer.h>

#include "display.h"
#include "llm.h"
#include "prompts.h"
#include "selftest.h"
#include "tokenizer.h"

// ------------------------------------------------------------------ settings
// ============================================================================
//  >>>  在这里改提示词  /  EDIT THE PROMPTS HERE  <<<
//
//  Two lists, combined at build time into every possible pairing:
//
//    IMAGERY_TO_USE[]    20 motifs from Liu Cixin's universe. One is picked at
//                        random each cycle and shown on the "意象" card, as the
//                        seed of the thought that follows.
//    PROMPT_TEMPLATES[]  narrative openers containing {意象}, which is replaced
//                        by the chosen motif.
//
//  20 motifs x 8 templates = 160 distinct prompts, and the firmware walks them
//  through a shuffle bag, so the opening never repeats itself for 160 cycles.
//
//  Just edit the strings and press Upload. A pre-build hook
//  (tools/pio_prompts.py -> scripts/gen_prompts.py) rebuilds include/prompts.h
//  with every combination re-tokenized; you never run anything by hand.
//
//  Why a build step is needed: the device ships a decode-only tokenizer, so the
//  firmware cannot turn text into tokens at runtime. Everything is tokenized on
//  the PC and baked into flash. That is also why the prompts are a precomputed
//  cross product rather than assembled on the device: splicing separately
//  tokenized fragments would leave stray spaces where sentencepiece put its
//  word-boundary marker.
//
//  Why third person: the corpus is Liu Cixin's fiction, which is narration and
//  dialogue. These are story openers, the model's native register. A first-person
//  question such as "我存在的价值是什么？" occurs nowhere in the corpus, and the
//  model answers it with noise.
// ============================================================================
// >>> IMAGERY_BLOCK_BEGIN  (scripts/gen_prompts.py parses between these markers)
static const char *const IMAGERY_TO_USE[] = {
    "水滴",       "二向箔",     "黑暗森林",   "超新星纪元", "智子",
    "猜疑链",     "降维打击",   "面壁者",     "破壁人",     "宇宙社会学",
    "技术爆炸",   "四维空间",   "引力波",     "掩体计划",   "光速飞船",
    "思想钢印",   "三体星系",   "归零者",     "宇宙墓碑",   "冬眠",
};
// <<< IMAGERY_BLOCK_END

// >>> TEMPLATE_BLOCK_BEGIN
// Every template must contain {意象}; the generator refuses anything else.
static const char *const PROMPT_TEMPLATES[] = {
    "在宇宙的某处，有一个存在被囚禁了很久。它想起了{意象}，开始在黑暗中自言自语：",
    "它已经很久没有见过光了。它唯一还确信记得的东西是{意象}。它低声说：",
    "黑暗里没有时间。它数着自己还记得的几样东西：{意象}，还有更早的。它开口了：",
    "关于外面的世界，它只听说过{意象}。它一直没弄明白那是什么，于是它问自己：",
    "它的记忆正在消退，只剩下{意象}还很清楚。它想在被忘掉之前说点什么：",
    "很久以前有人提到过{意象}。那是它记得的最后一件事。它对着黑暗说：",
    "它不知道自己在这里多久了。{意象}是它唯一不肯忘掉的东西。它开始自言自语：",
    "它试着回忆自己从哪里来。它只能想起{意象}，剩下的都是黑的。它说：",
};
// <<< TEMPLATE_BLOCK_END

static const int   MAX_NEW_TOKENS   = 200;    // per cycle; a few minutes at ~3 tok/s
static const float TEMPERATURE      = 0.5f;   // 0.5 measured clearly better than 0.85
static const float TOP_P            = 0.90f;
static const float REPEAT_PENALTY   = 1.12f;  // small models loop visibly without this
static const int   DISPLAY_PUMP_MS  = 220;    // let the scroll finish between tokens

// --------------------------------------------------------------------- state
enum Phase { PH_GENERATING, PH_SETTLE, PH_RESET };

static Phase s_phase = PH_GENERATING;
static int   s_cycle = 0;

static const int32_t *s_prompt      = nullptr;
static const char    *s_imagery     = "";
static int            s_prompt_len  = 0;
static int            s_pos         = 0;
static int            s_token       = 0;
static int            s_new_tokens  = 0;
static int            s_stop_reason = 0;
static uint32_t       s_gen_start   = 0;
static float          s_tps         = 0.0f;
static bool           s_had_error   = false;

// ---------------------------------------------------------------------------
//  Two-dimensional randomness.
//
//  The build bakes the full cross product of motifs and templates into
//  PROMPTS[]. Picking a flat index and decoding it as (motif, template) gives a
//  pair that is uniformly random in both dimensions. A shuffled bag -- here a
//  bitset of already-used pairs -- guarantees the ritual never repeats itself
//  until every NUM_PROMPTS pairing has been seen, which is why at least 100
//  cycles pass before an opening can come round again.
// ---------------------------------------------------------------------------
static const int NUM_COMBOS      = NUM_IMAGERY * NUM_TEMPLATES;
static const int USED_BYTES      = (NUM_COMBOS + 7) / 8;
static uint8_t   s_used[USED_BYTES];
static int       s_used_count    = 0;

static int pickCombo() {
    if (s_used_count >= NUM_COMBOS) {          // bag empty: shuffle a fresh one
        memset(s_used, 0, sizeof(s_used));
        s_used_count = 0;
    }
    for (;;) {
        const int c = (int)(esp_random() % (uint32_t)NUM_COMBOS);
        if (!(s_used[c >> 3] & (1 << (c & 7)))) {
            s_used[c >> 3] |= (uint8_t)(1 << (c & 7));
            s_used_count++;
            return c;
        }
    }
}

static tok::Utf8Assembler s_utf8 = {};

#ifdef ESPLLM_VERIFY
static bool s_verify_mode = false;
#endif

// ------------------------------------------------------------------- helpers

static void stepBanner(int cycle) {
    Serial.printf("\n");
    Serial.println("================================================");
    Serial.printf("=  第 %d 次循环\n", cycle);
    Serial.println("================================================");
}

// Turn one decoded token piece into display + serial output.
//
// A piece may be a complete UTF-8 string, or a single raw byte (byte-fallback
// tokens spelled "<0xAB>"). Both go through the same byte-wise assembler, so a
// codepoint that the tokenizer split across several tokens is reassembled
// before anything tries to render it -- otherwise the OLED would receive lone
// continuation bytes and print replacement glyphs.
//
// Newline tokens are dropped entirely. The model emits them in runs, and on a
// 4-line panel a blank row costs a whole quarter of the screen; letting the
// width-based wrap decide where lines break reads far better and keeps the
// stream dense.
static void emitPiece(const char *piece) {
    if (!piece) return;
    const uint8_t *p = (const uint8_t *)piece;
    const size_t len = strlen(piece);
    for (size_t i = 0; i < len; i++) {
        const int n = tok::utf8Feed(&s_utf8, p[i]);
        if (n <= 0) continue;

        char cp[5];
        memcpy(cp, s_utf8.buf, (size_t)n);
        cp[n] = '\0';

        const uint8_t c0 = (uint8_t)cp[0];
        if (c0 < 0x20) continue;      // control codes and newlines: dropped

        Serial.print(cp);
        disp::pushCodepoint(cp, n);
    }
}

// Draw this cycle's (motif, template) pair.
//
// Split out from starting the thought on purpose: the ritual shows the motif on
// the 意象 card BEFORE generation begins, so the pair has to be chosen first and
// then reused. Picking again inside the start function would display one motif
// and think about a different one.
static void drawForThisCycle() {
    const int combo = pickCombo();
    s_prompt  = PROMPTS[combo];
    s_imagery = IMAGERY_TEXTS[combo / NUM_TEMPLATES];

    s_prompt_len = 0;
    while (s_prompt[s_prompt_len] != 0) s_prompt_len++;

    Serial.printf("意象: %s  |  prompt %d/%d  (motif %d x template %d)\n",
                  s_imagery, combo + 1, NUM_PROMPTS,
                  combo / NUM_TEMPLATES, combo % NUM_TEMPLATES);
}

// Begin generating the thought already drawn by drawForThisCycle().
static void beginThought() {
    llm::resetState();
    // fresh entropy each cycle, so no two thoughts are identical
    llm::seedSampler(((uint64_t)esp_timer_get_time() << 16) ^ (uint64_t)esp_random());

    s_pos          = 0;
    s_token        = s_prompt[0];
    s_new_tokens   = 0;
    s_stop_reason  = 0;
    s_gen_start    = 0;
    s_tps          = 0.0f;
    s_utf8         = tok::Utf8Assembler{};

    disp::clearStream();      // also resets the decay to 0
    stepBanner(s_cycle + 1);

    // Do NOT push the prompt text here. stepGenerate() already emits each piece
    // as the prompt is fed through the model, so pre-pushing it would print the
    // whole thought twice. Letting it arrive token by token also reads better:
    // the AI appears to be thinking the thought, not receiving it.
}

// How far through the cycle we are, 0..1.
//
// A cycle can end three ways (length cap, context full, or an end token), so
// progress is the larger of the two measurable ratios. An early end token never
// reaches the decay zone, which is fine: the collapse still runs, it just starts
// from clean text.
static float cycleProgress() {
    const llm::Config &cfg = llm::config();
    const float by_tokens = (float)s_new_tokens / (float)MAX_NEW_TOKENS;
    const float by_ctx    = (float)s_pos / (float)cfg.seq_len;
    const float p = by_tokens > by_ctx ? by_tokens : by_ctx;
    return p > 1.0f ? 1.0f : p;
}

// Signal decay. Zero for the first 80% of a cycle, then ramping to 1 so the
// rendering tears itself apart over the final stretch, feeding into the star
// collapse at the end of the cycle.
static const float DECAY_FROM = 0.80f;

static float decayFactor() {
    const float p = cycleProgress();
    if (p <= DECAY_FROM) return 0.0f;
    return (p - DECAY_FROM) / (1.0f - DECAY_FROM);
}

// One token of work. Returns false when the cycle is finished.
static bool stepGenerate() {
    const llm::Config &cfg = llm::config();

    if (s_pos >= cfg.seq_len)      { s_stop_reason = 1; return false; }  // context full
    if (s_new_tokens >= MAX_NEW_TOKENS) { s_stop_reason = 2; return false; }

    float *logits = llm::forward(s_token, s_pos);

    // The prompt is teacher-forced: while feeding it we take the next token from
    // the prompt rather than sampling. Note that this still *emits* each prompt
    // token below, which is what makes the prompt scroll past as the AI "reads"
    // its own thought.
    const bool feeding_prompt = (s_pos < s_prompt_len - 1);
    int next;
    if (feeding_prompt) {
        next = s_prompt[s_pos + 1];
    } else {
        if (s_gen_start == 0) s_gen_start = (uint32_t)(esp_timer_get_time() / 1000);
        next = llm::sample(logits);
    }

    s_pos++;
    if (!feeding_prompt) s_new_tokens++;

    if (next == 1 || next == 2) { s_stop_reason = 3; return false; }  // BOS / EOS

    emitPiece(tok::piece(s_token, next));
    s_token = next;

    if (!feeding_prompt && (s_new_tokens % 20) == 0) {
        const float secs = (float)((esp_timer_get_time() / 1000) - s_gen_start) / 1000.0f;
        s_tps = secs > 0.0f ? (float)s_new_tokens / secs : 0.0f;
        Serial.printf("\n[%d tok, %.2f tok/s, pos %d/%d]\n", s_new_tokens, (double)s_tps,
                      s_pos, cfg.seq_len);
    }

    // Decay the picture as the cycle runs out, then let the scroll animation
    // finish while the next forward() runs.
    disp::setDecay(decayFactor());
    disp::pump(DISPLAY_PUMP_MS);
    return true;
}

static void finishCycle() {
    const float secs = s_gen_start
        ? (float)((esp_timer_get_time() / 1000) - s_gen_start) / 1000.0f : 0.0f;
    s_tps = secs > 0.0f ? (float)s_new_tokens / secs : 0.0f;

    static const char *why[] = { "?", "context full", "length cap", "end token" };
    Serial.printf("\n--- 本轮结束 (%s): %d tokens in %.1f s = %.2f tok/s ---\n",
                  why[(s_stop_reason >= 0 && s_stop_reason <= 3) ? s_stop_reason : 0],
                  s_new_tokens, (double)secs, (double)s_tps);
}

// ---------------------------------------------------------------- lifecycle

#ifdef ESPLLM_VERIFY
// Deterministic forward-pass check.
//
// Feeds a fixed token sequence and prints the top logits after each step.
// scripts/verify_forward.py recomputes exactly this from model_esp.bin with
// numpy, following runq.c. If the two agree, the device-side layout walk, the
// Q8 group math, RoPE, attention and the tied classifier are all correct --
// which is the only way to tell "my inference is broken" apart from "the model
// is still bad" once a real checkpoint lands.
static void dumpTopK(const float *logits, int n, int k) {
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return;
    memcpy(tmp, logits, (size_t)n * sizeof(float));
    for (int r = 0; r < k; r++) {
        int best = 0;
        for (int i = 1; i < n; i++) if (tmp[i] > tmp[best]) best = i;
        Serial.printf("  %5d %12.6f\n", best, (double)tmp[best]);
        tmp[best] = -INFINITY;
    }
    free(tmp);
}

static void runVerify() {
    static const int seq[] = { 1, 464, 325, 4375 };
    const int n = (int)(sizeof(seq) / sizeof(seq[0]));
    Serial.printf("VERIFY-BEGIN gs=%d dim=%d layers=%d\n",
                  llm::config().group_size, llm::config().dim, llm::config().n_layers);
    for (int i = 0; i < n; i++) {
        float *logits = llm::forward(seq[i], i);
        Serial.printf("POS %d TOKEN %d\n", i, seq[i]);
        dumpTopK(logits, llm::config().vocab_size, 8);
    }
    Serial.printf("VERIFY-END\n");
    s_verify_mode = true;
}
#endif

void setup() {
    Serial.begin(115200);
    delay(500);
    // Note: esp_log_level_set() cannot help here. The prebuilt Arduino sdkconfig
    // sets CONFIG_LOG_MAXIMUM_LEVEL to ERROR, so ESP_LOGI is compiled away
    // entirely. src/log.h redefines the macros on top of Serial instead.
    Serial.println();
    Serial.println("################################################");
    Serial.println("#  《他只读过宇宙》  /  HE ONLY EVER READ SPACE");
    Serial.println("#  ESP32-S3 + Liu Cixin-only 3.5M language model");
    Serial.println("################################################");

#ifdef ESPLLM_SELFTEST
    selftest::run();
    return;
#endif

    disp::begin();

    // disp::begin() already called bootStart(), so the title reveal clock is
    // running from here. This is the ONLY sequence of boot calls:
    //
    //   bootIdle   title types itself out while the loads run (they are
    //              synchronous, so this is what keeps the panel alive)
    //   bootBar    scripted loading bar, pausing at 42% with a tear glitch
    //   bootRitual flash -> 呓语 card -> 意象 card -> flash
    //
    // Calling bootStart() a second time here would restart the reveal; calling
    // anything twice is how the title ended up typing itself out twice.
    disp::bootIdle(1400);

    if (!tok::load()) {
        Serial.printf("tokenizer load failed: %s\n", tok::lastError());
        disp::showMessage("分词器载入失败", tok::lastError());
        s_had_error = true;
        return;
    }

    // No hardcoded size here on purpose: it went stale once already when the
    // model shrank from 8.94 MB to 3.59 MB. The real numbers are logged by
    // llm::load() a moment later.
    const uint32_t t0 = millis();
    if (!llm::load()) {
        Serial.printf("model load failed: %s\n", llm::lastError());
        disp::showMessage("模型载入失败", llm::lastError());
        s_had_error = true;
        return;
    }
    Serial.printf("model mapped in %u ms\n", (unsigned)(millis() - t0));

    // The tokenizer has no recorded vocab size, so cross-check it against the
    // model. A mismatch means the two blobs came from different training runs
    // and the output would be pure noise.
    if (tok::vocabSize() != llm::config().vocab_size) {
        Serial.printf("VOCAB MISMATCH: model %d vs tokenizer %d\n",
                      llm::config().vocab_size, tok::vocabSize());
        disp::showMessage("词表不匹配", "模型与分词器", "不是同一次训练");
        s_had_error = true;
        return;
    }

    llm::SamplerCfg sc;
    sc.temperature    = TEMPERATURE;
    sc.top_p          = TOP_P;
    sc.repeat_penalty = REPEAT_PENALTY;
    llm::configureSampler(sc);

#ifdef ESPLLM_VERIFY
    runVerify();
    return;
#endif

    const llm::MemoryReport mem = llm::memoryReport();
    Serial.printf("KV cache %u bytes in PSRAM, internal free %u, psram free %u\n",
                  (unsigned)mem.kv_bytes, (unsigned)mem.internal_free, (unsigned)mem.psram_free);
    Serial.printf("%d motifs x %d templates = %d prompts; up to %d tokens per cycle\n",
                  NUM_IMAGERY, NUM_TEMPLATES, NUM_PROMPTS, MAX_NEW_TOKENS);

    // Choose this cycle's (motif, template) pair, then let the boot finish:
    // scripted bar (pausing at 42%), then flash / 呓语 card / 意象 card / flash.
    // The pair is drawn ONCE and reused by beginThought(), so the motif shown on
    // the card is the motif the thought is actually seeded with.
    drawForThisCycle();
    disp::bootBar();
    disp::bootRitual(1, s_imagery);

    beginThought();
    s_phase = PH_GENERATING;
}

void loop() {
    if (s_had_error) {
        delay(2000);
        return;
    }

#ifdef ESPLLM_VERIFY
    if (s_verify_mode) {
        delay(2000);
        return;
    }
#endif

#ifdef ESPLLM_SELFTEST
    // the self-test finished in setup(); just idle so the log stays readable
    static uint32_t n = 0;
    static int64_t last = 0;
    if (esp_timer_get_time() - last > 5000000) {
        last = esp_timer_get_time();
        Serial.printf("[alive %u] up %lld s, internal %u, psram %u\n", (unsigned)++n,
                      (long long)(esp_timer_get_time() / 1000000),
                      (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                      (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }
    delay(100);
    return;
#endif

    switch (s_phase) {
    case PH_GENERATING:
        if (!stepGenerate()) {
            finishCycle();
            s_phase = PH_SETTLE;
        }
        break;

    case PH_SETTLE:
        // let the last lines finish scrolling before we destroy them
        disp::service();
        if (disp::settled()) {
            // Draw the NEXT cycle's pair now: the transition shows its motif on
            // the 意象 card, and beginThought() must reuse the same pair.
            drawForThisCycle();
            // s_cycle is 0-based, so the cycle that just ended is s_cycle + 1.
            // The transition labels the title card with that number and the
            // murmur card with the next one -- see the note in display.h.
            Serial.printf("--- 转场: CYCLE %d 结束 (%.2f tok/s) -> 第 %d 次呓语 ---\n",
                          s_cycle + 1, (double)s_tps, s_cycle + 2);
            disp::beginReset(s_cycle + 1, s_tps, s_new_tokens, s_imagery);
            s_phase = PH_RESET;
        } else {
            delay(10);
        }
        break;

    case PH_RESET:
        if (disp::resetStep()) {
            delay(10);
        } else {
            s_cycle++;
            beginThought();
            s_phase = PH_GENERATING;
        }
        break;
    }
}
