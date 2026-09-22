// ============================================================================
//  display.cpp -- see display.h
// ============================================================================

#include "display.h"

#include <Arduino.h>
#include <U8g2lib.h>
#include <esp_random.h>
#include <string.h>
#include <stdio.h>

namespace disp {

// SH1106 128x64, full framebuffer, hardware I2C.
// ctor: (rotation, reset, clock, data). U8g2 forwards clock/data to Wire.begin()
// on ESP32, so this is what wires GPIO3/GPIO8 up.
static U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, 3, 8);

#define FONT_CN u8g2_font_wqy14_t_gb2312   // 14px, GB2312 coverage, ~247 KB in flash
#define FONT_EN u8g2_font_6x10_tf

// ---------------------------------------------------------------------------
//  Vertical metrics -- this is the first-line-clipping fix.
//
//  u8g2.getAscent() returns font_info.ascent_A, the ascent of the LATIN letter
//  'A'. In u8g2_font_wqy14_t_gb2312 that is 9, but the font's header reports
//  max_char_height = 18: Chinese glyphs are much taller. Laying text out with a
//  baseline of 9 pushed the top of every Chinese character 3 px above the top of
//  the screen -- the reported clip, and it only ever affected the FIRST line,
//  because line j's ink top is j*16 + 9 - 12 = j*16 - 3, which is only negative
//  for j = 0.
//
//  U8g2 places a glyph as (u8g2_font_decode_glyph):
//      target_y -= glyph_height + char_y;    top = baseline - (h + char_y)
//  Typical CJK glyphs here are h = 14, char_y = -2, so ink reaches 12 px above
//  the baseline. Baseline 13 puts each row's ink in rows 1..15 of its 16 px slot.
// ---------------------------------------------------------------------------
static const int LINE_H          = 16;
static const int BASELINE_OFFSET = 13;
static const int INK_ABOVE_BASE  = 12;   // CJK ink extent above the baseline
static const int VISIBLE         = 4;
static const int MAX_LINES       = 200;
static const int LINE_BYTES      = 64;
static const int WRAP_PX         = 126;

static char  s_lines[MAX_LINES][LINE_BYTES];
static int   s_nlines   = 0;
static char  s_cur[LINE_BYTES];
static int   s_cur_len  = 0;

static float s_scroll   = 0.0f;
static float s_target   = 0.0f;
static bool  s_ready    = false;
static bool  s_dirty    = true;

// Signal decay, 0..1, driven by the caller from how far through the cycle we are.
// At 0 the stream renders cleanly; as it rises the text jitters sideways, whole
// rows get torn out, and snow increasingly covers the panel until nothing can be
// read. The star-collapse transition then consumes what is left.
static float s_decay = 0.0f;

// Cycle-transition state. Declared up here because service() needs to know a
// transition is running in order to stay out of the way.
static bool     s_rst_active = false;
static uint32_t s_rst_t0     = 0;
// Number of the cycle that just FINISHED (1-based). The title card labels its
// throughput figures with this; the murmur card uses s_rst_done + 1.
static int      s_rst_done   = 0;
static float    s_rst_tps    = 0.0f;
static int      s_rst_tokens = 0;
static const char *s_rst_imagery = "";

static const char TITLE[] = "他只读过宇宙";     // 6 CJK glyphs, 84 px wide
static const int  TITLE_GLYPHS = 6;

// ---------------------------------------------------------------------- stars
static const int N_STARS = 40;
struct Star { uint8_t x, y, step; };
static Star s_stars[N_STARS];

static void starsInit() {
    for (int i = 0; i < N_STARS; i++) {
        s_stars[i].x    = (uint8_t)(esp_random() % 128);
        s_stars[i].y    = (uint8_t)(esp_random() % 64);
        s_stars[i].step = (uint8_t)(1 + (esp_random() % 3));
    }
}

static void starsStep() {
    for (int i = 0; i < N_STARS; i++) {
        s_stars[i].y = (uint8_t)(s_stars[i].y + s_stars[i].step);
        if (s_stars[i].y >= 64) {
            s_stars[i].y    = 0;
            s_stars[i].x    = (uint8_t)(esp_random() % 128);
            s_stars[i].step = (uint8_t)(1 + (esp_random() % 3));
        }
    }
}

static void starsDraw() {
    for (int i = 0; i < N_STARS; i++) u8g2.drawPixel(s_stars[i].x, s_stars[i].y);
}

// ------------------------------------------------------------------- helpers

static void drawCenteredCN(const char *s, int baseline) {
    u8g2.setFont(FONT_CN);
    u8g2.drawUTF8((128 - u8g2.getUTF8Width(s)) / 2, baseline, s);
}

static void drawCenteredEN(const char *s, int baseline) {
    u8g2.setFont(FONT_EN);
    u8g2.drawStr((128 - u8g2.getStrWidth(s)) / 2, baseline, s);
}

// Reveal the first `n` glyphs of a UTF-8 string (typewriter effect).
static void drawUTF8Prefix(const char *s, int n, int x, int y) {
    u8g2.setFont(FONT_CN);
    const uint8_t *p = (const uint8_t *)s;
    int shown = 0, cx = x;
    while (*p && shown < n) {
        int len;
        if (*p < 0x80) len = 1;
        else if ((*p & 0xE0) == 0xC0) len = 2;
        else if ((*p & 0xF0) == 0xE0) len = 3;
        else if ((*p & 0xF8) == 0xF0) len = 4;
        else { p++; continue; }
        char cp[5];
        memcpy(cp, p, (size_t)len);
        cp[len] = '\0';
        u8g2.drawUTF8(cx, y, cp);
        cx += u8g2.getUTF8Width(cp);
        p += len;
        shown++;
    }
}

// White flash, then black. The beat that separates one state from the next.
//
// Kept short on purpose: at 70/90 ms a two-shot flash ran for 320 ms, which read
// as a slow blink rather than a cut. These values come out at ~180 ms for two.
static void flash(int times, int on_ms = 40, int off_ms = 50) {
    for (int i = 0; i < times; i++) {
        u8g2.clearBuffer();
        u8g2.drawBox(0, 0, 128, 64);
        u8g2.sendBuffer();
        delay(on_ms);
        u8g2.clearBuffer();
        u8g2.sendBuffer();
        delay(off_ms);
    }
}

// The murmur card: "第 N 次呓语", then half a second later the 意象 line.
//
// `murmur_no` is the number of the murmur about to START, which is one more than
// the cycle that just finished -- the title card shows the finished cycle's
// stats while this card announces the next thought. Passing the finished cycle
// number here made every transition announce the previous murmur.
static void drawMurmur(int murmur_no, const char *imagery, bool showImagery) {
    u8g2.clearBuffer();
    char buf[64];
    snprintf(buf, sizeof(buf), "第 %d 次呓语", murmur_no);
    drawCenteredCN(buf, 26);

    if (showImagery && imagery && imagery[0]) {
        snprintf(buf, sizeof(buf), "意象：%s", imagery);
        drawCenteredCN(buf, 48);
        // a hairline under the card, so it reads as a card and not as body text
        u8g2.drawHLine(14, 54, 100);
    }
    u8g2.sendBuffer();
}

// ------------------------------------------------------------------ text stream

static int totalLines() { return s_nlines + (s_cur_len > 0 ? 1 : 0); }

static const char *lineAt(int j) {
    if (j < 0) return nullptr;
    if (j < s_nlines) return s_lines[j];
    if (j == s_nlines && s_cur_len > 0) return s_cur;
    return nullptr;
}

static void updateTarget() {
    const int total = totalLines();
    s_target = (total > VISIBLE) ? (float)(total - VISIBLE) * (float)LINE_H : 0.0f;
}

static void redraw() {
    u8g2.clearBuffer();
    u8g2.setFont(FONT_CN);
    const int j0 = (int)(s_scroll / (float)LINE_H);

    // Decay drives three effects that compound: sideways jitter, rows chewed
    // out, and snow. Squaring the snow keeps the early part of the falloff
    // subtle -- the piece should degrade, not switch off.
    const int jitter = (int)(s_decay * 6.0f);
    const int tear   = (int)(s_decay * s_decay * 26.0f);

    for (int j = j0; j < j0 + VISIBLE + 1; j++) {
        const char *t = lineAt(j);
        if (!t) continue;
        const int base = (int)((float)j * (float)LINE_H - s_scroll) + BASELINE_OFFSET;
        // Draw a row if any of its ink (baseline-12 .. baseline+2) is on screen.
        if (base - INK_ABOVE_BASE >= 64 || base + 3 < 0) continue;

        int dx = 0;
        if (jitter > 0) {
            dx = (int)(esp_random() % (uint32_t)(2 * jitter + 1)) - jitter;
        }
        u8g2.drawUTF8(dx, base, t);

        // bite a chunk out of the row -- the "shattering" read
        if (tear > 0 && (esp_random() & 3) == 0) {
            const int w = 18 + (int)(esp_random() % (uint32_t)tear);
            const int x = (int)(esp_random() % 128);
            u8g2.setDrawColor(0);
            u8g2.drawBox(x, base - INK_ABOVE_BASE, w, 15);
            u8g2.setDrawColor(1);
        }
    }

    // snow
    const int snow = (int)(s_decay * s_decay * 520.0f);
    for (int i = 0; i < snow; i++) {
        u8g2.drawPixel((int)(esp_random() % 128), (int)(esp_random() % 64));
    }

    u8g2.sendBuffer();
}

void setDecay(float d) {
    if (d < 0.0f) d = 0.0f;
    if (d > 1.0f) d = 1.0f;
    if (d != s_decay) {
        s_decay = d;
        s_dirty = true;      // force a repaint even when the scroll has settled
    }
}

float decay() { return s_decay; }

// ------------------------------------------------------------- boot sequence
//
// The bar is a scripted animation, not a measurement. Two things forced this:
//
//  1. tok::load() and llm::load() are synchronous. While they run, nothing is
//     drawn at all, so a bar wired to real progress simply freezes mid-frame --
//     which is why it appeared to stall hard at 84% and never paused at 42%.
//  2. A bar that follows the clock can pause wherever we like, which is the
//     entire point of the beat at 42%.
//
// The script still tells the truth about the sequence (init -> weights -> ready);
// it just is not driven by it. bootIdle() is what covers the real loading.
static const uint32_t REVEAL_STEP_MS = 240;   // one glyph of the title

// The Answer, and the glitch that goes with it.
static const float ANSWER_PCT = 42.0f;

// Bar curve: (ms, percent). Uneven on purpose -- a linear ramp reads as fake,
// and real loaders always lurch. The two knots at 42 are the hold.
struct Knot { uint32_t t; float pct; };
static const Knot BAR_CURVE[] = {
    {    0,   0.0f },
    {  980,  19.0f },
    { 1650,  31.0f },
    { 2020,  42.0f },
    { 2940,  42.0f },   // <- the Answer: 500 ms of stillness, THEN the glitch
    { 3660,  63.0f },
    { 4220,  71.0f },
    { 4800,  88.0f },
    { 5360, 100.0f },
};
static const int    N_KNOTS    = (int)(sizeof(BAR_CURVE) / sizeof(BAR_CURVE[0]));
static const uint32_t BOOT_BAR_MS = BAR_CURVE[N_KNOTS - 1].t;

// The Answer, in two distinct beats.
//
// The pause has to come FIRST and be completely empty -- freezing at 42% is the
// joke, and filling that silence with effects throws it away. Only once the half
// second is up does the picture start to break: a hard white blink, then the
// tearing burst. Starting the glitch the instant the bar lands meant there was
// no pause at all, just an effect.
static const uint32_t ANSWER_T0   = 2020;                     // bar reaches 42%
static const uint32_t ANSWER_HOLD = 500;                      // ...and nothing happens
static const uint32_t ANSWER_TEAR = 420;                      // ...then it breaks
static const uint32_t ANSWER_T1   = ANSWER_T0 + ANSWER_HOLD + ANSWER_TEAR;

static uint32_t s_boot_t0 = 0;
static float    s_scan_x  = 0.0f;

static float barValueAt(uint32_t t) {
    if (t <= BAR_CURVE[0].t) return BAR_CURVE[0].pct;
    for (int i = 1; i < N_KNOTS; i++) {
        if (t <= BAR_CURVE[i].t) {
            const Knot &a = BAR_CURVE[i - 1];
            const Knot &b = BAR_CURVE[i];
            const float f = (float)(t - a.t) / (float)(b.t - a.t);
            return a.pct + (b.pct - a.pct) * f;
        }
    }
    return BAR_CURVE[N_KNOTS - 1].pct;
}

static const char *statusFor(float pct) {
    if (pct < 20.0f) return "initialising";
    if (pct < 42.0f) return "reading weights";
    if (pct < 64.0f) return "linking tokenizer";
    if (pct < 89.0f) return "warming up";
    return "ready";
}

// A torn-scanline glitch: bands of the frame are erased and redrawn displaced.
// strength 1..3, scaled by how deep into the burst we are.
static void tearOverlay(int strength) {
    for (int i = 0; i < strength * 3; i++) {
        const int y     = (int)(esp_random() % 64);
        const int h     = 1 + (int)(esp_random() % 3);
        const int shift = (int)(esp_random() % 11) - 5;
        u8g2.setDrawColor(0);
        u8g2.drawBox(0, y, 128, h);
        u8g2.setDrawColor(1);
        if (shift >= 0) u8g2.drawBox(shift, y, 128 - shift, h);
        else            u8g2.drawBox(0, y, 128 + shift, h);
    }
    // a few bright specks, like dropped signal
    const int specks = strength * 14;
    for (int i = 0; i < specks; i++) {
        u8g2.drawPixel((int)(esp_random() % 128), (int)(esp_random() % 64));
    }
    u8g2.setDrawColor(1);
}

void begin() {
    // Do NOT call Wire.begin() here. U8g2 calls it inside u8g2.begin(), and
    // re-initialising an already-started ESP32 Wire instance hangs the board
    // with no output at all.
    u8g2.begin();
    u8g2.setContrast(255);
    u8g2.setFont(FONT_CN);
    s_ready = true;
    bootStart();
}

bool ready() { return s_ready; }

void bootStart() {
    s_boot_t0 = millis();
    s_scan_x  = 0.0f;
    starsInit();
}

// Title vertical positions. The title is ONE thing that moves: it sits centred
// while the model loads, then slides up to sit above the bar. Drawing it fresh
// at each position made it look like two separate titles appearing.
static const int TITLE_Y_CENTER = 34;
static const int TITLE_Y_TOP    = 15;
static const uint32_t SLIDE_MS  = 620;

// Draw the title (and its frame) at any baseline, with an optional scan line
// underneath whose length can be shrunk to fade it out.
static void drawTitleAt(int ty, int scan_len) {
    u8g2.setFont(FONT_CN);
    const int tw = u8g2.getUTF8Width(TITLE);
    const int tx = (128 - tw) / 2;
    u8g2.drawUTF8(tx, ty, TITLE);
    u8g2.drawFrame(tx - 5, ty - 15, tw + 10, 19);

    if (scan_len > 0) {
        const int y = ty + 8;
        const int x0 = (128 - scan_len) / 2;
        u8g2.drawHLine(x0, y, scan_len);
        const int sx = x0 + (int)s_scan_x % (scan_len > 4 ? scan_len : 1);
        u8g2.drawBox(sx > 124 ? 124 : sx, y, 4, 1);
    }
}

// Title + stars, no bar. Used before the loads so the panel is alive while the
// CPU is busy, and to hold the title long enough to read.
static void idleFrame(uint32_t elapsed) {
    const int shown = (int)(elapsed / REVEAL_STEP_MS) + 1;
    const int reveal = shown > TITLE_GLYPHS ? TITLE_GLYPHS : shown;

    s_scan_x += 4.0f;
    if (s_scan_x > 128.0f) s_scan_x = 0.0f;
    starsStep();

    u8g2.clearBuffer();
    starsDraw();

    u8g2.setFont(FONT_CN);
    const int tw = u8g2.getUTF8Width(TITLE);
    const int tx = (128 - tw) / 2;
    drawUTF8Prefix(TITLE, reveal, tx, TITLE_Y_CENTER);
    if (reveal >= TITLE_GLYPHS) {
        u8g2.drawFrame(tx - 5, TITLE_Y_CENTER - 15, tw + 10, 19);
        const int sx = (int)s_scan_x;
        u8g2.drawHLine(0, TITLE_Y_CENTER + 8, 128);
        u8g2.drawBox(sx > 124 ? 124 : sx, TITLE_Y_CENTER + 8, 4, 1);
    }
    u8g2.sendBuffer();
}

void bootIdle(uint32_t ms) {
    if (!s_ready) return;
    const uint32_t t0 = millis();
    for (;;) {
        const uint32_t e = millis() - t0;
        idleFrame(e);
        if (e >= ms) break;
        delay(30);
    }
}

// Layout, 128x64:
//    0..19   title (framed)
//   20..40   percentage, drawn at 2x so it is the first thing you see
//   44..53   the bar
//   54..62   status word
static void barFrame(uint32_t t) {
    const float pct = barValueAt(t);

    s_scan_x += 4.0f;
    if (s_scan_x > 128.0f) s_scan_x = 0.0f;
    starsStep();

    u8g2.clearBuffer();
    starsDraw();

    // Title, drawn COMPLETE -- no typewriter reveal here.
    //
    // The reveal belongs to bootIdle(), which runs first while the model loads.
    // Re-deriving it from this function's own clock restarted it at zero, so the
    // title typed itself out a second time the moment the bar appeared. Here it
    // has already slid up into place, so there is nothing left to animate.
    drawTitleAt(TITLE_Y_TOP, 0);

    // percentage, 2x
    char buf[12];
    snprintf(buf, sizeof(buf), "%d%%", (int)(pct + 0.5f));
    u8g2.setFont(FONT_EN);
    const int pw = u8g2.getStrWidth(buf) * 2;      // drawStrX2 doubles the advance
    u8g2.drawStrX2((128 - pw) / 2, 40, buf);

    // bar
    const int bx = 4, bw = 120, by = 45, bh = 9;
    u8g2.drawFrame(bx, by, bw, bh);
    const int fill = (int)((bw - 4) * pct / 100.0f);
    if (fill > 0) u8g2.drawBox(bx + 2, by + 2, fill, bh - 4);
    // a highlight on the leading edge, so the bar looks like it is advancing
    if (fill > 2 && fill < bw - 6) u8g2.drawBox(bx + 2 + fill - 2, by + 2, 2, bh - 4);

    // status
    u8g2.setFont(FONT_EN);
    u8g2.drawStr(4, 62, statusFor(pct));

    // ---- the Answer ----
    //
    // First 500 ms: the bar is frozen at 42% and nothing else is drawn on top of
    // it. Then, and only then, the signal breaks. The blink is a single white
    // frame, which is what makes it read as a flash rather than a fade.
    if (t >= ANSWER_T0 && t < ANSWER_T1) {
        const uint32_t into = t - ANSWER_T0;
        if (into >= ANSWER_HOLD) {
            const uint32_t e = into - ANSWER_HOLD;
            const uint32_t BLINK = 45;

            if (e < BLINK) {
                u8g2.clearBuffer();
                u8g2.drawBox(0, 0, 128, 64);
                u8g2.sendBuffer();
                return;
            }

            // ramp 1 -> 3 -> 1 across the tear so it reads as a burst
            const int k    = (int)(e - BLINK);
            const int span = (int)(ANSWER_TEAR - BLINK);
            const int half = span / 2;
            const int d    = k < half ? k : span - k;
            int strength = 1 + (half > 0 ? (d * 3) / half : 0);
            if (strength > 3) strength = 3;
            tearOverlay(strength);
        }
    }

    u8g2.sendBuffer();
}

// One frame of the slide. The bar is already drawn underneath at 0%, so the
// title visibly travels to its place above it rather than being replaced.
static void slideFrame(float f) {
    // smoothstep: slow out of the centre, settle gently at the top
    const float e = f * f * (3.0f - 2.0f * f);
    const int ty = TITLE_Y_CENTER + (int)((TITLE_Y_TOP - TITLE_Y_CENTER) * e);
    // the scan line retracts as the title rises
    const int scan_len = (int)(128.0f * (1.0f - e));

    s_scan_x += 4.0f;
    if (s_scan_x > 128.0f) s_scan_x = 0.0f;
    starsStep();

    u8g2.clearBuffer();
    starsDraw();
    drawTitleAt(ty, scan_len);

    // the empty bar, so the destination is already visible during the slide
    u8g2.drawFrame(4, 45, 120, 9);
    u8g2.setFont(FONT_EN);
    u8g2.drawStr(4, 62, "initialising");

    u8g2.sendBuffer();
}

void bootBar() {
    if (!s_ready) return;

    // Slide the title from the centre up to its place above the bar. This is the
    // single title moving, which is why it no longer appears to show up twice.
    const uint32_t s0 = millis();
    for (;;) {
        const uint32_t e = millis() - s0;
        const float f = (float)e / (float)SLIDE_MS;
        slideFrame(f > 1.0f ? 1.0f : f);
        if (e >= SLIDE_MS) break;
        delay(25);
    }

    const uint32_t t0 = millis();
    for (;;) {
        const uint32_t t = millis() - t0;
        if (t >= BOOT_BAR_MS) break;
        barFrame(t);
        delay(30);              // ~30 fps
    }
    barFrame(BOOT_BAR_MS);      // land exactly on 100
    delay(420);                 // let 100% register
}

void bootRitual(int cycle, const char *imagery) {
    if (!s_ready) return;

    flash(2);
    drawMurmur(cycle, imagery, false);
    delay(900);

    // the seed appears half a second after the line above it
    drawMurmur(cycle, imagery, true);
    delay(1500);
    flash(1);
}

void showMessage(const char *l1, const char *l2, const char *l3) {
    if (!s_ready) return;
    u8g2.clearBuffer();
    u8g2.setFont(FONT_CN);
    int y = 18;
    if (l1) { u8g2.drawUTF8(2, y, l1); y += 18; }
    if (l2) { u8g2.drawUTF8(2, y, l2); y += 18; }
    if (l3) { u8g2.drawUTF8(2, y, l3); }
    u8g2.sendBuffer();
}

// --------------------------------------------------------------- text stream

void breakLine() {
    if (s_nlines >= MAX_LINES) {
        // Drop the oldest half rather than losing the newest text.
        const int drop = MAX_LINES / 2;
        memmove(s_lines[0], s_lines[drop], (size_t)(MAX_LINES - drop) * LINE_BYTES);
        s_nlines -= drop;
        s_scroll -= (float)drop * (float)LINE_H;
        if (s_scroll < 0) s_scroll = 0;
    }
    memcpy(s_lines[s_nlines], s_cur, (size_t)s_cur_len + 1);
    s_nlines++;
    s_cur_len = 0;
    s_cur[0] = '\0';
    updateTarget();
    s_dirty = true;
}

void pushCodepoint(const char *cp, int len) {
    if (!s_ready || len <= 0 || len > 4) return;
    if (s_cur_len + len + 1 > LINE_BYTES) breakLine();

    memcpy(s_cur + s_cur_len, cp, (size_t)len);
    s_cur_len += len;
    s_cur[s_cur_len] = '\0';

    // Width-based wrap: CJK glyphs are 14 px, Latin ones narrower.
    u8g2.setFont(FONT_CN);
    if (u8g2.getUTF8Width(s_cur) > WRAP_PX) {
        s_cur_len -= len;
        s_cur[s_cur_len] = '\0';
        breakLine();
        memcpy(s_cur, cp, (size_t)len);
        s_cur_len = len;
        s_cur[s_cur_len] = '\0';
    }
    updateTarget();
    s_dirty = true;
}

void clearStream() {
    s_nlines  = 0;
    s_cur_len = 0;
    s_cur[0]  = '\0';
    s_scroll  = 0.0f;
    s_target  = 0.0f;
    s_decay   = 0.0f;          // a fresh cycle starts perfectly clean
    if (s_ready) {
        u8g2.clearBuffer();
        u8g2.sendBuffer();
    }
    s_dirty = true;
}

void pushString(const char *utf8) {
    if (!utf8) return;
    const uint8_t *p = (const uint8_t *)utf8;
    while (*p) {
        int len;
        if (*p < 0x80) len = 1;
        else if ((*p & 0xE0) == 0xC0) len = 2;
        else if ((*p & 0xF0) == 0xE0) len = 3;
        else if ((*p & 0xF8) == 0xF0) len = 4;
        else { p++; continue; }
        pushCodepoint((const char *)p, len);
        p += len;
    }
}

// ----------------------------------------------------------------- animation

void service() {
    if (!s_ready || s_rst_active) return;
    bool moved = false;
    if (s_scroll < s_target) {
        const float d = s_target - s_scroll;
        s_scroll += (d > 3.0f) ? 3.0f : d;
        if (s_scroll > s_target) s_scroll = s_target;
        moved = true;
    }
    if (moved || s_dirty) {
        redraw();
        s_dirty = false;
    }
}

bool settled() { return s_scroll >= s_target - 0.01f; }

void pump(uint32_t ms_budget) {
    const uint32_t t0 = millis();
    do {
        service();
        if (settled()) break;
        delay(4);
    } while (millis() - t0 < ms_budget);
}

// --------------------------------------------------------- cycle transition
//
// The end of a cycle. By this point the caller has driven setDecay() to 1, so the
// stream is already tearing apart; the transition then finishes the job by
// letting the star collapse. Timed beats, ~3.8 s:
//
//   0.00-0.43  IMPLODE  the decayed rows fall inward onto the centre line
//   0.43-0.68  CORE     a bright core shrinks while a shock ring expands
//   0.68-0.77  BOOM     detonation: full white
//   0.77-0.90  flash to black
//   0.90-1.95  TITLE    starfield + 《他只读过宇宙》 + cycle stats
//   1.95-2.09  flash
//   2.09-2.71  呓语     "第 N 次呓语"
//   2.71-3.23  意象     "意象：X" appears half a second later
//   3.23-3.37  flash
//   3.37-3.77  IRIS     reopens from a centre line
//
static const uint32_t T_IMPLODE = 430;
static const uint32_t T_CORE    = 250;
static const uint32_t T_BOOM    = 90;
static const uint32_t T_E2      = T_IMPLODE + T_CORE + T_BOOM;   // 770
static const uint32_t T_FLASH2  = T_E2 + 130;                    // 900
static const uint32_t T_TITLE   = T_FLASH2 + 1050;               // 1950
static const uint32_t T_FLASH3  = T_TITLE + 140;                 // 2090
static const uint32_t T_MURMUR  = T_FLASH3 + 620;                // 2710
static const uint32_t T_IMAGERY = T_MURMUR + 520;                // 3230
static const uint32_t T_FLASH4  = T_IMAGERY + 140;               // 3370
static const uint32_t T_IRIS    = T_FLASH4 + 400;                // 3770

static bool s_rst_cleared = false;

void beginReset(int finished_cycle, float tokens_per_s, int total_tokens,
                const char *imagery) {
    s_rst_active  = true;
    s_rst_t0      = millis();
    s_rst_done    = finished_cycle;
    s_rst_tps     = tokens_per_s;
    s_rst_tokens  = total_tokens;
    s_rst_imagery = imagery ? imagery : "";
    s_rst_cleared = false;

    starsInit();

    // Do NOT clear the text here: the collapse is supposed to consume the
    // decayed words, so they have to still be in the buffer. They are dropped
    // once the implosion is over.
}

static void drawTitleCard(int cycle, float tps, int tokens) {
    starsStep();
    u8g2.clearBuffer();
    starsDraw();

    u8g2.setFont(FONT_CN);
    const int tw = u8g2.getUTF8Width(TITLE);
    const int tx = (128 - tw) / 2;
    const int ty = 28;
    u8g2.drawUTF8(tx, ty, TITLE);
    u8g2.drawFrame(tx - 5, ty - 15, tw + 10, 19);

    char buf[48];
    snprintf(buf, sizeof(buf), "CYCLE %d", cycle);
    drawCenteredEN(buf, 44);
    snprintf(buf, sizeof(buf), "%.1f tok/s   %d tok", (double)tps, tokens);
    drawCenteredEN(buf, 56);
    u8g2.sendBuffer();
}

// Star collapse, stage 1: the rows rush toward the centre line and brighten as
// they compress, so the panel reads as matter falling into a point.
static void collapseImplode(uint32_t t) {
    const float f = 1.0f - (float)t / (float)T_IMPLODE;   // 1 -> 0
    const int squeeze = (int)((1.0f - f) * 14.0f);

    u8g2.clearBuffer();
    u8g2.setFont(FONT_CN);

    const int j0 = (int)(s_scroll / (float)LINE_H);
    uint32_t v = (uint32_t)(t * 2654435761u) | 1u;
    for (int j = j0; j < j0 + VISIBLE + 1; j++) {
        const char *txt = lineAt(j);
        if (!txt) continue;
        const int base = (int)((float)j * (float)LINE_H - s_scroll) + BASELINE_OFFSET;
        const int ny = 32 + (int)((float)(base - 32) * f);
        if (ny < -4 || ny > 68) continue;

        // sideways chaos grows as the squeeze tightens
        v = v * 1664525u + 1013904223u;
        const int dx = squeeze > 0 ? (int)(v % (uint32_t)(2 * squeeze + 1)) - squeeze : 0;
        u8g2.drawUTF8(dx, ny, txt);
    }

    // inbound debris, denser as it converges
    const int debris = 120 + (int)((1.0f - f) * 420.0f);
    for (int i = 0; i < debris; i++) {
        v = v * 1664525u + 1013904223u;
        const int x = (int)(v % 128);
        const int spread = (int)(32.0f * f) + 1;
        v = v * 1664525u + 1013904223u;
        const int y = 32 + (int)(v % (uint32_t)(2 * spread + 1)) - spread;
        if (y >= 0 && y < 64) u8g2.drawPixel(x, y);
    }
    u8g2.sendBuffer();
}

// Stage 2: the core shrinks to nothing while a shock ring races outward.
static void collapseCore(uint32_t u) {
    u8g2.clearBuffer();
    const int r = 30 - (int)(u * 28 / T_CORE);
    if (r > 0) {
        u8g2.drawDisc(64, 32, r);
        u8g2.drawDisc(64, 32, (r > 2) ? r - 2 : 1);
    }
    const int rr = 5 + (int)(u * 62 / T_CORE);
    u8g2.drawCircle(64, 32, rr);
    if (rr > 4) u8g2.drawCircle(64, 32, rr - 4);
    u8g2.sendBuffer();
}

static void collapseBoom() {
    u8g2.clearBuffer();
    u8g2.drawBox(0, 0, 128, 64);
    u8g2.sendBuffer();
}

bool resetStep() {
    if (!s_rst_active) return false;

    const uint32_t t = millis() - s_rst_t0;

    // ---- IMPLODE ----
    if (t < T_IMPLODE) {
        collapseImplode(t);
        delay(8);
        return true;
    }

    // ---- CORE ----
    if (t < T_IMPLODE + T_CORE) {
        collapseCore(t - T_IMPLODE);
        delay(10);
        return true;
    }

    // ---- BOOM ----
    if (t < T_E2) {
        collapseBoom();
        delay(12);
        // now that the words have been destroyed, drop them and the decay
        if (!s_rst_cleared) {
            s_nlines  = 0;
            s_cur_len = 0;
            s_cur[0]  = '\0';
            s_scroll  = 0.0f;
            s_target  = 0.0f;
            s_decay   = 0.0f;
            s_rst_cleared = true;
        }
        return true;
    }

    // ---- flash to black ----
    if (t < T_FLASH2) {
        u8g2.clearBuffer();
        u8g2.sendBuffer();
        delay(10);
        return true;
    }

    // ---- TITLE ----
    if (t < T_TITLE) {
        drawTitleCard(s_rst_done, s_rst_tps, s_rst_tokens);
        delay(40);
        return true;
    }

    // ---- flash ----
    if (t < T_FLASH3) {
        u8g2.clearBuffer();
        u8g2.drawBox(0, 0, 128, 64);
        u8g2.sendBuffer();
        delay(10);
        return true;
    }

    // ---- 呓语 ----
    if (t < T_MURMUR) {
        drawMurmur(s_rst_done + 1, s_rst_imagery, false);
        delay(20);
        return true;
    }

    // ---- 意象 ----
    if (t < T_IMAGERY) {
        drawMurmur(s_rst_done + 1, s_rst_imagery, true);
        delay(20);
        return true;
    }

    // ---- flash ----
    if (t < T_FLASH4) {
        u8g2.clearBuffer();
        u8g2.drawBox(0, 0, 128, 64);
        u8g2.sendBuffer();
        delay(10);
        return true;
    }

    // ---- IRIS reopen ----
    if (t < T_IRIS) {
        const int k = (int)((t - T_FLASH4) * 16 / 400);
        const int h = 1 + k * 4;
        u8g2.clearBuffer();
        u8g2.drawBox(0, 32 - h, 128, h * 2);
        u8g2.setDrawColor(0);
        u8g2.drawBox(0, 32 - h + 6, 128, h * 2 - 12);
        u8g2.setDrawColor(1);
        u8g2.sendBuffer();
        if (t < T_IRIS - 20) delay(12);
        return true;
    }

    u8g2.clearBuffer();
    u8g2.sendBuffer();
    s_rst_active = false;
    s_dirty = true;
    return false;
}

}  // namespace disp
