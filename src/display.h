// ============================================================================
//  display.h -- SH1106 OLED rendering for 《他只读过宇宙》.
//
//  128x64 panel, 4 lines of 14px Chinese text, smooth vertical scrolling of the
//  model's stream of thought. At the end of a cycle the text is erased and the
//  title card returns -- the reset is a real loss of context, which is the point.
//
//  The ritual
//  ----------
//    boot        title + starfield + loading bar, held long enough to read
//    flash       panel goes white, then black
//    murmur      "第 N 次呓语", then half a second later "意象：X"
//    flash
//    generation  the opening prompt streams in and the thought begins
//
//  I2C: SCL = GPIO3, SDA = GPIO8 (verified: panel answers at 0x3C).
// ============================================================================
#pragma once

#include <stdint.h>

namespace disp {

// ---- lifecycle ----
void begin();
bool ready();

// ---- boot sequence ----
//
// The loading bar is deliberately TIME-DRIVEN, not tied to the real load
// progress. The loads are synchronous and take a few hundred milliseconds, so a
// bar wired to them simply freezes mid-draw while the CPU is blocked -- which is
// exactly what happened when it was fed real percentages. The bar now narrates a
// fixed script (~5 s) with an uneven, believable curve.
//
// That is a deliberate fiction, but not a lie about anything in particular: the
// sequence it labels (init -> weights -> ready) really did happen, and it
// finishes before the bar runs. bootIdle() is what covers the actual loading.
//
//    bootStart()   reset and draw the first frame
//    bootIdle(ms)  animate title + starfield for ms (call this around the loads)
//    bootBar()     play the scripted loading bar, including the pause at 42%
//    bootRitual()  flash -> 呓语 card -> 意象 card -> flash
void bootStart();
void bootIdle(uint32_t ms);
void bootBar();
void bootRitual(int cycle, const char *imagery);

// Full-screen message for hard errors.
void showMessage(const char *l1, const char *l2 = nullptr, const char *l3 = nullptr);

// ---- text stream ----
void pushString(const char *utf8);              // split into codepoints, then push
void pushCodepoint(const char *cp, int len);    // cp is one complete UTF-8 char
void breakLine();
void clearStream();                             // discard all text, blank the panel

// ---- animation ----
void service();                    // one frame; cheap when nothing changed
void pump(uint32_t ms_budget);     // service() until settled or budget spent
bool settled();

// ---- decay ----
// 0..1. As it rises the rendered text jitters, tears and drowns in snow, until
// nothing is legible. The caller drives it from how far through the cycle we are
// (the last 20%), and the star-collapse transition takes over at 1.0.
void setDecay(float d);
float decay();

// ---- cycle transition ----
// Plays out the end of a cycle: the decayed screen collapses like a star, then
// the title card returns and the 呓语 / 意象 ritual runs for `imagery`.
//
// `finished_cycle` is the 1-based number of the cycle that just ENDED. The title
// card labels its throughput figures with it ("CYCLE 3, 3.1 tok/s"), and the
// murmur card announces the NEXT one ("第 4 次呓语"). Those are deliberately two
// different numbers: passing the finished cycle to both made every transition
// announce the previous murmur.
//
// The caller keeps calling resetStep() until it returns false.
void beginReset(int finished_cycle, float tokens_per_s, int total_tokens,
                const char *imagery);
bool resetStep();

}  // namespace disp
