// ============================================================================
//  selftest.h -- hardware bring-up diagnostics.
//
//  Kept as a separate unit rather than deleted: this is what actually proved the
//  board works (PSRAM presence, partition headers, and the flash read bandwidth
//  that decides the whole streaming design). Build with -DESPLLM_SELFTEST to run
//  it instead of the artwork.
//
//  Deliberately contains no display code: display.cpp owns the single U8G2
//  instance, and a second one would re-initialise Wire and hang the board.
// ============================================================================
#pragma once

namespace selftest {
void run();
}
