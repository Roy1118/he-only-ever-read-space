// ============================================================================
//  log.h -- make ESP_LOGI actually print.
//
//  The prebuilt Arduino sdkconfig for this platform sets
//
//      CONFIG_LOG_DEFAULT_LEVEL = 1          (ERROR)
//      CONFIG_LOG_MAXIMUM_EQUALS_DEFAULT = 1
//
//  and CONFIG_LOG_MAXIMUM_LEVEL is a COMPILE-time ceiling. So ESP_LOGI and
//  ESP_LOGW are preprocessed into nothing -- calling esp_log_level_set() at
//  runtime cannot bring them back. That silently swallowed the model-layout
//  diagnostics during bring-up, which is exactly when they are most needed.
//
//  Rather than fight the sdkconfig (which would mean shipping a custom one for
//  the whole framework), we redefine the macros on top of Serial. Every existing
//  ESP_LOGI() call site keeps working unchanged.
//
//  Include this AFTER esp_log.h (or instead of it).
// ============================================================================
#pragma once

#include <Arduino.h>

#undef ESP_LOGE
#undef ESP_LOGW
#undef ESP_LOGI
#undef ESP_LOGD
#undef ESP_LOGV

#define ESP_LOGE(tag, fmt, ...) \
    do { Serial.printf("E %s: " fmt "\n", tag, ##__VA_ARGS__); } while (0)
#define ESP_LOGW(tag, fmt, ...) \
    do { Serial.printf("W %s: " fmt "\n", tag, ##__VA_ARGS__); } while (0)
#define ESP_LOGI(tag, fmt, ...) \
    do { Serial.printf("I %s: " fmt "\n", tag, ##__VA_ARGS__); } while (0)
#define ESP_LOGD(tag, fmt, ...) \
    do { Serial.printf("D %s: " fmt "\n", tag, ##__VA_ARGS__); } while (0)
#define ESP_LOGV(tag, fmt, ...) \
    do { Serial.printf("V %s: " fmt "\n", tag, ##__VA_ARGS__); } while (0)
