// ============================================================================
//  selftest.cpp -- see selftest.h
// ============================================================================

#include "selftest.h"

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_partition.h>
#include <esp_spi_flash.h>
#include <esp_timer.h>
#include <esp_ota_ops.h>

namespace selftest {

static const uint32_t ESP_MODEL_MAGIC = 0x4C4C4D45;  // "LLME"
static const uint32_t LLAMA2C_MAGIC   = 0x616B3432;  // "ak42"

static void hr(const char *t) { Serial.printf("\n===== %s =====\n", t); }

static void testChip() {
    hr("CHIP");
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    Serial.printf("model    : %s rev %d, %d cores @ %u MHz\n",
                  chip.model == CHIP_ESP32S3 ? "ESP32-S3" : "other",
                  chip.revision, chip.cores, (unsigned)getCpuFrequencyMhz());
    Serial.printf("xtal/apb : %u / %u MHz\n",
                  (unsigned)getXtalFrequencyMhz(), (unsigned)(getApbFrequency() / 1000000));
    Serial.printf("flash    : %u MB, sketch %u bytes\n",
                  (unsigned)(ESP.getFlashChipSize() / (1024 * 1024)),
                  (unsigned)ESP.getSketchSize());
}

static void testPsram() {
    hr("PSRAM");
    const size_t total = ESP.getPsramSize();
    Serial.printf("size/free: %u / %u bytes\n", (unsigned)total, (unsigned)ESP.getFreePsram());
    if (!total) {
        Serial.println("!! no PSRAM -- is board_build.arduino.memory_type = qio_opi set?");
        return;
    }
    void *p7 = heap_caps_malloc(7 * 1024 * 1024, MALLOC_CAP_SPIRAM);
    Serial.printf("alloc 7MB: %s\n", p7 ? "OK" : "FAILED");
    if (p7) heap_caps_free(p7);

    const size_t N = 1 * 1024 * 1024;
    uint32_t *buf = (uint32_t *)heap_caps_aligned_alloc(16, N, MALLOC_CAP_SPIRAM);
    if (!buf) return;
    int64_t t0 = esp_timer_get_time();
    for (size_t i = 0; i < N / 4; i++) buf[i] = (uint32_t)(i * 2654435761u);
    int64_t t1 = esp_timer_get_time();
    volatile uint32_t sink = 0;
    int64_t t2 = esp_timer_get_time();
    for (size_t i = 0; i < N / 4; i++) sink ^= buf[i];
    int64_t t3 = esp_timer_get_time();
    Serial.printf("write    : %.1f MB/s | read: %.1f MB/s\n",
                  (N / 1048576.0) / ((t1 - t0) / 1e6), (N / 1048576.0) / ((t3 - t2) / 1e6));
    heap_caps_free(buf);
}

static void testHeap() {
    hr("HEAP");
    Serial.printf("internal free/largest: %u / %u\n",
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    Serial.printf("psram free           : %u\n",
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static void testPartitions() {
    hr("PARTITIONS");
    Serial.printf("running app: %s\n", esp_ota_get_running_partition()->label);

    const esp_partition_t *pm = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const esp_partition_t *pt = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "tok");

    if (pm) {
        Serial.printf("model @ 0x%06X  %u bytes (%.2f MB)\n",
                      (unsigned)pm->address, (unsigned)pm->size, pm->size / 1048576.0);
        uint8_t hdr[64] = {};
        if (esp_partition_read(pm, 0, hdr, sizeof(hdr)) == ESP_OK) {
            uint32_t magic; memcpy(&magic, hdr, 4);
            bool blank = true;
            for (int i = 0; i < 64; i++) if (hdr[i] != 0xFF) { blank = false; break; }
            if (blank) {
                Serial.println("  BLANK -- run tools/flash_assets.py");
            } else if (magic == ESP_MODEL_MAGIC) {
                int32_t v[10];
                for (int i = 0; i < 10; i++) memcpy(&v[i], hdr + 4 + i * 4, 4);
                Serial.printf("  LLME ok: ver=%d dim=%d hidden=%d layers=%d heads=%d "
                              "kv=%d vocab=%d seq=%d gs=%d shared=%d\n",
                              (int)v[0], (int)v[1], (int)v[2], (int)v[3], (int)v[4],
                              (int)v[5], (int)v[6], (int)v[7], (int)v[8], (int)v[9]);
            } else if (magic == LLAMA2C_MAGIC) {
                Serial.println("  raw llama2.c file -- run tools/convert_esp.py first");
            } else {
                Serial.printf("  unknown magic 0x%08X\n", magic);
            }
        }
    } else {
        Serial.println("model partition missing");
    }

    if (pt) {
        Serial.printf("tok   @ 0x%06X  %u bytes\n", (unsigned)pt->address, (unsigned)pt->size);
        int32_t mtl = 0;
        if (esp_partition_read(pt, 0, &mtl, 4) == ESP_OK)
            Serial.printf("  max_token_length = %d (expect 24)\n", (int)mtl);
    }
}

static void testFlashBandwidth(const esp_partition_t *pm) {
    hr("FLASH READ BANDWIDTH");
    if (!pm) return;

    uint8_t *dst = (uint8_t *)heap_caps_malloc(4096, MALLOC_CAP_SPIRAM);
    if (dst) {
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < 512; i++) esp_partition_read(pm, (size_t)i * 4096, dst, 4096);
        int64_t dt = esp_timer_get_time() - t0;
        Serial.printf("esp_partition_read 4K chunks: %.1f MB/s  (do NOT stream weights this way)\n",
                      2.0 / (dt / 1e6));
        heap_caps_free(dst);
    }

    size_t sizes[] = { 9 * 1024 * 1024, 8 * 1024 * 1024, 4 * 1024 * 1024, 1 * 1024 * 1024 };
    for (size_t s : sizes) {
        if (s > pm->size) continue;
        const void *ptr = nullptr;
        spi_flash_mmap_handle_t h;
        if (esp_partition_mmap(pm, 0, s, SPI_FLASH_MMAP_DATA, &ptr, &h) == ESP_OK && ptr) {
            volatile uint32_t acc = 0;
            const uint32_t *w = (const uint32_t *)ptr;
            int64_t t0 = esp_timer_get_time();
            for (size_t i = 0; i < s / 4; i++) acc ^= w[i];
            int64_t dt = esp_timer_get_time() - t0;
            Serial.printf("mmap %4u KB + sequential read: %.1f MB/s  <-- weight streaming speed\n",
                          (unsigned)(s / 1024), (s / 1048576.0) / (dt / 1e6));
            spi_flash_munmap(h);
            (void)acc;
            break;
        }
    }
}

void run() {
    delay(1200);
    Serial.println();
    Serial.println("############################################");
    Serial.println("#  ESP32-S3 hardware self-test             #");
    Serial.println("############################################");
    testChip();
    testPsram();
    testHeap();
    testPartitions();
    const esp_partition_t *pm = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    testFlashBandwidth(pm);
    hr("DONE -- heartbeat every 5 s");
}

}  // namespace selftest
