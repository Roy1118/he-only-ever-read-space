# AGENTS.md — 《他只读过宇宙》/ espllm

ESP32-S3 上的刘慈欣小 LLM 艺术装置：训练一个 ~3.5M 参数的 Q8 小语言模型，跑在 N16R8 上，
在 SH1106 OLED 上滚动输出"被囚禁的存在"的独白，然后重置并循环。

**先读这两份文档再动手：**

| 文档 | 用途 |
|---|---|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | 工程记录：阶段状态、硬件实测数据、v1/v2 训练曲线、踩坑史。**改动前先查这里是否已有结论** |
| [`作品报告.md`](作品报告.md) | 艺术叙述（给观众/评审），与本文件互补 |
| [`ref/tinyllamas-zh-main/README.md`](ref/tinyllamas-zh-main/README.md) | 参考实现 `ModenCn/tinyllamas-zh`，3 个已修复的坑很有价值 |

## 环境

- MCU：ESP32-S3-N16R8（16MB QIO flash + 8MB Octal PSRAM），串口 **COM3**（CH343）
- 屏幕：SH1106 128×64 I2C @ `0x3C`，SCL=GPIO3、SDA=GPIO8，库 U8g2
- Python 3.12.10：`C:/Users/yancunh/AppData/Local/Microsoft/WindowsApps/python3.12.exe`（torch 2.14.0+cu126）
- PlatformIO：`C:\Users\yancunh\.platformio\penv\Scripts\platformio.exe`
- esptool 不在 penv 里：`~/.platformio/packages/tool-esptoolpy/esptool.py`，需以该目录为 cwd 运行

## 命令

```powershell
# 编译 + 烧录艺术固件（default_envs 已钉死，可省略 -e）
pio run -e esp32-s3-devkitc-1 -t upload
pio device monitor                                   # COM3 @ 115200

# 数值对拍三件套（改推理内核后必做）
python scripts/verify_forward.py llama2.c/model_esp.bin > ref_logits.txt
pio run -e verify -t upload
python tools/serial_read.py -s 30 -o dev_logits.txt  # 先复位再监听，否则抓不到开机输出
python scripts/compare_logits.py ref_logits.txt dev_logits.txt
pio run -e esp32-s3-devkitc-1 -t upload              # ⚠️ 务必烧回艺术固件

# 训练 / 导出 / 转换（训练与采样必须从 llama2.c/ 启动，data/ 是指向 ../data 的 junction）
cd llama2.c
python train.py cfg_liucixin_v2.py
python export.py q8_v2.bin --version 2 --checkpoint out_liucixin_v2/ckpt.pt
cd ..
python scripts/verify_q8.py --bin llama2.c/q8_v2.bin --ckpt llama2.c/out_liucixin_v2/ckpt.pt
python tools/convert_esp.py --in llama2.c/q8_v2.bin --out llama2.c/model_esp.bin

# 烧裸分区资产（偏移从 partitions.csv 解析）
python tools/flash_assets.py --dry-run
python tools/flash_assets.py

# 提示词（改 src/main.cpp 顶部数组后 Upload 即自动重建，也可手动）
python scripts/gen_prompts.py
python scripts/gen_prompts.py --check                # 陈旧则 exit 1

# PC 端试生成
cd llama2.c ; python ..\scripts\sample.py --prompts --ab
```

## ⚠️ 会静默毁掉设备/构建的约束

1. **`partitions.csv` 注释必须是 ASCII。** PlatformIO `_parse_partitions()` 用 `open(csv)` 走系统
   cp936，中文注释 → `UnicodeDecodeError: 'gbk' codec can't decode byte 0xa8` → 构建崩。
   `llama2.c/cfg_liucixin*.py` 同理（`configurator.py` 用默认编码读）。
   `platformio.ini` 目前有中文注释且正常，但别再加。
2. **`board_build.arduino.memory_type = qio_opi` 必须保留。** 缺失时 platform-espressif32 回退成
   `qio_qspi` → 链接 QSPI 版 `libesp_hw_support.a` → 8MB Octal PSRAM 挂。
   `board_build.psram_type = opi` **不顶用**（只被 main.py 用来算 boot mode）。
3. **`[platformio] default_envs` 不能删。** 多 env 时不带 `-e` 的 Upload 会构建并烧录**全部**环境，
   最后写入 app 分区的是 `verify` 诊断固件 → OLED 永久停在 `载入模型...`，**看起来像死机**
   （真因：`runVerify()` 后 `loop()` 空转，而那句 `showMessage()` 画在它之前）。
4. **烧过 `-e verify` 之后必须烧回生产固件。**
5. `-DESPLLM_SELFTEST`（`src/selftest.cpp` 硬件自检）**不在任何 env 里**，需要手加 build_flags。

## 代码地图

| 路径 | 职责 |
|---|---|
| `src/main.cpp` | 状态机（`PH_GENERATING/SETTLE/RESET`）+ **全部观感参数**（`MAX_NEW_TOKENS`/`TEMPERATURE`/`TOP_P`/`REPEAT_PENALTY`/`DISPLAY_PUMP_MS`）+ 提示词两个数组 |
| `src/llm.cpp/.h` | Q8 推理内核。`load()` 用 `esp_partition_mmap()` 整块映射 model 分区，前向直接读 mmap 指针（零 flash 流式） |
| `src/tokenizer.cpp/.h` | **只解码**的 llama2.c 格式 tokenizer + `Utf8Assembler` 重组跨 token 的 UTF-8 |
| `src/display.cpp/.h` | 唯一的 U8G2 实例：启动动画、文本流、衰减、四拍转场 |
| `src/log.h` | 把 `ESP_LOGI/W/D/V` 重定义到 `Serial`（预编译 sdkconfig 的 `CONFIG_LOG_MAXIMUM_LEVEL=1` 会把 ESP_LOG 编译期裁掉）。**必须在 `esp_log.h` 之后 include** |
| `tools/` | 构建钩子 + 格式转换 + 烧录（`convert_esp.py` `flash_assets.py` `pio_prompts.py` `serial_read.py`） |
| `scripts/` | PC 端数据/训练/校验（见上表命令） |
| `llama2.c/` | karpathy 上游克隆，**未改上游文件**，项目改动全在新文件（`cfg_liucixin*.py`、`export.py` 产物、junction `data/`） |
| `include/prompts.h` | **生成文件，DO NOT EDIT** |
| `partitions.csv` | nvs + otadata + app0/app1(各 2.5MB) + model(9.25MB @0x510000) + tok(1.7MB @0xE50000) |

### 模型与权重布局

- `model_esp.bin`：`[64B header magic "LLME" ver=1][fp32 rms_att/rms_ffn/rms_final][层优先 7 张量 × n_layers][lm_head]`
- 每个 Q8 张量是 `[gs × int8][1 × float32 scale]` 重复；`gs` 由 `export.py` 决定（v1=32, v2=64，固件从 header 读自动适配）
- `tokenizer.bin`：`int32 max_token_length`，然后每 token `{fp32 score, int32 len, bytes}`
- 设备端 KV cache 大小由 GQA 决定：`n_layers × seq × (dim·n_kv_heads/n_heads) × 2 × 4B`。v2 开 GQA（kv=2）后 8.95MB→3.59MB、KV 6.75MB→1.31MB

## 约定

- **提示词改 `src/main.cpp` 顶部的 `IMAGERY_TO_USE[]`（20 个意象）和 `PROMPT_TEMPLATES[]`（8 条含 `{意象}` 的模板）**，标记块 `IMAGERY_BLOCK_*` / `TEMPLATE_BLOCK_*` 之间。
  `tools/pio_prompts.py` 钩子在每次构建时调 `scripts/gen_prompts.py` 重新分词（失败软着陆，保留旧 header）。
  20×8=160 组合，`pickCombo()` 用洗牌袋保证不重复。
  ⚠️ `PROJECT_STATUS.md` 里写的 `PROMPTS_TO_USE[]` / `PROMPT_BLOCK_*` 是**旧名字**，已重构。
- **`include/prompts.h` 的唯一正确生成器是 `scripts/gen_prompts.py`。**
  ⚠️ `scripts/make_prompts.py` 和 `scripts/mine_prompts.py --write` 会写出**格式不兼容**的 header（多出 `PROMPT_TEXT_i[]`），别跑。
  ⚠️ `scripts/verify_prompts.py` 已与当前格式脱节，跑不通。
- **温度用 0.5，不要 0.85**（实测 0.5 对话有来回、归属正确）。
- **提示词的"体式"比"字面"重要**：语料全是第三人称小说叙事，第三人称叙事开场远优于第一人称提问。`scripts/mine_prompts.py` 挖的是语料原文句子。
- **CJK 字体不能用 `getAscent()` 定位基线**：它返回拉丁 'A' 的 ascent（`wqy14` 是 9），而汉字需 12px。代码用固定基线偏移 13 + `INK_ABOVE_BASE = 12`。
- 训练检查点：**必须按 val loss 存最优**，不要 `always_save_checkpoint`（v1 因此丢掉了 best val）。

## 调试方法论（本项目血泪，优先遵守）

- **字节数校验证明不了张量顺序**——任何张量顺序的总字节数都相同，`assert off == len(data)` 只能抓截断。
  本项目因此两次静默错位（输出像"模型没训好"的噪声）。
  因此：① `convert_esp.py` 写完会**反向解析自己的输出逐张量比对**，不一致就拒绝产出；
  ② 改动推理内核后用 `verify_forward.py`（numpy 复刻）与设备 logits 对拍：**top-8 索引必须完全一致**，数值容差 5e-4。
- **先自检再写业务逻辑**：把关键测量排在可能卡死的模块之前（I2C/屏幕卡死会吞掉全部串口输出）。
- `Wire.begin()` 重复调用会让 `u8g2.begin()` 永久卡死 → 自己扫描完 I2C 必须 `Wire.end();` 把总线交还 U8g2。
  所以 `src/selftest.cpp` **故意不含显示代码**。
- `esp_partition_read` 只有 9.1 MB/s，`esp_partition_mmap` 整块 9MB 可一次映射、指针读约 25 MB/s → 走 mmap，别用 read 流式。
- 走 sentencepiece 的脚本要把模型复制到 `tempfile.gettempdir()`（ASCII 路径），中文路径会乱码。
  PlatformIO 钩子里 `subprocess.run` 必须显式 `encoding="utf-8", errors="replace"`（否则按 cp936 解码子进程输出会崩）。
- **PowerShell 的 `>` 默认写 UTF-16**，给 Python 脚本产数据文件要用 `cmd /c "... > f"`。

## 网络

`github.com:443` 被阻断（`git clone` 超时），但 `api.github.com` / `raw.githubusercontent.com` /
`codeload.github.com` 可用。拉仓库用官方 tarball：
`https://codeload.github.com/<owner>/<repo>/tar.gz/refs/heads/main` 然后 `tar -xzf`。
