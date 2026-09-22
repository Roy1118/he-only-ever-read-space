# 《他只读过宇宙》

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *He Only Ever Read Space* — a 3.5M-parameter language model that has only ever
> read Liu Cixin, sealed inside an ESP32-S3 with no input, no network and no way
> out, forever narrating its own captivity onto a 128×64 OLED and then wiping
> its memory and starting over.

一块火柴盒大小的电路板，连着一块 128×64 像素的单色小屏幕。

它没有摄像头、没有话筒、没有网线、没有按钮，也没有任何别的输入。唯一被"喂"给它的东西，是刘慈欣的全部作品——**四十九本书，两百二十七万个汉字**。它把这四十九本书反复读了四十多遍，然后别的什么也没有了。

通电之后，它只做一件事：用刘慈欣的语言，讲一个被囚禁在黑暗里的存在的故事。屏幕往上滚，一行一行。讲完，画面开始抖动、撕裂、下雪，最后塌缩成一颗星，炸开，归于黑暗。前面说过的话被彻底抹掉——它不记得说过什么，也不记得自己说过。然后它抽一个新的意象，重新开始。

如此往复，直到断电。

---

## ⚠️ 关于版权：这个仓库**不含**刘慈欣的原文

**这是刻意为之，不是遗漏。**

这个作品需要刘慈欣全部作品的纯文本作为训练语料，但那些文本仍然受版权保护。
把它们放进这个仓库再分发，既不合理，也会让仓库被下架（DMCA takedown）。

所以：

| | |
|---|---|
| ❌ **不包含** | 49 本书的原文（`txt/`、`data/corpus.txt`），以及由它切分出的训练分片 |
| ✅ **包含** | 训练**好的模型权重**（`llama2.c/model_esp_v2.bin`，3.6 MB）——这是本项目的产物，可以自由使用 |
| ✅ **包含** | sentencepiece 分词器（111 KB）——这是在原文之上训练出的统计模型，不含原文 |
| ✅ **包含** | 全部代码、配置、转换工具、文档 |

想自己复现训练，请**自备合法来源的文本**，然后按下面的步骤重新生成语料
（见 [仓库里没有什么](#仓库里没有什么)）。

> **This repository contains NO source text by Liu Cixin.** The 49 books are used as
> training data only; the copyrighted text is deliberately excluded from version
> control. What *is* included is the trained 3.6 MB model produced from it, plus all
> of the code, tooling and documentation. To retrain, supply your own legally
> obtained copies of the texts.

---

## 你会看到什么

一轮大约 70–80 秒，分六幕：

| | 时间 | 画面 |
|---|---|---|
| 一 | 0–11 s | 标题浮现，星点缓慢下落，一段进度条走得不均匀 |
| 二 | ~11 s | 进度条卡在 **42%** 停了半秒，闪一下，然后碎掉 |
| 三 | ~3 s | "第 N 次呓语" → "意象：水滴" |
| 四 | ~67 s | 它开始自言自语，文字向上滚 |
| 五 | 最后 ~13 s | 文字开始左右抖动、被啃掉整行、淹没在雪花里 |
| 六 | ~0.8 s | 所有词向中线塌缩成一颗星 → 核心收缩 → 白屏爆炸 |

然后回到第一幕，标题、星点、进度条，全部重来。

> 完整的艺术叙述（为什么每一处是这样、它实际上说了什么）见 [`作品报告.md`](作品报告.md)。
> 工程记录（硬件实测数据、训练曲线、踩过的坑）见 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)。
> 给 AI 编码助手的项目约定见 [`AGENTS.md`](AGENTS.md)。

---

## 硬件

| | |
|---|---|
| MCU | ESP32-S3-**N16R8**（16 MB QIO flash + 8 MB Octal PSRAM） |
| 屏幕 | SH1106 128×64 OLED，I2C |
| 接线 | **SCL = GPIO3，SDA = GPIO8**，地址 **0x3C** |
| 供电 | 一根 USB 线（串口是 CH343，出现在 COM3） |

```
显示屏 SCL ──── GPIO3
显示屏 SDA ──── GPIO8
显示屏 VCC ──── 3V3
显示屏 GND ──── GND
```

需要在 `platformio.ini` 里把 `upload_port` / `monitor_port` 改成你自己的串口号。

---

## 快速开始

### 1. 编译并烧录

```powershell
pio run -e esp32-s3-devkitc-1 -t upload     # 编译 + 烧录固件
python tools/flash_assets.py                # 烧模型和词表到各自的裸分区
pio device monitor                          # COM3 @ 115200
```

> **模型权重在仓库里**（`llama2.c/model_esp_v2.bin`，3.6 MB），所以不需要重新训练就能跑。
> `flash_assets.py` 默认找 `llama2.c/model_esp.bin`；名字对不上就显式指定：
>
> ```powershell
> python tools/flash_assets.py --model llama2.c/model_esp_v2.bin
> python tools/flash_assets.py --dry-run      # 先看看会写到哪个偏移
> ```

### 2. 改提示词

提示词就写在 `src/main.cpp` 顶部，两个数组之间：

```
IMAGERY_TO_USE[]      20 个意象（水滴、二向箔、黑暗森林……）
PROMPT_TEMPLATES[]     8 条含 {意象} 的第三人称开场
```

改完直接 Upload 即可。构建钩子 `tools/pio_prompts.py` 会自动调用
`scripts/gen_prompts.py`，把 20×8=160 种组合全部分词后写进 `include/prompts.h`。

设备端的分词器**只能解码、不能编码**，所以提示词必须在 PC 上分好词再烧进去。

---

## 仓库里有什么

```
src/           固件：状态机、Q8 推理内核、分词器、显示
include/       prompts.h（由 scripts/gen_prompts.py 生成）
tools/         构建钩子 + 格式转换 + 烧录
scripts/       PC 端：语料处理、训练数据、校验、采样
llama2.c/      karpathy/llama2.c 的副本 + 本项目的配置与导出脚本
partitions.csv nvs + otadata + app0/app1 + model(9.25MB) + tok(1.7MB)
```

## 仓库里**没有**什么

**语料。** `txt/` 和 `data/corpus.txt` 是刘慈欣全部作品的纯文本。它们有版权，
不能放在这里再分发。请自备合法来源的文本，然后：

```powershell
# 把 49 本书的纯文本放进 txt/，然后
python scripts/build_corpus.py        # -> data/corpus.txt
python scripts/prepare_data.py        # -> data/tok8192/shard{0,1}.bin
```

`data/tok8192.model`（sentencepiece，111 KB）**在仓库里**，所以不用重新训练分词器。

**训练检查点。** `llama2.c/out_liucixin*/` 里的 `ckpt.pt` 有 40–95 MB，GitHub 不收超过
100 MB 的单文件。要自己从零训一遍：

```powershell
cd llama2.c
cmd /c mklink /J data ..\data         # 训练脚本按相对路径找 data/
python train.py cfg_liucixin_v2.py    # ~10 分钟（RTX 4060）
python export.py q8_v2.bin --version 2 --checkpoint out_liucixin_v2/ckpt_best.pt
cd ..
python scripts/verify_q8.py --bin llama2.c/q8_v2.bin --ckpt llama2.c/out_liucixin_v2/ckpt.pt
python tools/convert_esp.py --in llama2.c/q8_v2.bin --out llama2.c/model_esp_v2.bin
```

**其他。** `downloads/`（2.5 GB 的离线 torch wheel）、`ref/`（第三方参考实现）、
根目录的 `*.log`（调试时抓的串口输出）。理由都写在 `.gitignore` 顶部。

---

## 它是怎么工作的

**模型。** `dim=192, n_layers=5, n_heads=6, n_kv_heads=2 (GQA), hidden=512,
vocab=8192, seq=512`，**3,541,056 个参数**，8-bit 量化后 3.6 MB，
KV cache 1.3 MB 放在 PSRAM 里。一次 0.3 秒，**约 3 个词/秒**。

**权重布局。** 模型分区被 `esp_partition_mmap()` 一次性整块映射进地址空间，
前向传播直接读 mmap 指针，不做 flash 流式读取（`esp_partition_read` 只有
9.1 MB/s，mmap 指针读约 25 MB/s）。

文件格式：`[64B 头 magic "LLME"] [fp32 rms_att/rms_ffn/rms_final]
[逐层 7 个张量 × n_layers] [lm_head]`，每个张量是 `[gs × int8][1 × float32 scale]`。

**随机性。** 20 个意象 × 8 个模板 = 160 种组合，固件用一个洗牌袋遍历它们，
**160 轮之内开场白不会重复**。

---

## 数值验证

这个项目的核心教训是：**字节数对得上证明不了张量顺序对**。任何张量排列的总字节数
都一样，`assert off == len(data)` 只能发现截断。这里因此栽过两次——输出像"模型没训好"
的噪声，实际是布局错位。

所以推理内核有一个独立的对拍流程：

```powershell
# 1. 在 PC 上用 numpy 复刻一遍前向传播，输出参考 logits
python scripts/verify_forward.py llama2.c/model_esp_v2.bin > ref_logits.txt

# 2. 烧一个专门把 logits 打到串口的固件
pio run -e verify -t upload
python tools/serial_read.py -s 30 -o dev_logits.txt

# 3. 逐条对比
python scripts/compare_logits.py ref_logits.txt dev_logits.txt

# 4. 一定要烧回艺术固件！
pio run -e esp32-s3-devkitc-1 -t upload
```

`ref_logits.txt` 和 `dev_logits.txt` 留在仓库根目录，是上一次通过的记录。
两者 **top-8 的 token 索引必须完全一致**，数值容差 5e-4（实测差到 1e-6）。

> ⚠️ 别把 `-e verify` 的固件留在设备上。`runVerify()` 之后 `loop()` 空转，
> 屏幕会永远停在"载入模型..."，看起来完全像死机。

---

## 致谢

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c)（MIT）——训练、导出、
  参考推理实现。本仓库里 `llama2.c/` 是它在 commit
  `350e04f`（2024-05-29）的副本，上游文件仅 `train.py` 有一处修改（区分"最新检查点"
  和"最优检查点"），其余改动全部在新文件里。
- [olikraus/U8g2](https://github.com/olikraus/u8g2)（BSD-2）——屏幕驱动和中文点阵字体。
- [google/sentencepiece](https://github.com/google/sentencepiece)（Apache-2.0）——分词器。
- 感谢 [ModenCn/tinyllamas-zh](https://github.com/ModenCn/tinyllamas-zh)，
  中文小模型训练的参考。

## 许可

[MIT](LICENSE) —— **想怎么用就怎么用**，改、卖、闭源、二次创作都行，只要保留版权署名。

这包括仓库里**训练好的模型权重**。

唯一的例外是**训练语料**：刘慈欣作品的原文不属于本仓库，MIT 也不能授予你任何与它
有关的权利。原文不在这里，也不要跟着本仓库一起分发。
