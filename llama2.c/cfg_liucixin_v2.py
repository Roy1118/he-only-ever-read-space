# Liu Cixin "imprisoned" mini model -- config v2 (run from llama2.c/)
# Usage: python train.py cfg_liucixin_v2.py
#
# WHAT CHANGED FROM v1, AND WHY
# -----------------------------
# v1 (dim=288, 6 layers, 8.33M params) trained on 1.56M tokens and overfit hard:
#
#     step    train    val
#      200     5.35    5.64   <- best
#      400     4.06    5.83
#      600     3.12    6.65
#      800     2.51    7.45
#
# The cause is not a bad hyperparameter, it is the data/model ratio:
#
#     1,561,344 tokens / 8,331,264 params = 0.19 tokens/param
#
# Chinchilla-optimal is about 20 tokens/param, so v1 was ~100x over-parameterised
# for the corpus. The corpus cannot grow (Liu Cixin wrote 49 books), so the only
# lever is to shrink the model. v2 sits at 3.54M params:
#
#     1,561,344 / 3,538,944 = 0.44 tokens/param     (2.3x better)
#
# Three fixes are applied together:
#   1. smaller model (above)
#   2. dropout 0.1 -- v1 had 0.0. model.py really does apply it (attention,
#      residual and FFN), so this is not a no-op.
#   3. dense eval + best-val checkpointing, so the optimum cannot be lost and the
#      val curve is actually visible. -- max_iters is therefore generous; the
#      run is stopped by looking at the curve, and ckpt_best.pt always holds the
#      best point.
#
# A bonus of the smaller architecture: with GQA (n_kv_heads=2 < n_heads=6) the KV
# cache falls from 6.75 MB to 1.25 MB, and the Q8 weights from 8.95 MB to 3.80 MB.
# Together with the tokenizer that is ~5.2 MB, so the ESP32 can hold the ENTIRE
# model in its 8 MB PSRAM and stop streaming weights from flash. Inference should
# be several times faster than the 1.4 tok/s measured with v1.

# ---------- data ----------
vocab_source = "custom"     # use the trained mini-BPE
vocab_size = 8192
max_seq_len = 512           # keep consistent with the ESP32 context
batch_size = 64
# ---------- model (~3.5M params) ----------
dim = 192
n_layers = 5
n_heads = 6
n_kv_heads = 2              # GQA: shrinks the KV cache 3x on the device
multiple_of = 32
dropout = 0.1               # v1 had 0.0, which is the main overfitting lever
# ---------- optimization ----------
gradient_accumulation_steps = 4  # effective batch = 64*4*512 = 131072 tokens/iter
learning_rate = 8e-4        # smaller models tolerate a larger step
max_iters = 500
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
min_lr = 8e-5
# ---------- training control ----------
eval_interval = 25          # dense: we are hunting for the val minimum
log_interval = 10
eval_iters = 32
always_save_checkpoint = True   # keeps ckpt.pt current; ckpt_best.pt holds the optimum
out_dir = "out_liucixin_v2"
# ---------- system ----------
device = "cuda"
dtype = "bfloat16"          # RTX 4060 (sm_89) supports bf16
compile = False             # torch.compile needs Triton, unavailable on Windows
init_from = "scratch"
eval_only = False
wandb_log = False
