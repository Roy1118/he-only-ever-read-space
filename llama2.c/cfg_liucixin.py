# Liu Cixin "imprisoned" mini model training config (run from llama2.c/)
# Usage: python train.py cfg_liucixin.py

# ---------- data ----------
vocab_source = "custom"     # use custom trained mini-BPE
vocab_size = 8192
max_seq_len = 512           # keep consistent with ESP32 context
batch_size = 64
# ---------- model (~10.7M params) ----------
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# ---------- optimization ----------
gradient_accumulation_steps = 4  # effective batch = 64*4*512 = 131072 tokens/iter
learning_rate = 5e-4
max_iters = 1200            # ~100 epochs (1.5M tokens / 131k)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
# ---------- training control ----------
eval_interval = 200
log_interval = 10
eval_iters = 64
always_save_checkpoint = True  # keep a usable ckpt after every eval
out_dir = "out_liucixin"
# ---------- system ----------
device = "cuda"
dtype = "bfloat16"          # RTX 4060 (sm_89) supports bf16
compile = False             # torch.compile uses Triton, unavailable on Windows
init_from = "scratch"
eval_only = False
wandb_log = False
