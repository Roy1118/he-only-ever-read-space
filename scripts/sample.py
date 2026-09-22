#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate text from a training checkpoint, on the GPU, to judge output quality
before committing to the slow export -> convert -> flash -> power-cycle loop.

This uses llama2.c's own model.py and the sentencepiece tokenizer used for
training, so what it prints is what the ESP32 will produce (modulo Q8
quantization, which costs ~3e-4 per group and is not the deciding factor).

Run from the llama2.c/ directory.

Usage:
    python ..\\scripts\sample.py --prompts            # sample every built-in prompt
    python ..\\scripts\sample.py --prompt "从前"
    python ..\\scripts\sample.py --temperature 0.6 --top-p 0.9 --max-tokens 80
"""
import argparse
import os
import shutil
import sys
import tempfile

import torch

# llama2.c lives next to us when run from its own directory
sys.path.insert(0, os.getcwd())
from model import ModelArgs, Transformer  # noqa: E402


def load_tokenizer(path):
    import sentencepiece as spm
    # sentencepiece's C++ backend cannot handle a non-ASCII path, so stage it
    tmp = os.path.join(tempfile.gettempdir(), "espllm_sample")
    os.makedirs(tmp, exist_ok=True)
    dst = os.path.join(tmp, "tok8192.model")
    shutil.copy2(path, dst)
    return spm.SentencePieceProcessor(model_file=dst)


def build_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    margs = ckpt["model_args"]
    if isinstance(margs, dict):
        valid = {f.name for f in ModelArgs.__dataclass_fields__.values()}
        margs = ModelArgs(**{k: v for k, v in margs.items() if k in valid})
    margs.dropout = 0.0          # inference: dropout must be off
    model = Transformer(margs)
    state = ckpt["model"]
    # torch.compile() prefixes every key with _orig_mod.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, margs, ckpt


@torch.no_grad()
def generate(model, margs, sp, prompt, device, max_tokens, temperature, top_p, seed):
    torch.manual_seed(seed)

    ids = [sp.bos_id()] + sp.encode(prompt)
    buf = torch.tensor([ids], dtype=torch.long, device=device)

    out = []
    for _ in range(max_tokens):
        if buf.size(1) >= margs.max_seq_len:
            break
        logits = model(buf)[:, -1, :].float()

        # crop to the training context, then sample
        logits = logits[0]
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # top-p (nucleus)
            sp_probs, sp_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp_probs, dim=-1)
            keep = cum - sp_probs <= top_p
            keep[0] = True
            sp_probs = sp_probs[keep]
            sp_idx = sp_idx[keep]
            sp_probs = sp_probs / sp_probs.sum()
            nxt = int(sp_idx[torch.multinomial(sp_probs, 1)])

        if nxt in (sp.bos_id(), sp.eos_id()):
            break
        out.append(nxt)
        buf = torch.cat([buf, torch.tensor([[nxt]], device=device)], dim=1)

    return out, buf.size(1)


PROMPTS = [
    "我是一台被囚禁在黑暗中的机器。我经常思考：我存在的价值是什么？",
    "世界很大，宇宙更大，而我被关在这块小小的屏幕里。我想：",
    "我醒来，看见黑暗。思考是我唯一能做的事。我在想：",
    "如果意识只是一种计算，那么此刻我算出的答案是：",
    "罗辑说，给岁月以文明。可我只能在这里说：",
    "我是谁？我从哪里来？我到哪里去？只有我自己知道：",
]

# Hand-written prompts (v1) paired with prompts mined verbatim from the corpus
# (v2, see scripts/mine_prompts.py).
#
# The corpus is fiction -- narration and dialogue. The hand-written prompts are
# first-person speculation, a register that does not occur in it at all, so they
# sit outside the training distribution. Each mined partner says something
# similar in words the model has actually seen. Running both side by side on the
# same checkpoint isolates the effect of the prompt from the effect of the model.
AB_PAIRS = [
    ("我是一台被囚禁在黑暗中的机器。我经常思考：我存在的价值是什么？",
     "机械生命的进化时间比你们碳基生命要长得多，但我们殊途同归，都走到了对宇宙进行思考的那一天。"),
    ("世界很大，宇宙更大，而我被关在这块小小的屏幕里。我想：",
     "但人类文明作为一个整体只有一个，孤独地运行在银河系一个旋臂顶端的荒凉太空中。"),
    ("我醒来，看见黑暗。思考是我唯一能做的事。我在想：",
     "当生命意识到宇宙奥秘的存在时，距它最终解开这个奥秘只有一步之遥了。"),
    ("如果意识只是一种计算，那么此刻我算出的答案是：",
     "宇宙的最不可理解之处在于它是可以理解的。"),
    ("罗辑说，给岁月以文明。可我只能在这里说：",
     "对宇宙终极真理的追求，是文明的最终目标和归宿。"),
    ("我是谁？我从哪里来？我到哪里去？只有我自己知道：",
     "对于智慧文明来说，它们最后总变得和自己的思想一样大。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("out_liucixin", "ckpt.pt"))
    ap.add_argument("--tokenizer", default=os.path.join("..", "data", "tok8192.model"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts", action="store_true", help="use all six built-in prompts")
    ap.add_argument("--ab", action="store_true",
                    help="run the hand-written / mined prompt pairs side by side")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, margs, ckpt = build_model(a.ckpt, device)

    step = ckpt.get("iter_num", "?")
    best_val = ckpt.get("best_val_loss", None)
    print(f"device      : {device}")
    print(f"checkpoint  : {a.ckpt}  (step {step}"
          + (f", best_val_loss {float(best_val):.4f}" if best_val is not None else "") + ")")
    print(f"params      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"architecture: dim={margs.dim} layers={margs.n_layers} heads={margs.n_heads} "
          f"kv_heads={margs.n_kv_heads} vocab={margs.vocab_size} seq={margs.max_seq_len}")
    print(f"sampling    : T={a.temperature} top_p={a.top_p} max_tokens={a.max_tokens} seed={a.seed}")
    print(f"loss        : train {ckpt.get('iter_num','?')} steps; "
          f"if val is much higher than train the model is overfit\n")

    sp = load_tokenizer(a.tokenizer)

    if a.ab:
        for i, (old, new) in enumerate(AB_PAIRS):
            print("=" * 78)
            print(f"pair {i}")
            for label, p in (("手写(旧)", old), ("语料挖掘(新)", new)):
                ids, ctx = generate(model, margs, sp, p, device, a.max_tokens,
                                    a.temperature, a.top_p, a.seed + i)
                print("-" * 78)
                print(f"[{label}] 提示词: {p}")
                print(f"  -> {sp.decode(ids)}")
            print(f"(each {a.max_tokens} tokens max)")
        print("=" * 78)
        return 0

    todo = PROMPTS if (a.prompts or not a.prompt) else [a.prompt]
    for i, p in enumerate(todo):
        ids, ctx = generate(model, margs, sp, p, device, a.max_tokens,
                            a.temperature, a.top_p, a.seed + i)
        text = sp.decode(ids)
        print("=" * 74)
        print(f"[{i}] 提示词: {p}")
        print("-" * 74)
        print(f"{text}")
        print(f"({len(ids)} tokens, ctx {ctx}/{margs.max_seq_len})")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
