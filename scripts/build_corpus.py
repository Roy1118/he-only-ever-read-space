#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段1: 语料构建 —— 把 txt/ 下的刘慈欣作品清洗、拼接成训练语料。

- 编码检测: 优先 utf-8(-sig)，失败回退 gb18030 (GBK/GB2312 超集)
- 清洗: 去 BOM、统一换行、压缩空行、去装饰分隔线
- 每本书开头插入《书名》标题（从文件名年份前缀里提取），增加风味
- 输出: data/corpus.txt (UTF-8) + data/stats.json (字符统计)
"""
import glob, json, os, re, sys
from collections import Counter

# ---------- 路径 ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_DIR = os.path.join(ROOT, "txt")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def decode_bytes(raw: bytes) -> str:
    """尝试 utf-8 → gb18030"""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace")

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\ufeff", "")
    # 删掉常见装饰分隔线 (如 ---- / ****** / ... ...)
    t = re.sub(r"^[=\-*—·•~\s]{4,}$", "", t, flags=re.MULTILINE)
    # 压缩连续空行
    t = re.sub(r"\n{3,}", "\n\n", t)
    # 行首行尾空白
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def title_from_filename(fname: str) -> str:
    """'2006-三体.txt' → '三体'; '1997-2-诗云(李白).txt' → '诗云(李白)'"""
    base = os.path.splitext(os.path.basename(fname))[0]
    # 去掉开头的年份号 (如 2006- / 1997-2-)
    m = re.match(r"^\d{4}-\d*\-?", base)
    if m:
        base = base[m.end():]
    return base.strip(" -").replace("（", "(").replace("）", ")")

def main():
    files = sorted(glob.glob(os.path.join(TXT_DIR, "*.txt")))
    if not files:
        sys.exit("没有找到 txt 文件！")
    print(f"发现 {len(files)} 个文件\n")

    books = []          # [(书名, 字数)]
    corpus_parts = []
    total_chars = 0
    all_counts = Counter()
    skipped = []

    for fp in files:
        title = title_from_filename(fp)
        try:
            with open(fp, "rb") as f:
                raw = f.read()
            text = decode_bytes(raw)
        except Exception as e:
            skipped.append((fp, str(e)))
            continue
        text = clean_text(text)
        if len(text) < 50:
            skipped.append((fp, "内容过短"))
            continue
        # 插入书名标题（去重防重复标题）
        header = f"《{title}》\n\n"
        if text.startswith(f"《{title}》"):
            header = ""
        part = header + text
        corpus_parts.append(part)
        books.append((title, len(part)))
        total_chars += len(part)
        all_counts.update(part)
        print(f"  ✓ {title:20s} {len(part):>8,} 字符")

    print(f"\n{'─'*50}")
    print(f"汇总: {len(books)} 本 | 总字符 {total_chars:,} | 唯一字符 {len(all_counts):,}")

    # 写出语料
    corpus = "\n\n".join(corpus_parts) + "\n"
    corpus_path = os.path.join(DATA_DIR, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    # 统计信息
    stats = {
        "file_count": len(books),
        "total_chars": total_chars,
        "unique_chars": len(all_counts),
        "books": [{"title": t, "chars": c} for t, c in books],
        "skipped": skipped,
        "top_characters": all_counts.most_common(80),
    }
    with open(os.path.join(DATA_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n语料已写入: {corpus_path}")
    print(f"统计已写入: {os.path.join(DATA_DIR, 'stats.json')}")
    if skipped:
        print(f"\n⚠ 跳过 {len(skipped)} 个文件:")
        for s in skipped:
            print("   ", s)

if __name__ == "__main__":
    main()
