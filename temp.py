#!/usr/bin/env python3
"""
learn_patterns_parallel.py

Same logic & outputs as your script, but parallelized with a pool.

Usage examples:
  python learn_patterns_parallel.py --path merged.txt --workers 8 --backend process
  python learn_patterns_parallel.py --path merged.txt.gz --workers 6 --backend process
"""

import argparse, gzip, json, os, re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

# --------------------------- config ---------------------------

LEET_TABLE = {
    "4":"a", "3":"e", "0":"o", "1":"l", "!":"i", "$":"s", "@":"a", "5":"s",
    "7":"t", "8":"b"
}
LEET = str.maketrans(LEET_TABLE)

def digit_bucket(n: int) -> str:
    return f"DIGITS{min(n,4)}"

# --------------------------- tokenization ---------------------------

def tokenize(password: str, keep_wordup: bool = False):
    parts = []
    append = parts.append
    n = len(password); i = 0
    isalpha = str.isalpha; isdigit = str.isdigit; isalnum = str.isalnum
    while i < n:
        ch = password[i]
        if isalpha(ch):
            j = i + 1
            while j < n and isalpha(password[j]): j += 1
            seg = password[i:j]
            if seg.isupper():
                t = "WORDUP" if keep_wordup else "WORD"
            elif seg[0].isupper() and seg[1:].islower():
                t = "WORDCAP"
            else:
                t = "WORD"
            append((t, seg)); i = j
        elif isdigit(ch):
            j = i + 1
            while j < n and isdigit(password[j]): j += 1
            seg = password[i:j]
            append((digit_bucket(j - i), seg)); i = j
        else:
            j = i + 1
            while j < n and not isalnum(password[j]): j += 1
            seg = password[i:j]
            append(("SYMBOL", seg)); i = j
    return parts

def normalize_word(seg: str) -> str:
    return seg.translate(LEET).lower()

# --------------------------- batch worker ---------------------------

def process_batch(lines, min_len, max_len, keep_wordup, skip_spaces):
    t_counter = Counter()
    w_counter = Counter()
    total_templates = 0
    total_words = 0
    join = "|".join
    tokenize_local = tokenize
    normalize = normalize_word

    for line in lines:
        pw = line.rstrip("\n\r")
        if not pw: 
            continue
        L = len(pw)
        if L < min_len or L > max_len:
            continue
        if skip_spaces and (' ' in pw):
            continue
        parts = tokenize_local(pw, keep_wordup=keep_wordup)
        if not parts:
            continue
        tpl_key = join([t for t,_ in parts])
        t_counter[tpl_key] += 1
        total_templates += 1
        for t, seg in parts:
            if t == "WORD" or t == "WORDCAP" or t == "WORDUP":
                w = normalize(seg)
                if w:
                    w_counter[w] += 1
                    total_words += 1

    # Return plain dicts to reduce pickle overhead
    return dict(t_counter), dict(w_counter), total_templates, total_words

# --------------------------- streaming & parallel orchestration ---------------------------

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def batched_iter(f, batch_lines: int, limit: int | None):
    buf = []
    for idx, line in enumerate(f, 1):
        if limit is not None and idx > limit:
            break
        buf.append(line)
        if len(buf) >= batch_lines:
            yield buf
            buf = []
    if buf:
        yield buf

def learn_parallel(path: str, min_len=8, max_len=64, limit=None, keep_wordup=False,
                   skip_spaces=True, workers=None, backend="process", batch_lines=100_000):
    # Counters to merge
    template_counter = Counter()
    word_counter = Counter()
    total_templates = 0
    total_words = 0

    # Choose executor
    if backend == "thread":
        Executor = ThreadPoolExecutor
    else:
        Executor = ProcessPoolExecutor

    worker = partial(process_batch, min_len=min_len, max_len=max_len,
                     keep_wordup=keep_wordup, skip_spaces=skip_spaces)

    with open_maybe_gzip(path) as f, Executor(max_workers=workers) as ex:
        futures = []
        # pipeline: submit a few batches ahead to keep workers fed
        in_flight = (workers or os.cpu_count() or 4) * 2
        for batch in batched_iter(f, batch_lines=batch_lines, limit=limit):
            futures.append(ex.submit(worker, batch))
            if len(futures) >= in_flight:
                # drain at least one to avoid unbounded queue
                done, futures = futures[0:1], futures[1:]
                for fut in done:
                    t_dict, w_dict, t_sum, w_sum = fut.result()
                    template_counter.update(t_dict)
                    word_counter.update(w_dict)
                    total_templates += t_sum
                    total_words += w_sum

        # drain remaining
        for fut in futures:
            t_dict, w_dict, t_sum, w_sum = fut.result()
            template_counter.update(t_dict)
            word_counter.update(w_dict)
            total_templates += t_sum
            total_words += w_sum

    # Normalize to probabilities (same as original)
    if total_templates:
        inv_tt = 1.0 / total_templates
        template_freq = {k: v * inv_tt for k, v in template_counter.items()}
    else:
        template_freq = {}
    if total_words:
        inv_tw = 1.0 / total_words
        word_freq = {w: c * inv_tw for w, c in word_counter.items()}
    else:
        word_freq = {}
    return template_freq, word_freq

# --------------------------- CLI / IO ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Learn template/word frequencies from a password corpus (parallel).")
    ap.add_argument("--path", default="merged.txt", help="Path to rockyou.txt (or .gz)")
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-wordup", action="store_true")
    ap.add_argument("--no-skip-spaces", action="store_true")
    ap.add_argument("--top-words", type=int, default=0)
    ap.add_argument("--top-templates", type=int, default=0)
    ap.add_argument("--out-prefix", default="learned")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of workers")
    ap.add_argument("--backend", choices=["process","thread"], default="process", help="Pool backend")
    ap.add_argument("--batch-lines", type=int, default=100_000, help="Lines per task (tune for throughput)")
    args = ap.parse_args()

    template_freq, word_freq = learn_parallel(
        args.path,
        min_len=args.min_len,
        max_len=args.max_len,
        limit=args.limit,
        keep_wordup=args.keep_wordup,
        skip_spaces=not args.no_skip_spaces,
        workers=args.workers,
        backend=args.backend,
        batch_lines=args.batch_lines,
    )

    tpl_path  = f"learning_json/{args.out_prefix}1_template_freq.json"
    word_path = f"learning_json/{args.out_prefix}1_word_freq.json"
    leet_path = f"learning_json/{args.out_prefix}1_leet_table.json"

    with open(tpl_path, "w", encoding="utf-8") as fo:
        json.dump(template_freq, fo, ensure_ascii=False, indent=2)
    with open(word_path, "w", encoding="utf-8") as fo:
        json.dump(word_freq, fo, ensure_ascii=False, indent=2)
    with open(leet_path, "w", encoding="utf-8") as fo:
        json.dump(LEET_TABLE, fo, ensure_ascii=False, indent=2)

    if args.top_words:
        print("\nTop words:")
        for i, (w, p) in enumerate(sorted(word_freq.items(), key=lambda x: -x[1])[:args.top_words], 1):
            print(f"{i:>3} {w:<20} {p:.6e}")

    if args.top_templates:
        print("\nTop templates:")
        for i, (tpl, p) in enumerate(sorted(template_freq.items(), key=lambda x: -x[1])[:args.top_templates], 1):
            print(f"{i:>3} {tpl:<40} {p:.6e}")

    print(f"\nSaved:\n  {tpl_path}\n  {word_path}\n  {leet_path}")

if __name__ == "__main__":
    main()
