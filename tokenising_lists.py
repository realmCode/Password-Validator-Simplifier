#!/usr/bin/env python3
"""
learn_patterns.py

Extracts:
  - TEMPLATE_FREQ: probabilities of token templates (e.g., ["WORDCAP","DIGITS2"])
  - WORD_FREQ: probabilities of normalized words (after simple l33t de-mapping)

Features:
  - Streams rockyou.txt (or rockyou.txt.gz)
  - JSON-safe output (string keys only; templates joined with "|")
  - Optional top-k previews
  - Configurable min/max length, line limit, and WORDUP collapsing
"""

import argparse, gzip, json, re
from collections import Counter

# --------------------------- config ---------------------------

# simple l33t expansion table
LEET_TABLE = {
    "4":"a", "3":"e", "0":"o", "1":"l", "!":"i", "$":"s", "@":"a", "5":"s",
    "7":"t", "8":"b"
}
LEET = str.maketrans(LEET_TABLE)

def digit_bucket(n: int) -> str:
    """Bucket digit runs into DIGITS1..DIGITS4 (5+ -> DIGITS4)."""
    return f"DIGITS{min(n,4)}"

# --------------------------- tokenization ---------------------------

def tokenize(password: str, keep_wordup: bool = False):
    """
    Split into runs of alpha / digit / symbol.
    Returns list of (TYPE, SEGMENT_STR) where TYPE in:
      WORD, WORDCAP, WORDUP (optional), DIGITS1..4, SYMBOL
    """
    parts = []
    i, n = 0, len(password)
    while i < n:
        ch = password[i]
        if ch.isalpha():
            j = i + 1
            while j < n and password[j].isalpha():
                j += 1
            seg = password[i:j]
            if seg.isupper():
                t = "WORDUP" if keep_wordup else "WORD"
            elif seg[0].isupper() and seg[1:].islower():
                t = "WORDCAP"
            else:
                t = "WORD"
            parts.append((t, seg))
            i = j
        elif ch.isdigit():
            j = i + 1
            while j < n and password[j].isdigit():
                j += 1
            seg = password[i:j]
            parts.append((digit_bucket(len(seg)), seg))
            i = j
        else:
            j = i + 1
            while j < n and not password[j].isalnum():
                j += 1
            seg = password[i:j]
            parts.append(("SYMBOL", seg))
            i = j
    return parts

def normalize_word(seg: str) -> str:
    """Lower + de-l33t."""
    return seg.translate(LEET).lower()

# --------------------------- learning ---------------------------

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def learn(path: str, min_len=8, max_len=64, limit=None, keep_wordup=False, skip_spaces=True):
    """
    Returns:
      template_freq: dict[str, float] where key is template string joined by "|"
      word_freq: dict[str, float]
    """
    template_counter = Counter()
    word_counter = Counter()
    total_templates = 0
    total_words = 0

    with open_maybe_gzip(path) as f:
        for idx, line in enumerate(f, 1):
            if limit and idx > limit:
                break
            pw = line.rstrip("\r\n")
            if not pw:
                continue
            if not (min_len <= len(pw) <= max_len):
                continue
            if skip_spaces and " " in pw:
                continue

            parts = tokenize(pw, keep_wordup=keep_wordup)
            if not parts:
                continue

            # template as list -> json-safe string via "|"
            tpl_types = [t for t, _ in parts]
            template_counter["|".join(tpl_types)] += 1
            total_templates += 1

            # collect normalized words
            for t, seg in parts:
                if t in ("WORD", "WORDCAP", "WORDUP"):
                    w = normalize_word(seg)
                    if w:
                        word_counter[w] += 1
                        total_words += 1

    template_freq = {k: v / total_templates for k, v in template_counter.items()} if total_templates else {}
    word_freq     = {w: c / total_words      for w, c in word_counter.items()}   if total_words else {}
    return template_freq, word_freq

# --------------------------- cli / io ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Learn template/word frequencies from a password corpus (e.g., rockyou).")
    ap.add_argument("--path", default="merged.txt",help="Path to rockyou.txt (or .gz)")
    ap.add_argument("--min-len", type=int, default=8, help="Minimum password length to consider")
    ap.add_argument("--max-len", type=int, default=64, help="Maximum password length to consider")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N lines (useful for testing)")
    ap.add_argument("--keep-wordup", action="store_true", help="Keep WORDUP as a separate class (default collapses to WORD)")
    ap.add_argument("--no-skip-spaces", action="store_true", help="Do not skip passwords containing spaces")
    ap.add_argument("--top-words", type=int, default=0, help="Print top-K words")
    ap.add_argument("--top-templates", type=int, default=0, help="Print top-K templates")
    ap.add_argument("--out-prefix", default="learned", help="Output prefix for JSON files")
    args = ap.parse_args()

    template_freq, word_freq = learn(
        args.path,
        min_len=args.min_len,
        max_len=args.max_len,
        limit=args.limit,
        keep_wordup=args.keep_wordup,
        skip_spaces=not args.no_skip_spaces,
    )
    args_out= args.out_prefix
    tpl_path  = f"learning_json/{args.out_prefix}_template_freq.json"
    word_path = f"learning_json/{args.out_prefix}_word_freq.json"
    leet_path = f"learning_json/{args.out_prefix}_leet_table.json"

    # dump JSON (keys are strings, so this is safe)
    with open(tpl_path, "w", encoding="utf-8") as fo:
        json.dump(template_freq, fo, ensure_ascii=False, indent=2)
    with open(word_path, "w", encoding="utf-8") as fo:
        json.dump(word_freq, fo, ensure_ascii=False, indent=2)
    with open(leet_path, "w", encoding="utf-8") as fo:
        json.dump(LEET_TABLE, fo, ensure_ascii=False, indent=2)

    # previews
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
