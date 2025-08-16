# marisa_build_words.py
# Build a marisa-trie from word->prob JSON for ultra-fast, mmap'd lookups.

import json, math, sys
import marisa_trie

IN_WORDS = "learning_json/learned_word_freq.json"   # big JSON {word: prob}
OUT_TRIE = "learning_json/word_logp.trie"           # will be RecordTrie or BytesTrie

def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    # convert to log-prob, clamp floor
    return {w: math.log(max(p, 1e-10)) for w, p in wf.items()}

def try_record_trie(fmt, items):
    """fmt: 'f' or 'd' (no endianness prefix). items: iterable[(key, (val,))]"""
    print(f"Trying RecordTrie with format '{fmt}' ...", flush=True)
    tr = marisa_trie.RecordTrie(fmt, items)
    return tr

def build():
    word_logp = load_words(IN_WORDS)
    keys = list(word_logp.keys())
    vals = [float(word_logp[k]) for k in keys]

    # 1) Try RecordTrie float32 ('f')
    try:
        tr = try_record_trie("f", zip(keys, [(v,) for v in vals]))
        tr.save(OUT_TRIE)
        print(f"OK: RecordTrie('f') saved to {OUT_TRIE} with {len(keys):,} entries")
        return
    except Exception as e:
        print(f"RecordTrie('f') failed: {e!r}")

    # 2) Try RecordTrie float64 ('d')
    try:
        tr = try_record_trie("d", zip(keys, [(v,) for v in vals]))
        tr.save(OUT_TRIE)
        print(f"OK: RecordTrie('d') saved to {OUT_TRIE} with {len(keys):,} entries")
        return
    except Exception as e:
        print(f"RecordTrie('d') failed: {e!r}")

    # 3) Fallback: BytesTrie (always works)
    import struct
    print("Falling back to BytesTrie (packs <d>) ...")
    items = [(k, struct.pack("<d", v)) for k, v in zip(keys, vals)]
    trb = marisa_trie.BytesTrie(items)
    trb.save(OUT_TRIE)
    print(f"OK: BytesTrie saved to {OUT_TRIE} with {len(keys):,} entries")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        IN_WORDS = sys.argv[1]
    if len(sys.argv) > 2:
        OUT_TRIE = sys.argv[2]
    build()
