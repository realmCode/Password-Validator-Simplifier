# Note Chat Gpt 5 is used for  commenting and writing functions
# brainstroming and logic is mine, to complete the project with speed 90% parts are written by ai
# my task was ensuring testing and capabilities
import json, math, re, struct
from dataclasses import dataclass

try:
    import marisa_trie
except Exception:
    marisa_trie = None  # still works with JSON dicts

# ----------------------- tokenization (matches trainer) -----------------------

def digit_bucket(n: int) -> str:
    return f"DIGITS{min(n,4)}"

def tokenize(password: str, keep_wordup: bool = True):
    parts = []
    i, n = 0, len(password)
    while i < n:
        ch = password[i]
        if ch.isalpha():
            j = i + 1
            while j < n and password[j].isalpha(): j += 1
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
            while j < n and password[j].isdigit(): j += 1
            seg = password[i:j]
            parts.append((digit_bucket(len(seg)), seg))
            i = j
        else:
            j = i + 1
            while j < n and not password[j].isalnum(): j += 1
            seg = password[i:j]
            parts.append(("SYMBOL", seg))
            i = j
    return parts

# ----------------------- keyboard walk detector -----------------------

_KEYBOARD_ROWS = ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./"]
def _has_keyboard_walk(pw: str, run_len: int = 4) -> bool:
    s = pw.lower()
    rows = _KEYBOARD_ROWS + [r[::-1] for r in _KEYBOARD_ROWS]
    for row in rows:
        for i in range(len(row) - run_len + 1):
            if row[i:i+run_len] in s:
                return True
    return False

# ----------------------- word probability backends -----------------------

class WordLogpLookup:
    """
    Abstracts word->logp lookup.
    - JSON dict backend
    - marisa-trie backends: RecordTrie('f' or 'd') or BytesTrie(<d>)
    """
    def __init__(self, floor: float = 1e-10):
        self.floor = floor
        self._backend = "dict"
        self._dict = {}
        self._tr = None
        self._bytes = False

    @staticmethod
    def from_json_dict(word_freq_json_path: str, floor: float = 1e-10):
        inst = WordLogpLookup(floor)
        with open(word_freq_json_path, "r", encoding="utf-8") as f:
            wf = json.load(f)
        inst._dict = {w: math.log(max(p, floor)) for w, p in wf.items()}
        inst._backend = "dict"
        return inst

    @staticmethod
    def from_trie(trie_path: str, floor: float = 1e-10):
        if marisa_trie is None:
            raise RuntimeError("marisa_trie not installed")
        inst = WordLogpLookup(floor)
        # try RecordTrie('f'), then RecordTrie('d'), then BytesTrie
        try:
            tr = marisa_trie.RecordTrie("f")
            tr.load(trie_path)
            inst._tr = tr
            inst._backend = "record_f"
            return inst
        except Exception:
            pass
        try:
            tr = marisa_trie.RecordTrie("d")
            tr.load(trie_path)
            inst._tr = tr
            inst._backend = "record_d"
            return inst
        except Exception:
            pass
        # bytes fallback
        trb = marisa_trie.BytesTrie()
        trb.load(trie_path)
        inst._tr = trb
        inst._bytes = True
        inst._backend = "bytes"
        return inst

    def get(self, word: str) -> float:
        if self._backend == "dict":
            return self._dict.get(word, math.log(self.floor))
        if self._bytes:
            b = self._tr.get(word)
            if not b:
                return math.log(self.floor)
            return struct.unpack("<d", b[0])[0]
        rec = self._tr.get(word)
        return rec[0][0] if rec else math.log(self.floor)

# ----------------------- model + scoring -----------------------

@dataclass
class HumanPwModel:
    template_logp: dict      # "WORDCAP|DIGITS2|..." -> log(prob)
    word_lookup: WordLogpLookup
    leet_map: dict           # {"4":"a","3":"e",...}

    # priors/backoffs
    tpl_floor: float = 1e-12
    word_floor: float = 1e-10
    symbol_alphabet: int = 33  # printable symbols rough set
    classes_required: int = 2
    min_len: int = 8
    max_len: int = 64
    forbid_ambiguous: bool = False   # relaxed as discussed
    ambiguous_chars: str = "0l"      # only truly confusing by default
    max_repeat_run: int = 4
    min_guess_threshold: float = 1e10

    # ---- loading helpers
    @staticmethod
    def from_json(template_path: str, word_json_path: str, leet_path: str):
        with open(template_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
        with open(leet_path, "r", encoding="utf-8") as f:
            leet = json.load(f)
        template_logp = {k: math.log(max(v, 1e-12)) for k, v in tpl.items()}
        wl = WordLogpLookup.from_json_dict(word_json_path, floor=1e-10)
        return HumanPwModel(template_logp, wl, leet)

    @staticmethod
    def from_trie(template_path: str, trie_path: str, leet_path: str):
        with open(template_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)
        with open(leet_path, "r", encoding="utf-8") as f:
            leet = json.load(f)
        template_logp = {k: math.log(max(v, 1e-12)) for k, v in tpl.items()}
        wl = WordLogpLookup.from_trie(trie_path, floor=1e-10)
        return HumanPwModel(template_logp, wl, leet)

    # ---- normalization
    def _de_leet(self, s: str) -> str:
        if not self.leet_map:
            return s.lower()
        table = str.maketrans(self.leet_map)
        return s.translate(table).lower()

    # ---- PCFG-ish log-prob
    def logprob(self, pw: str):
        parts = tokenize(pw, keep_wordup=True)
        if not parts:
            return -math.inf, parts
        tpl = "|".join(t for t,_ in parts)
        logp = self.template_logp.get(tpl, math.log(self.tpl_floor))
        for t, seg in parts:
            if t.startswith("WORD"):
                base = self._de_leet(seg)
                logp += self.word_lookup.get(base)
            elif t.startswith("DIGITS"):
                logp += -math.log(10) * len(seg)
            else:  # SYMBOL
                logp += -math.log(self.symbol_alphabet) * len(seg)
        return logp, parts

    def est_guesses(self, pw: str) -> float:
        logp, _ = self.logprob(pw)
        if not math.isfinite(logp):
            return float("inf")
        return math.exp(-logp)

# ----------------------- policy checks -----------------------

_RE_CLASSES = [re.compile(r"[a-z]"), re.compile(r"[A-Z]"), re.compile(r"\d"), re.compile(r"[^A-Za-z0-9]")]
def _count_classes(pw: str) -> int:
    return sum(bool(r.search(pw)) for r in _RE_CLASSES)

def _has_long_repeat(pw: str, max_run: int) -> bool:
    return re.search(rf"(.)\1{{{max_run-1},}}", pw) is not None

# ----------------------- main predicate -----------------------

def is_valid_password(pw: str, model: HumanPwModel, explain: bool = False):
    reasons = []

    if not (model.min_len <= len(pw) <= model.max_len):
        reasons.append(f"length {len(pw)} outside [{model.min_len},{model.max_len}]")
        return (False, reasons) if explain else False

    if model.forbid_ambiguous:
        if all(c in model.ambiguous_chars for c in pw):
            reasons.append("entirely ambiguous characters")
            return (False, reasons) if explain else False

    if _count_classes(pw) < model.classes_required:
        reasons.append(f"needs ≥{model.classes_required} char classes")
        return (False, reasons) if explain else False

    if _has_long_repeat(pw, model.max_repeat_run):
        reasons.append(f"repeated character run ≥{model.max_repeat_run}")
        return (False, reasons) if explain else False

    if _has_keyboard_walk(pw, run_len=4):
        reasons.append("keyboard walk sequence detected (len≥4)")
        return (False, reasons) if explain else False

    logp, parts = model.logprob(pw)
    guesses = math.exp(-logp) if math.isfinite(logp) else float("inf")

    if guesses < model.min_guess_threshold:
        reasons.append(f"too guessable (~{int(guesses):,} guesses < {int(model.min_guess_threshold):,})")
        return (False, reasons) if explain else False

    reasons.append(f"looks intentional (template: {'|'.join(t for t,_ in parts)})")
    reasons.append(f"rare under human patterns (~{int(guesses):,} guesses)")
    return (True, reasons) if explain else True

# ----------------------- example usage -----------------------
if __name__ == "__main__":
    # JSON path mode (works with your current files)
    TEMPLATE_JSON = "learning_json/learned_template_freq.json"
    WORD_JSON     = "learning_json/learned_word_freq.json"       # big/slow
    LEET_JSON     = "learning_json/learned_leet_table.json"

    # FAST trie mode (use after you build word_logp.trie with the builder I gave)
    TRIE_PATH     = "learning_json/word_logp.trie"               # mmap, <1s open

    # choose one:
    # fastest inferences
    # https://github.com/realmCode/Password-Validator-Simplifier
    use_trie = True  # set True after you build the trie

    if use_trie:
        model = HumanPwModel.from_trie(TEMPLATE_JSON, TRIE_PATH, LEET_JSON)
    else:
        model = HumanPwModel.from_json(TEMPLATE_JSON, WORD_JSON, LEET_JSON)

    tests = [
        "12345678","password123","qwertyuiop","aaaa1111","summer2024",
        "hello123","iloveyou","abcabcabc","letmein!!","welcome1",
        "Dragon2025","Winter#23","Skyline99","Freedom!21","Oceanic47",
        "TigerKing8","H@ppyDay1","MatrixNeo5","C@tLover7","GamingZone2",
        "AquaTrail!47","EchoRidge2029","Starlit*Valley12","NebulaQuest99!",
        "CedarHawk_84","ZenithFlow#67","Crimson$River88","PixelForge2027!",
        "SilentWolf!39","ArcticDream_55",
    ]
    failed = []
    succeed = []
    for pw in tests:
        ok, why = is_valid_password(pw, model, explain=True)
        if ok:
            succeed.append(pw)
        else:
            failed.append(pw)

        print(f"{pw:>16} -> {ok} :: {', '.join(why)}")
    print(failed, succeed)