# refer the github repository for json files
# https://github.com/realmCode/Password-Validator-Simplifier in learning_json folder
# due to late time if i fail to upload json i will add readme to new json files link





import json, math, re, struct  # stdlib: JSON IO, math ops, regex, binary packing/unpacking
from dataclasses import dataclass  # dataclass for a clean model container

try:
    import marisa_trie  # optional ultra-fast mmap'ed trie for word lookup
    # pip install marisa_trie
except Exception:
    print("use this  pip install marisa_trie")
    exit()
    marisa_trie = None  # if unavailable, we fall back to plain JSON dict backend

# ----------------------- tokenization (matches trainer) -----------------------


def digit_bucket(n: int) -> str:
    return f"DIGITS{min(n,4)}"  # map any digit-run length (1..∞) into DIGITS1..DIGITS4 (4 = 4+). This matches trainer's bucketing.


def tokenize(password: str, keep_wordup: bool = True):
    parts = []  # will hold (TYPE, SEGMENT) tuples in order
    i, n = 0, len(password)  # i = current index cursor; n = length cache for speed
    while i < n:  # iterate until we've consumed all characters
        ch = password[i]  # current character
        if ch.isalpha():  # alphabetic run: letters only
            j = i + 1  # start scanning forward
            while j < n and password[j].isalpha():
                j += 1  # extend while alphabetic
            seg = password[i:j]  # slice the contiguous alpha segment
            if seg.isupper():  # classify capitalization pattern
                t = (
                    "WORDUP" if keep_wordup else "WORD"
                )  # optionally preserve all-caps as WORDUP else collapse to WORD
            elif seg[0].isupper() and seg[1:].islower():
                t = "WORDCAP"  # Title-case: first uppercase, rest lowercase
            else:
                t = "WORD"  # mixed/lowercase etc.
            parts.append((t, seg))  # record token
            i = j  # advance cursor
        elif ch.isdigit():  # numeric run: digits only
            j = i + 1
            while j < n and password[j].isdigit():
                j += 1  # extend while numeric
            seg = password[i:j]
            parts.append((digit_bucket(len(seg)), seg))  # bucket by length: DIGITS1..4
            i = j
        else:  # symbol run: neither alpha nor digit
            j = i + 1
            while j < n and not password[j].isalnum():
                j += 1  # extend while non-alphanumeric
            seg = password[i:j]
            parts.append(
                ("SYMBOL", seg)
            )  # tag as SYMBOL (no length bucket for symbols here)
            i = j
    return parts  # ordered list of tokens, used for template + word scoring


# ----------------------- keyboard walk detector -----------------------

_KEYBOARD_ROWS = [
    "`1234567890-=",
    "qwertyuiop[]\\",
    "asdfghjkl;'",
    "zxcvbnm,./",
]  # simple QWERTY rows (US layout)


def _has_keyboard_walk(pw: str, run_len: int = 4) -> bool:
    s = pw.lower()  # case-insensitive check
    rows = _KEYBOARD_ROWS + [
        r[::-1] for r in _KEYBOARD_ROWS
    ]  # include reversed rows to detect descending walks
    for row in rows:  # for every row direction
        for i in range(len(row) - run_len + 1):  # slide a window of size run_len
            if (
                row[i : i + run_len] in s
            ):  # if any contiguous keyboard subsequence appears in password
                return True  # consider as a keyboard walk
    return False  # otherwise, no walk detected


# ----------------------- word probability backends -----------------------


class WordLogpLookup:
    """
    Provides word -> log(probability) lookup via either:
    - a JSON dict kept in memory (simple, but heavy for huge vocabularies)
    - a marisa-trie index (RecordTrie or BytesTrie) for mmap'ed, fast, low-memory lookups
    """

    def __init__(self, floor: float = 1e-10):
        self.floor = floor  # floor probability for OOV (out-of-vocabulary) words
        self._backend = "dict"  # backend type label
        self._dict = {}  # storage for dict backend
        self._tr = None  # marisa trie object (RecordTrie or BytesTrie)
        self._bytes = False  # True if using BytesTrie (values are raw bytes)

    @staticmethod
    def from_json_dict(word_freq_json_path: str, floor: float = 1e-10):
        inst = WordLogpLookup(floor)  # create instance with floor
        with open(word_freq_json_path, "r", encoding="utf-8") as f:
            wf = json.load(f)  # load {word: probability} map
        inst._dict = {
            w: math.log(max(p, floor)) for w, p in wf.items()
        }  # convert to log-probs with floor
        inst._backend = "dict"  # mark backend
        return inst  # ready for get()

    @staticmethod
    def from_trie(trie_path: str, floor: float = 1e-10):
        if marisa_trie is None:  # ensure marisa_trie is available
            raise RuntimeError("marisa_trie not installed")
        inst = WordLogpLookup(floor)
        # try float32 RecordTrie first (smallest / fast)
        try:
            tr = marisa_trie.RecordTrie(
                "f"
            )  # open a RecordTrie where each value is a single float32
            tr.load(trie_path)  # memory-map the trie from disk
            inst._tr = tr
            inst._backend = "record_f"
            return inst
        except Exception:
            pass
        # try float64 RecordTrie next (if built that way)
        try:
            tr = marisa_trie.RecordTrie("d")  # open a RecordTrie with float64 entries
            tr.load(trie_path)
            inst._tr = tr
            inst._backend = "record_d"
            return inst
        except Exception:
            pass
        # final fallback: BytesTrie with manually packed doubles
        trb = marisa_trie.BytesTrie()  # BytesTrie stores raw bytes per key
        trb.load(trie_path)
        inst._tr = trb
        inst._bytes = True  # mark bytes mode (struct.unpack needed)
        inst._backend = "bytes"
        return inst

    def get(self, word: str) -> float:
        if self._backend == "dict":  # simple dict lookup
            return self._dict.get(
                word, math.log(self.floor)
            )  # return logp or log(floor) if OOV
        if self._bytes:  # BytesTrie: values are raw bytes (we packed "<d")
            b = self._tr.get(word)  # get a list of values; take first if present
            if not b:
                return math.log(self.floor)  # OOV -> floor
            return struct.unpack("<d", b[0])[0]  # unpack little-endian double -> logp
        rec = self._tr.get(
            word
        )  # RecordTrie returns a list of tuples; first tuple, first element
        return rec[0][0] if rec else math.log(self.floor)  # OOV -> floor


# ----------------------- model + scoring -----------------------


@dataclass
class HumanPwModel:
    template_logp: (
        dict  # map: "WORDCAP|DIGITS2|..." -> log(prob) learned from templates
    )
    word_lookup: WordLogpLookup
    leet_map: (
        dict  # single-char leet map for de-leeting words (e.g., {"4":"a","3":"e"})
    )

    # priors/backoffs
    tpl_floor: float = 1e-12  # tiny prior for unseen templates
    word_floor: float = (
        1e-10  # tiny prior for unseen words (redundant with lookup.floor, but kept explicit)
    )
    symbol_alphabet: int = (
        33  # approximate symbol alphabet size (used as uniform for symbol runs)
    )
    classes_required: int = 2  # require at least 2 char classes (e.g., letter+digit)
    min_len: int = 8  # minimum length policy
    max_len: int = 64  # maximum length policy
    forbid_ambiguous: bool = (
        False  # relaxed: do not auto-fail if ambiguous chars appear
    )
    ambiguous_chars: str = (
        "0l"  # if enabling forbid_ambiguous, only truly confusing ones by default
    )
    max_repeat_run: int = 4  # disallow >=4 same chars in a row
    min_guess_threshold: float = 1e10  # require estimated guesses ≥ threshold to pass

    # ---- loading helpers
    @staticmethod
    def from_json(template_path: str, word_json_path: str, leet_path: str):
        with open(template_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)  # load template probabilities
        with open(leet_path, "r", encoding="utf-8") as f:
            leet = json.load(f)  # load leet translation table
        template_logp = {
            k: math.log(max(v, 1e-12)) for k, v in tpl.items()
        }  # convert template probs -> log
        wl = WordLogpLookup.from_json_dict(
            word_json_path, floor=1e-10
        )  # build dict-based word lookup
        return HumanPwModel(template_logp, wl, leet)  # construct model instance

    @staticmethod
    def from_trie(template_path: str, trie_path: str, leet_path: str):
        with open(template_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)  # load template probabilities
        with open(leet_path, "r", encoding="utf-8") as f:
            leet = json.load(f)  # load leet translation table
        template_logp = {
            k: math.log(max(v, 1e-12)) for k, v in tpl.items()
        }  # convert to log
        wl = WordLogpLookup.from_trie(
            trie_path, floor=1e-10
        )  # build trie-backed word lookup
        return HumanPwModel(
            template_logp, wl, leet
        )  # construct model with fast backend

    # ---- normalization
    def _de_leet(self, s: str) -> str:
        if not self.leet_map:  # if no leet map provided, just lowercase
            return s.lower()
        table = str.maketrans(self.leet_map)  # build translation table from map
        return s.translate(table).lower()  # translate leet -> letters then lowercase

    # ---- PCFG-ish log-prob
    def logprob(self, pw: str):
        parts = tokenize(
            pw, keep_wordup=True
        )  # segment the password into (TYPE,SEGMENT)
        if not parts:  # empty tokens if string was empty; guard against log(0)
            return -math.inf, parts
        tpl = "|".join(
            t for t, _ in parts
        )  # construct template key e.g., "WORDCAP|WORD|DIGITS2"
        logp = self.template_logp.get(
            tpl, math.log(self.tpl_floor)
        )  # template prior log-prob (back off if unseen)
        for t, seg in parts:  # add component log-likelihoods
            if t.startswith(
                "WORD"
            ):  # for word segments, de-leet and score by word frequency
                base = self._de_leet(seg)
                logp += self.word_lookup.get(base)  # adds log(prob(word)) or log(floor)
            elif t.startswith("DIGITS"):  # for digits, assume uniform over 10^k
                logp += -math.log(10) * len(seg)
            else:  # for symbols, assume uniform over ~33 printable symbols
                logp += -math.log(self.symbol_alphabet) * len(seg)
        return logp, parts  # return overall log-prob and tokenization (for explanation)

    def est_guesses(self, pw: str) -> float:
        logp, _ = self.logprob(pw)  # compute log-probability under the model
        if not math.isfinite(
            logp
        ):  # if -inf (empty/unscorable), treat as "impossibly strong"
            return float("inf")
        return math.exp(-logp)  # crude mapping: guesses ≈ 1 / probability


# ----------------------- policy checks -----------------------

_RE_CLASSES = [
    re.compile(r"[a-z]"),
    re.compile(r"[A-Z]"),
    re.compile(r"\d"),
    re.compile(r"[^A-Za-z0-9]"),
]


def _count_classes(pw: str) -> int:
    return sum(
        bool(r.search(pw)) for r in _RE_CLASSES
    )  # count how many of the 4 char classes are present


def _has_long_repeat(pw: str, max_run: int) -> bool:
    return (
        re.search(rf"(.)\1{{{max_run-1},}}", pw) is not None
    )  # detect any run of the same char of length >= max_run


# ----------------------- main predicate -----------------------


def is_valid_password(pw: str, model: HumanPwModel, explain: bool = False):
    reasons = []  # accumulate human-readable reasons for accept/reject

    if not (model.min_len <= len(pw) <= model.max_len):  # length policy gate
        reasons.append(f"length {len(pw)} outside [{model.min_len},{model.max_len}]")
        return (False, reasons) if explain else False

    if model.forbid_ambiguous:  # optional ambiguity policy (disabled by default)
        if all(
            c in model.ambiguous_chars for c in pw
        ):  # only fail if the whole password is made of ambiguous chars
            reasons.append("entirely ambiguous characters")
            return (False, reasons) if explain else False

    if (
        _count_classes(pw) < model.classes_required
    ):  # require diversity of character classes
        reasons.append(f"needs ≥{model.classes_required} char classes")
        return (False, reasons) if explain else False

    if _has_long_repeat(pw, model.max_repeat_run):  # reject excessive repetition
        reasons.append(f"repeated character run ≥{model.max_repeat_run}")
        return (False, reasons) if explain else False

    if _has_keyboard_walk(pw, run_len=4):  # reject simple keyboard walks
        reasons.append("keyboard walk sequence detected (len≥4)")
        return (False, reasons) if explain else False

    logp, parts = model.logprob(pw)  # score under the human-pattern model
    guesses = (
        math.exp(-logp) if math.isfinite(logp) else float("inf")
    )  # convert to estimated guesses

    if guesses < model.min_guess_threshold:  # enforce minimum strength threshold
        reasons.append(
            f"too guessable (~{int(guesses):,} guesses < {int(model.min_guess_threshold):,})"
        )
        return (False, reasons) if explain else False

    reasons.append(
        f"looks intentional (template: {'|'.join(t for t,_ in parts)})"
    )  # positive explanation: human-like
    reasons.append(
        f"rare under human patterns (~{int(guesses):,} guesses)"
    )  # positive explanation: strong enough
    return (True, reasons) if explain else True  # final verdict


# ----------------------- example usage -----------------------
if __name__ == "__main__":
    # JSON path mode (works with your current files)
    TEMPLATE_JSON = (
        "learning_json/learned_template_freq.json"  # small: template probabilities
    )
    WORD_JSON = "learning_json/learned_word_freq.json"  # huge: word frequencies (slow to load; use trie for speed)
    LEET_JSON = "learning_json/learned_leet_table.json"  # leet translation map

    # FAST trie mode (use after you build word_logp.trie with the builder I gave)
    TRIE_PATH = "learning_json/word_logp.trie"  # mmap'ed trie file for instant opens

    # choose one:
    use_trie = True  # True -> fast path (trie); False -> JSON dict path

    if use_trie:
        model = HumanPwModel.from_trie(
            TEMPLATE_JSON, TRIE_PATH, LEET_JSON
        )  # load template+leet JSON and trie word lookup
    else:
        model = HumanPwModel.from_json(
            TEMPLATE_JSON, WORD_JSON, LEET_JSON
        )  # load all JSON (slow on big word map)

    tests = [  # a small curated test set: weak, borderline, strong
        "12345678",
        "password123",
        "qwertyuiop",
        "aaaa1111",
        "summer2024",
        "hello123",
        "iloveyou",
        "abcabcabc",
        "letmein!!",
        "welcome1",
        "Dragon2025",
        "Winter#23",
        "Skyline99",
        "Freedom!21",
        "Oceanic47",
        "TigerKing8",
        "H@ppyDay1",
        "MatrixNeo5",
        "C@tLover7",
        "GamingZone2",
        "AquaTrail!47",
        "EchoRidge2029",
        "Starlit*Valley12",
        "NebulaQuest99!",
        "CedarHawk_84",
        "ZenithFlow#67",
        "Crimson$River88",
        "PixelForge2027!",
        "SilentWolf!39",
        "ArcticDream_55",
    ]
    failed = []  # collect those that fail the policy/model
    succeed = []  # collect those that pass

    for pw in tests:  # iterate sample passwords
        ok, why = is_valid_password(
            pw, model, explain=True
        )  # evaluate with explanations
        if ok:
            succeed.append(pw)  # record passing ones
        else:
            failed.append(pw)  # record failing ones
        print(f"{pw:>16} -> {ok} :: {', '.join(why)}")  # print verdict and reasons

    print(failed, succeed)  # final summary lists
