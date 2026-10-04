"""
Canonical text cleaning for the IndoBERT ABSA pipeline.

WHY THIS FILE EXISTS
--------------------
`data/processed/dataset_train_clean.csv` was produced by an EARLIER version of
`preprocessing/preprocess.py` that had no slang-expansion step.  The cleaner in
that file reproduces the stored training column EXACTLY for 800/800 rows, while
the current `preprocess.py` (with the slang dict) does not reproduce any of them.

That matters because the fine-tuned models in `models/` were trained on that
stored column.  If the cleaner used at inference time differs from the one used
at training time, the model receives text in a distribution it never saw and the
predictions are quietly degraded.

Verified reproduction (see docs/BUG_TRACKER.md, BUG-05):
    normalize(text) == dataset_train_clean.csv["clean_text"]   ->  800 / 800 exact
    normalize(text, expand_slang=True)                         ->  0  / 800 exact

TWO TEXT FORMS ON PURPOSE
------------------------
`normalize()`      the MODEL-INPUT form.  Byte-identical to training data.
                   Keeps numerals, does NOT expand slang, does NOT strip the
                   trailing TikTok date stamp.

`normalize_for_gate()`  the CORPUS-INCLUSION form.  Used only to decide whether a
                   row carries enough meaning to keep.  Strips the trailing
                   TikTok date stamp (`3-27`, `12-5`) and expands slang, so that
                   short but meaningful comments such as "setuju", "parah" and
                   "pecat" survive the length filter instead of being discarded
                   because a date stamp padded them past the threshold.
"""

import re

# Trailing TikTok timestamp, e.g. "... 3-27", "... 12-5", "... 2/28".
# Present on most scraped rows and is NOT part of the comment content.
DATE_STAMP = re.compile(r"\s*\b\d{1,2}\s*[-/]\s*\d{1,2}\b\s*$")

_URL = re.compile(r"http\S+|www\S+")
_MENTION = re.compile(r"@\w+")
_HASH = re.compile(r"#")
# Keeps numerals. This character class is what reproduces the training column.
_PUNCT_KEEP_DIGITS = re.compile(r"[^a-zA-Z0-9\s]")
_SPACES = re.compile(r"\s+")

# Chat abbreviation expansion. Used for the inclusion gate only -- NOT for the
# text fed to the model, because the training data never had this applied.
SLANG = {
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "keamanan",
    "krna": "karena",
    "dr": "dari",
    "sm": "sama",
    "aja": "saja",
    "dlm": "dalam",
    "td": "tadi",
    "tp": "tapi",
    "klo": "kalau",
    "kl": "kalau",
    "skrg": "sekarang",
}

# Minimum number of characters a comment needs to be considered analysable.
# Chosen empirically: the old threshold was 10, which silently discarded 24
# genuinely sentiment-bearing comments (see BUG-02).
MIN_LENGTH = 3


def strip_date_stamp(text):
    """Remove a trailing TikTok `M-D` / `M-D` timestamp."""
    if text is None:
        return ""
    return DATE_STAMP.sub("", str(text))


def normalize(text, expand_slang=False):
    """
    Canonical cleaner.

    Parameters
    ----------
    text : str
        Raw comment.
    expand_slang : bool, default False
        Must stay False for anything fed to the fine-tuned models.  The stored
        training column was built without this step.

    Returns
    -------
    str
    """
    if text is None:
        return ""

    t = str(text).lower()

    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASH.sub(" ", t)

    # Keep the numeric magnitude legible ("8,5 miliar" -> "8 5 miliar").
    t = t.replace("8,5", "8 5").replace("8.5", "8 5")

    t = _PUNCT_KEEP_DIGITS.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()

    if expand_slang:
        t = " ".join(SLANG.get(w, w) for w in t.split())

    return t


def normalize_for_gate(text):
    """
    Text used only to decide whether a row is worth keeping.

    Strips the date stamp and expands slang so that short comments are judged on
    their actual content rather than on the length of an attached timestamp.
    """
    return normalize(strip_date_stamp(text), expand_slang=True)


def is_analysable(text, min_length=MIN_LENGTH):
    """True when the comment carries enough content to classify."""
    return len(normalize_for_gate(text)) >= min_length