"""Single source of truth for text normalization. Import this everywhere
(training transcripts, dev scoring, test scoring, model output). Never edit
the rules after you have recorded a baseline."""
import re
import unicodedata

PUNCT = r"[।,.?!;:\"'“”‘’()\[\]{}\-–—…/]"
# Assamese digits -> ASCII digits (decide once; here we keep digits as digits)
DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def normalize(text: str, collapse_ra_wa: bool = False) -> str:
    """collapse_ra_wa=False -> strict (keeps Assamese ৰ ৱ).
    collapse_ra_wa=True  -> maps ৰ->র and ৱ->ব so Bengali-style typing is not penalised.
    Report both scores."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(DIGITS)
    if collapse_ra_wa:
        text = text.replace("ৰ", "র").replace("ৱ", "ব")
    text = re.sub(PUNCT, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
