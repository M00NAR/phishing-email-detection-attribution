"""
Six NEW candidate features, specifically targeted at the diagnosed weakness:
short legitimate emails (quick replies, one-line confirmations) getting
flagged as phishing because the existing "count per 100 words" features
behave erratically on very little text. These new features are either
presence-based (robust at any length) or an explicit length signal (so the
model can learn the length confound directly instead of being blind to it):

  - quoted_reply_flag (categorical): contains reply/quote markers (">",
    "wrote:", "-----Original Message-----", "From: ... Sent: ... To: ...").
    A short email that's part of a real back-and-forth thread is strong
    evidence of being legitimate, and this is undiluted by length -- a
    5-word reply with a quote block is just as detectable as a 50-word one.
  - signoff_flag (categorical): a conventional sign-off phrase near the end
    (regards/thanks/best/sincerely/cheers). Real correspondence closes this
    way far more often than phishing templates.
  - allcaps_word_flag (categorical): contains an ALL-CAPS word >=4 letters,
    excluding a short stoplist of common all-caps acronyms (a shouting/
    urgency marker, presence-based so it doesn't get diluted by email length).
  - multiple_urls_flag (categorical): more than one distinct URL -- phishing
    often repeats the same malicious link multiple ways; ordinary short
    replies essentially never contain multiple links.
  - log_word_count (continuous): log(1+word_count) -- an EXPLICIT length
    signal. Rather than working around the length confound with a hard gate,
    this lets the Rank-Binned NB learn the length-class relationship directly
    from data, which is more principled than a hand-picked threshold.
  - avg_word_length (continuous): mean characters per word -- a ratio, not a
    raw count, so comparatively more stable on short text than the density
    features already in the pipeline.

These are NEW CANDIDATES, not yet validated -- run the statistical gate on
them jointly with the original 20 (see rerun_full_gate_with_new_features.py)
before trusting any of them.
"""
import re
import numpy as np
import pandas as pd

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CAP_CHARS = 6000  # same cap used in extract_features.py, for consistency

QUOTE_RE = re.compile(
    r"^\s*>|wrote:\s*$|-----\s*Original Message\s*-----|^From:.*\n.*Sent:.*\n.*To:",
    re.IGNORECASE | re.MULTILINE,
)
SIGNOFF_RE = re.compile(
    r"\b(regards|thanks|thank you|best regards|best|sincerely|cheers|many thanks)\b\s*,?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
URL_RE = re.compile(r'https?://[^\s\'"<>\)\]]+', re.IGNORECASE)
ALLCAPS_STOPLIST = {
    "I", "A", "OK", "CEO", "CFO", "IT", "US", "UK", "HR", "PM", "AM", "ID", "FYI",
    "ASAP", "RE", "FW", "FWD", "TO", "CC", "BCC", "URL", "FAQ", "EOD", "COB",
}


def cap(text):
    return str(text)[:CAP_CHARS]


def extract_new_features(subject, body_full):
    subject = str(subject) if pd.notna(subject) else ""
    body_full = str(body_full) if pd.notna(body_full) else ""
    body = cap(body_full)
    text_full = subject + "\n" + body_full
    words = re.findall(r"[A-Za-z']+", body_full)
    n_words = max(len(words), 1)

    quoted_reply_flag = int(bool(QUOTE_RE.search(body)))
    signoff_flag = int(bool(SIGNOFF_RE.search(body[-400:])))  # check near the end only

    allcaps_words = [w for w in re.findall(r"\b[A-Z]{4,}\b", body) if w not in ALLCAPS_STOPLIST]
    allcaps_word_flag = int(len(allcaps_words) > 0)

    urls = set(URL_RE.findall(text_full))
    multiple_urls_flag = int(len(urls) > 1)

    word_count = len(re.findall(r"[A-Za-z']+", body_full))
    log_word_count = float(np.log1p(word_count))
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    return {
        "quoted_reply_flag": quoted_reply_flag,
        "signoff_flag": signoff_flag,
        "allcaps_word_flag": allcaps_word_flag,
        "multiple_urls_flag": multiple_urls_flag,
        "log_word_count": log_word_count,
        "avg_word_length": avg_word_length,
    }


def process(clean_path, feat_path, out_path):
    clean = pd.read_csv(clean_path)[["id", "subject", "body"]]
    feats = pd.read_csv(feat_path)
    new_rows = clean.apply(lambda r: extract_new_features(r["subject"], r["body"]), axis=1, result_type="expand")
    merged = pd.concat([feats, new_rows], axis=1)
    merged.to_csv(out_path, index=False)
    return merged


if __name__ == "__main__":
    train = process(f"{CLEAN}/training_set.csv", f"{FEAT}/training_features.csv", f"{FEAT}/training_features_v2.csv")
    holdout = process(f"{CLEAN}/holdout_set.csv", f"{FEAT}/holdout_features.csv", f"{FEAT}/holdout_features_v2.csv")

    new_cols = ["quoted_reply_flag", "signoff_flag", "allcaps_word_flag", "multiple_urls_flag",
                "log_word_count", "avg_word_length"]
    print("New feature means by class (training):")
    print(train.groupby("class")[new_cols].mean())
