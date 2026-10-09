"""
Builds training_features_clair.csv / holdout_features_clair.csv for the
CLAIR-only-Human-Phishing-training experiment, WITHOUT re-running the full
spaCy pipeline on every row.

Why that's valid: only 2 of the 26 features (avg_token_logprob,
sentence_perplexity_variance) depend on the background trigram LM, and the
LM has to change because the training text composition changed (CLAIR
phishing replaces Nazario phishing in training). The other 24 features are
pure per-row computations (spaCy dependency-parse features, regex/VADER
features, the 6 new linguistic features) that do NOT depend on what's in
the training set -- their already-extracted values are reused unchanged via
a lookup by row id from the two existing feature tables:
  - training_features_v2.csv       (original train: Legit+AI+Nazario-HP)
  - holdout_features_v2_augmented.csv (original holdout + all CLAIR rows)
Together these cover every row id needed by the new split (verified by
assertion below).

Sentence splitting for the 2 recomputed features uses nltk.sent_tokenize
here instead of spaCy doc.sents (the original method) -- a deliberate
speed/consistency tradeoff for this experiment: avoids a full second spaCy
pass over 14,000+ rows, and is applied identically to every row in this
experiment, so comparisons within this experiment stay valid. Flagged
explicitly as a methodological note, not hidden.
"""
import re
import math
import time
import importlib.util
from collections import Counter
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CAP_CHARS = 6000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cap(text):
    return str(text)[:CAP_CHARS]


def main():
    t0 = time.time()
    ef = _load("extract_features", f"{FEAT}/extract_features.py")  # reuse TrigramLM class + build_background_lm

    train = pd.read_csv(f"{CLEAN}/training_set_clair.csv")
    holdout = pd.read_csv(f"{CLEAN}/holdout_set_clair.csv")

    # ---- 1. refit the LM on the NEW training text ----
    lm = ef.build_background_lm(train)

    # ---- 2. lookup table of the 24 LM-independent features, from existing extractions ----
    lookup = pd.concat([
        pd.read_csv(f"{FEAT}/training_features_v2.csv"),
        pd.read_csv(f"{FEAT}/holdout_features_v2_augmented.csv"),
    ], ignore_index=True).drop_duplicates(subset=["id"]).set_index("id")

    needed_ids = set(train["id"]) | set(holdout["id"])
    missing = needed_ids - set(lookup.index)
    assert not missing, f"{len(missing)} row ids have no prior feature extraction -- {list(missing)[:5]}"

    non_lm_cols = [c for c in lookup.columns if c not in
                   ("class", "avg_token_logprob", "sentence_perplexity_variance")]

    def recompute_lm_feats(df):
        records = []
        for i, row in df.iterrows():
            body_full = str(row["body"]) if pd.notna(row["body"]) else ""
            body = cap(body_full)
            sentences = [s.strip() for s in sent_tokenize(body) if s.strip()]
            sent_token_lists = [re.findall(r"[a-z']+", s.lower()) for s in sentences]
            sent_token_lists = [t for t in sent_token_lists if t]
            sent_lps = lm.sentence_logprobs(sent_token_lists) if sent_token_lists else []
            avg_token_logprob = float(np.mean(sent_lps)) if sent_lps else 0.0
            sentence_perplexity_variance = float(np.var(sent_lps)) if len(sent_lps) > 1 else 0.0
            records.append({"id": row["id"], "avg_token_logprob": avg_token_logprob,
                             "sentence_perplexity_variance": sentence_perplexity_variance})
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(df)}  ({time.time()-t0:.0f}s elapsed)", flush=True)
        return pd.DataFrame(records).set_index("id")

    print(f"Recomputing LM-dependent features for {len(train)} training rows ...")
    train_lm = recompute_lm_feats(train)
    print(f"Recomputing LM-dependent features for {len(holdout)} holdout rows ...")
    holdout_lm = recompute_lm_feats(holdout)

    def assemble(df, lm_feats):
        base = lookup.loc[df["id"], non_lm_cols].copy()
        base["class"] = df.set_index("id")["class"]
        base = base.join(lm_feats)
        base = base.reset_index()
        cols = ["id", "class"] + [c for c in base.columns if c not in ("id", "class")]
        return base[cols]

    train_feats = assemble(train, train_lm)
    holdout_feats = assemble(holdout, holdout_lm)

    train_feats.to_csv(f"{FEAT}/training_features_clair.csv", index=False)
    holdout_feats.to_csv(f"{FEAT}/holdout_features_clair.csv", index=False)

    print(f"\nWrote training_features_clair.csv {train_feats.shape}, holdout_features_clair.csv {holdout_feats.shape}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
