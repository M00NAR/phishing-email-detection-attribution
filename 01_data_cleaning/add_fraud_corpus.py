"""
Adds the CLAIR-style fraud/advance-fee-fraud corpus (fraud_corpus_parsed.csv,
already parsed from the raw mbox-style upload) into the dataset as a SECOND,
HOLDOUT-ONLY source of Human-Phishing data -- exactly mirroring how raw Enron
is already a second, holdout-only source for Legitimate (see
clean_and_merge.py's holdout_reason for Enron: "held out to test
Legitimate-class generalization beyond Nazario's legit mix").

Why holdout-only, not mixed into training: the training set (1,497/1,497/1,497,
already the ACCEPTED final model's training data) is left completely
untouched -- this only enlarges the Human-Phishing holdout sample, which is
exactly the problem being fixed (n=42 was too small to trust). No change to
Legitimate or AI-Phishing holdout, no change to any training row, so nothing
already accepted gets perturbed by this.

Cleaning applied (decisions and why):
  - Drop 3 corrupted "mega-blob" rows (word_count 10,370 / 10,421 / 23,480) --
    inspected manually: these are the same scam-letter template pasted 50+
    times into a single record (a known artifact of this corpus's original
    mbox formatting, not real long emails). Confirmed by counting repeated
    phrase occurrences inside the body text.
  - Dedupe exact-duplicate bodies (573 found) -- same scam letters were
    mailed to many recipients and collected multiple times.
  - Fill missing subject with "[no subject]" (16 rows), consistent with the
    existing Nazario/Enron cleaning convention.
  - Drop empty-body rows.
  - NOT scrubbing email addresses from body text here, unlike Nazario:
    checked first (see conversation) that no single address is
    over-concentrated in this corpus (max 0.2% of rows) -- there's no
    analogous identity leak to remove, and neither the AI corpus nor Enron
    were scrubbed either, so this keeps the same precedent (only scrub when
    a specific leak is found, not as a blanket policy).
  - Cross-corpus duplicate check: drop any new row whose body exactly matches
    a body already present anywhere in the existing training_set.csv or
    holdout_set.csv (any class) -- would be a genuine data leak if two
    corpora happened to share a letter.

Output: holdout_set_augmented.csv (= existing holdout_set.csv rows + new
        fraud-corpus rows). training_set.csv is NOT touched/rewritten.
"""
import re
import hashlib
import pandas as pd
import numpy as np

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
URL_RE = re.compile(r'https?://[^\s\'"<>\)\]]+', re.IGNORECASE)


def make_id(prefix, text):
    h = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}_{h}"


def main():
    parsed = pd.read_csv(f"{CLEAN}/fraud_corpus_parsed.csv")
    n0 = len(parsed)

    # 1. drop corrupted mega-blob rows
    parsed = parsed[parsed["word_count"] <= 5000].copy()
    n_dropped_corrupted = n0 - len(parsed)

    # 2. fill missing subject, drop empty bodies
    parsed["subject"] = parsed["subject"].fillna("[no subject]")
    parsed["subject"] = parsed["subject"].replace("", "[no subject]")
    parsed["body"] = parsed["body"].fillna("")
    parsed = parsed[parsed["body"].str.strip().str.len() > 0].copy()

    # 3. dedupe exact-duplicate bodies (within the fraud corpus itself)
    n_before_dedup = len(parsed)
    parsed = parsed.drop_duplicates(subset=["body"]).copy()
    n_dropped_internal_dupes = n_before_dedup - len(parsed)

    # 4. recompute word_count/url_count from final body text, build schema
    parsed["word_count"] = parsed["body"].apply(lambda s: len(str(s).split()))
    parsed["url_count"] = parsed["body"].apply(lambda s: len(URL_RE.findall(str(s))))
    parsed["class"] = "Human-Phishing"
    parsed["model"] = np.nan
    parsed["category"] = np.nan
    parsed["source"] = "clair_fraud_corpus (419/advance-fee fraud letters)"
    parsed["id"] = parsed["body"].apply(lambda s: make_id("fraud", s))
    parsed["holdout_reason"] = ("raw CLAIR fraud-email corpus, held out to test Human-Phishing-class "
                                 "generalization beyond Nazario's phishing style")

    fraud_clean = parsed[["id", "class", "subject", "body", "word_count", "url_count",
                           "model", "category", "source", "holdout_reason"]].copy()

    # 5. cross-corpus duplicate check against EVERYTHING already in the dataset
    train = pd.read_csv(f"{CLEAN}/training_set.csv")
    holdout = pd.read_csv(f"{CLEAN}/holdout_set.csv")
    existing_bodies = set(train["body"].astype(str)) | set(holdout["body"].astype(str))

    n_before_cross = len(fraud_clean)
    is_cross_dupe = fraud_clean["body"].astype(str).isin(existing_bodies)
    n_cross_dupes = int(is_cross_dupe.sum())
    fraud_clean = fraud_clean[~is_cross_dupe].copy()

    # also guard against id collisions with existing ids (would silently merge rows downstream)
    existing_ids = set(train["id"]) | set(holdout["id"])
    id_collision = fraud_clean["id"].isin(existing_ids)
    n_id_collisions = int(id_collision.sum())
    fraud_clean = fraud_clean[~id_collision].copy()

    # 6. append to holdout only; training_set.csv untouched
    holdout_aug = pd.concat([holdout, fraud_clean], ignore_index=True)
    holdout_aug.to_csv(f"{CLEAN}/holdout_set_augmented.csv", index=False)

    print(f"Raw parsed fraud-corpus rows:            {n0}")
    print(f"Dropped (corrupted mega-blob >5000 wds):  {n_dropped_corrupted}")
    print(f"Dropped (internal exact-dup bodies):      {n_dropped_internal_dupes}")
    print(f"Dropped (cross-corpus exact duplicate):   {n_cross_dupes}")
    print(f"Dropped (id collision w/ existing data):  {n_id_collisions}")
    print(f"Final new Human-Phishing rows added:      {len(fraud_clean)}")
    print()
    print(f"Old holdout Human-Phishing count: {(holdout['class']=='Human-Phishing').sum()}")
    print(f"New holdout Human-Phishing count: {(holdout_aug['class']=='Human-Phishing').sum()}")
    print(f"\nholdout_set.csv shape (unchanged, for reference): {holdout.shape}")
    print(f"holdout_set_augmented.csv shape: {holdout_aug.shape}")
    print(f"training_set.csv: NOT modified.")


if __name__ == "__main__":
    main()
