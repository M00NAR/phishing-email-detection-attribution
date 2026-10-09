"""
EXPERIMENT: train Human-Phishing entirely on the CLAIR fraud corpus instead
of Nazario. Per Mannan's explicit scope decision: only the Human-Phishing
side swaps source. Legitimate training stays on Nazario-legit (unchanged --
same pool, same RNG seed, same 1,497 rows). AI-Phishing training unchanged.

target_n is STILL 1,497 -- verified this is bounded by the Legitimate-Nazario
pool (1,497 total, all of it already used in training, 0 left over), not by
either phishing pool. Since that pool and the AI-Phishing pool are both
unchanged, legit_train and ai_train come out IDENTICAL to the previous
training set (same seed, same source data) -- only humphish_train changes.

Human-Phishing holdout for this experiment = CLAIR rows NOT selected for
training (1,902 rows) PLUS all 42 original Nazario-Human-Phishing rows,
explicitly kept as a reverse generalization check: does a model trained
entirely on CLAIR-style fraud letters still recognize Nazario-style
credential-phishing it's never seen in training? (mirrors the original
Enron-holdout logic for Legitimate: a second source, held out specifically
to test generalization beyond what was trained on.)

Input: training_set.csv, holdout_set.csv (original, unmodified on disk),
       fraud_corpus_parsed.csv (already cleaned once in add_fraud_corpus.py
       -- reusing that cleaning here rather than redoing it)
Output: training_set_clair.csv, holdout_set_clair.csv
"""
import re
import hashlib
import numpy as np
import pandas as pd

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
RNG = 42
URL_RE = re.compile(r'https?://[^\s\'"<>\)\]]+', re.IGNORECASE)


def make_id(prefix, text):
    h = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}_{h}"


def main():
    orig_train = pd.read_csv(f"{CLEAN}/training_set.csv")
    orig_holdout = pd.read_csv(f"{CLEAN}/holdout_set.csv")

    # ---- reuse the same CLAIR cleaning as add_fraud_corpus.py ----
    parsed = pd.read_csv(f"{CLEAN}/fraud_corpus_parsed.csv")
    parsed = parsed[parsed["word_count"] <= 5000].copy()
    parsed["subject"] = parsed["subject"].fillna("[no subject]").replace("", "[no subject]")
    parsed["body"] = parsed["body"].fillna("")
    parsed = parsed[parsed["body"].str.strip().str.len() > 0].copy()
    parsed = parsed.drop_duplicates(subset=["body"]).copy()
    parsed["word_count"] = parsed["body"].apply(lambda s: len(str(s).split()))
    parsed["url_count"] = parsed["body"].apply(lambda s: len(URL_RE.findall(str(s))))
    parsed["class"] = "Human-Phishing"
    parsed["model"] = np.nan
    parsed["category"] = np.nan
    parsed["source"] = "clair_fraud_corpus (419/advance-fee fraud letters)"
    parsed["id"] = parsed["body"].apply(lambda s: make_id("fraud", s))
    clair_pool = parsed[["id", "class", "subject", "body", "word_count", "url_count",
                          "model", "category", "source"]].copy().reset_index(drop=True)

    # ---- unchanged pools, pulled straight from the original (accepted) split ----
    legit_train = orig_train[orig_train["class"] == "Legitimate"].copy()
    ai_train = orig_train[orig_train["class"] == "AI-Phishing"].copy()
    naz_hp_train_old = orig_train[orig_train["class"] == "Human-Phishing"].copy()  # being REPLACED
    naz_hp_holdout = orig_holdout[orig_holdout["class"] == "Human-Phishing"].copy()  # the original 42

    legit_holdout_rest = orig_holdout[orig_holdout["class"] == "Legitimate"].copy()  # Enron + any leftover Nazario-legit
    ai_holdout_rest = orig_holdout[orig_holdout["class"] == "AI-Phishing"].copy()

    target_n = len(legit_train)  # 1,497 -- same binding constraint as the original split
    assert target_n <= len(clair_pool), "CLAIR pool smaller than target_n -- would need to shrink target_n"

    clair_train = clair_pool.sample(n=target_n, random_state=RNG).reset_index(drop=True)
    clair_holdout = clair_pool[~clair_pool["id"].isin(clair_train["id"])].reset_index(drop=True)
    clair_holdout = clair_holdout.assign(
        holdout_reason="CLAIR fraud-corpus rows not used in training (downsampled for balance)")
    naz_hp_holdout = naz_hp_holdout.assign(
        holdout_reason="Nazario-style Human-Phishing, entirely held out -- reverse generalization check: "
                        "does a model trained ONLY on CLAIR-style fraud letters recognize this different "
                        "phishing style it never saw in training?")

    # ---- assemble ----
    new_training = pd.concat([legit_train, clair_train.assign(holdout_reason=np.nan), ai_train], ignore_index=True)
    new_training = new_training.sample(frac=1, random_state=RNG).reset_index(drop=True)

    new_holdout = pd.concat(
        [legit_holdout_rest, ai_holdout_rest, clair_holdout, naz_hp_holdout],
        ignore_index=True,
    )

    new_training.to_csv(f"{CLEAN}/training_set_clair.csv", index=False)
    new_holdout.to_csv(f"{CLEAN}/holdout_set_clair.csv", index=False)

    print("New training set class balance:")
    print(new_training["class"].value_counts())
    print("\nNew holdout set class balance:")
    print(new_holdout["class"].value_counts())
    print(f"\nHuman-Phishing holdout breakdown: CLAIR-leftover={len(clair_holdout)}, Nazario-crosscheck={len(naz_hp_holdout)}")
    print(f"\nSanity: legit_train identical to original? {set(legit_train['id'])==set(orig_train[orig_train['class']=='Legitimate']['id'])}")
    print(f"Sanity: ai_train identical to original?    {set(ai_train['id'])==set(orig_train[orig_train['class']=='AI-Phishing']['id'])}")

    # cross-corpus duplicate sanity check (should already be 0 from add_fraud_corpus.py, re-verify here)
    other_bodies = set(legit_train["body"]) | set(ai_train["body"]) | set(naz_hp_train_old["body"]) | set(naz_hp_holdout["body"])
    dupes = clair_pool["body"].isin(other_bodies).sum()
    print(f"CLAIR rows duplicating any Legitimate/AI-Phishing/Nazario-phishing body: {dupes}")


if __name__ == "__main__":
    main()
