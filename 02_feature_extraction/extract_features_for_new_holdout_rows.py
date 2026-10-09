"""
Extracts all 26 features (original 20 + 6 new) for ONLY the newly-added
fraud-corpus holdout rows (the ones in holdout_set_augmented.csv but not in
holdout_set.csv), and appends them to holdout_features_v2.csv.

Training is NOT re-run and the statistical gate is NOT re-run: both only
ever depended on training_set.csv, which is byte-for-byte unchanged by this
addition. Only the evaluation inputs grow.

The trigram LM is refit here (deterministically, from the same unchanged
training_set.csv) rather than cached from a previous run, so this file is a
standalone, independently-reproducible step -- not reliant on in-memory
state from an earlier script.
"""
import time
import pandas as pd
import importlib.util

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    ef = _load("extract_features", f"{FEAT}/extract_features.py")
    enf = _load("extract_new_features", f"{FEAT}/extract_new_features.py")

    old_holdout = pd.read_csv(f"{CLEAN}/holdout_set.csv")
    new_holdout = pd.read_csv(f"{CLEAN}/holdout_set_augmented.csv")

    new_rows = new_holdout[~new_holdout["id"].isin(old_holdout["id"])].reset_index(drop=True)
    print(f"New rows needing feature extraction: {len(new_rows)}")
    assert (new_rows["class"] == "Human-Phishing").all(), "expected only new Human-Phishing rows"

    # same trigram LM as the original run: fit once on training_set.csv, unchanged
    train_df = pd.read_csv(f"{CLEAN}/training_set.csv")
    lm = ef.build_background_lm(train_df)

    # --- original 20 features, via spaCy ---
    bodies_capped = new_rows["body"].fillna("").apply(ef.cap).tolist()
    records = []
    docs_iter = ef.nlp.pipe(bodies_capped, batch_size=64)
    for i, doc in enumerate(docs_iter):
        row = new_rows.iloc[i]
        feats = ef.extract_one(row, lm, doc)
        feats["id"] = row["id"]
        feats["class"] = row["class"]
        records.append(feats)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(new_rows)}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    new_feats20 = pd.DataFrame(records)
    cols = ["id", "class"] + [c for c in new_feats20.columns if c not in ("id", "class")]
    new_feats20 = new_feats20[cols]

    # --- 6 new regex-based features ---
    new_rows6 = new_rows.apply(lambda r: enf.extract_new_features(r["subject"], r["body"]), axis=1, result_type="expand")
    new_feats26 = pd.concat([new_feats20.reset_index(drop=True), new_rows6.reset_index(drop=True)], axis=1)

    # --- append to the existing v2 holdout feature table ---
    old_feats_v2 = pd.read_csv(f"{FEAT}/holdout_features_v2.csv")
    combined = pd.concat([old_feats_v2, new_feats26], ignore_index=True)

    assert combined["id"].is_unique, "duplicate ids after appending new feature rows"
    assert set(old_feats_v2.columns) == set(new_feats26.columns), "column mismatch between old and new feature rows"

    combined.to_csv(f"{FEAT}/holdout_features_v2_augmented.csv", index=False)
    print(f"\nWrote holdout_features_v2_augmented.csv: {old_feats_v2.shape} -> {combined.shape}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
