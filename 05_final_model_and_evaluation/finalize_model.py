"""
FINALIZES configuration (C) from 13_rebuild_and_evaluate_v2.py as the accepted
production model: 26 features (locked_features_v2.json) + the length gate
(MIN_WORDS=30) baked in permanently, not as a toggle.

This replaces the original locked_model.json (20 features, no gate) as the
model actually used going forward. Nothing here recomputes statistics --
every parameter is refit from training_set.csv / training_features_v2.csv
using the already-locked feature lists, so this is a straight rebuild, not a
new analysis.

Requires: locked_features_v2.json, training_features_v2.csv (both already
          produced and gate-checked)
Output:   locked_model_final.json (both pipelines' fitted parameters, priors,
          feature lists, and the length-gate constants, all in one file)
"""
import json
import importlib.util
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
FINAL = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/final"

MIN_WORDS = 30
GATED_FEATURES = ["paragraph_density", "sentence_length_std", "sentence_perplexity_variance", "sentiment_variance"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


model1 = _load("model1", f"{FEAT}/../individual_tests/08_model1_rank_binned_naive_bayes.py")
model2 = _load("model2", f"{FEAT}/../individual_tests/09_model2_categorical_naive_bayes.py")


def build_pipeline(df, continuous_feats, categorical_feats, classes, class_col="class"):
    priors = {c: df[class_col].eq(c).mean() for c in classes}
    rb_params = model1.fit(df, continuous_feats, classes, class_col) if continuous_feats else {}
    cat_params = model2.fit(df, categorical_feats, classes, class_col) if categorical_feats else {}
    return {"classes": classes, "priors": priors, "rb_params": rb_params, "cat_params": cat_params,
            "continuous_feats": continuous_feats, "categorical_feats": categorical_feats}


def main():
    with open(f"{FEAT}/locked_features_v2.json") as f:
        locked = json.load(f)

    train = pd.read_csv(f"{FEAT}/training_features_v2.csv")
    train_wc = pd.read_csv(f"{CLEAN}/training_set.csv")[["id", "word_count"]]
    train = train.merge(train_wc, on="id")
    assert train["id"].is_unique, "duplicate ids after word_count merge -- would silently corrupt training counts"

    main_classes = ["Legitimate", "Human-Phishing", "AI-Phishing"]
    main_pipeline = build_pipeline(train, locked["continuous_main"], locked["categorical_main"], main_classes)

    attr_classes = ["Human-Phishing", "AI-Phishing"]
    phishing_train = train[train["class"].isin(attr_classes)].reset_index(drop=True)
    attr_pipeline = build_pipeline(phishing_train, locked["continuous_attribution"],
                                    locked["categorical_attribution"], attr_classes)

    locked_model = {
        "feature_set_version": "v2_26_features_length_gated",
        "locked_features": locked,
        "length_gate": {"min_words": MIN_WORDS, "gated_features": GATED_FEATURES},
        "main_pipeline": main_pipeline,
        "attribution_pipeline": attr_pipeline,
    }
    with open(f"{FINAL}/locked_model_final.json", "w") as f:
        json.dump(locked_model, f, indent=2)

    print("Final model built and saved to locked_model_final.json")
    print(f"  Main pipeline:        {len(locked['continuous_main'])} continuous + "
          f"{len(locked['categorical_main'])} categorical features, {len(train)} training rows")
    print(f"  Attribution pipeline: {len(locked['continuous_attribution'])} continuous + "
          f"{len(locked['categorical_attribution'])} categorical features, {len(phishing_train)} training rows")
    print(f"  Length gate: skip {GATED_FEATURES} for emails under {MIN_WORDS} words")


if __name__ == "__main__":
    main()
