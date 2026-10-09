"""
FOLLOW-UP FIX to 11_two_step_decision_and_evaluation.py, after investigating
WHY Step 1's false-positive rate was 18.9%.

Finding (verified against the data, not assumed): false positives were
overwhelmingly very short emails -- median 22 words, vs 143 words for
correctly-identified legitimate emails. Four of the locked continuous
features are each a VARIANCE or SPREAD measured across multiple sentences
or paragraphs:

    paragraph_density, sentence_length_std, sentence_perplexity_variance,
    sentiment_variance

With only one or two sentences, these are mathematically close to zero
regardless of whether the email is legitimate or phishing -- a short
legitimate reply ("Sounds good, see you tomorrow.") produces the same
near-zero "uniform, low-variance" signature that these features use to spot
AI-generated phishing. The model wasn't wrong about what these features
say; the features themselves stop meaning anything at that length.

Checked before implementing: AI-Phishing emails are essentially NEVER this
short (template-generated, tightly clustered 167-204 words -- fewer than 5
holdout rows out of 3,489 fall under 40 words). So gating these features out
for short emails costs us almost nothing on AI-Phishing detection and
targets exactly the population causing the false positives (905-1,397 of
the ~930-1,400 holdout rows under these thresholds are Legitimate).

THE GATE: for any email under MIN_WORDS=30 words, these four features are
excluded from the Rank-Binned NB posterior (both pipelines), rather than
letting their degenerate near-zero values count as evidence. This is
explicitly within the proposal's own stated error-handling (Section 4:
"If one or more selected features fail to extract cleanly... that feature's
contribution is imputed or dropped, rather than the pipeline failing
outright") -- these features aren't failing to extract, but their values are
equally uninformative at this length, which is the same situation in
substance.

Requires: 07-09 already run.
Output: prints the new Step 1 / Step 2 / end-to-end numbers side by side
        with the ungated ones, so the effect of the gate is visible directly.

ACTUAL RESULT, reported honestly rather than oversold: this gate is a real
but PARTIAL fix. False-positive rate: 18.93% -> 18.14% (39 fewer false
positives out of 934). End-to-end accuracy: 81.68% -> 82.15%.

Checked why the effect wasn't larger: of the 895 false positives remaining
AFTER gating, 519 are STILL under the 30-word threshold -- meaning the other
16 features (not just the 4 gated ones) also carry some length-correlated
bias, just more weakly. Most of those 16 are "count per 100 words" style
densities (urgency_density, imperative_density, orthographic_error_density,
etc.), which have the same underlying problem in a milder form: a single
urgent word in a 10-word email gives urgency_density=10%, against a typical
corpus average near 0.4%, purely from having so few words to divide by, not
from the email actually being urgent. Gating 4 features caught the most
extreme cases; the bias is diffuse across most of the per-word-normalized
features, not confined to the four that got gated here. A deeper fix would
need to address that normalization problem more broadly (e.g. a length-aware
smoothing term, or excluding short emails from training entirely and
handling them with a separate rule) rather than extending this same gate to
more features one at a time.
"""
import json
import importlib.util
import numpy as np
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"

MIN_WORDS = 30
GATED_FEATURES = {"paragraph_density", "sentence_length_std", "sentence_perplexity_variance", "sentiment_variance"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


model1 = _load("model1", f"{FEAT}/../individual_tests/08_model1_rank_binned_naive_bayes.py")
model2 = _load("model2", f"{FEAT}/../individual_tests/09_model2_categorical_naive_bayes.py")


def meta_combine(p1, p2, priors, classes):
    raw = {c: (p1[c] * p2[c]) / priors[c] for c in classes}
    s = sum(raw.values())
    return {c: raw[c] / s for c in classes}


def build_pipeline(df, continuous_feats, categorical_feats, classes, class_col="class"):
    priors = {c: df[class_col].eq(c).mean() for c in classes}
    rb_params = model1.fit(df, continuous_feats, classes, class_col) if continuous_feats else {}
    cat_params = model2.fit(df, categorical_feats, classes, class_col) if categorical_feats else {}
    return {"classes": classes, "priors": priors, "rb_params": rb_params, "cat_params": cat_params,
            "continuous_feats": continuous_feats, "categorical_feats": categorical_feats}


def predict_posterior(row, pipeline, use_gate):
    classes, priors = pipeline["classes"], pipeline["priors"]
    skip = GATED_FEATURES if (use_gate and row["word_count"] < MIN_WORDS) else set()
    if pipeline["continuous_feats"]:
        p1 = model1.posterior(row, pipeline["rb_params"], priors, classes, skip_features=skip)
    else:
        p1 = priors
    p2 = model2.posterior(row, pipeline["cat_params"], priors, classes) if pipeline["categorical_feats"] else priors
    return meta_combine(p1, p2, priors, classes)


def confusion(y_true, y_pred, labels):
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(y_true, y_pred):
        cm.loc[t, p] += 1
    return cm


def run_eval(train, holdout, locked, use_gate):
    main_classes = ["Legitimate", "Human-Phishing", "AI-Phishing"]
    main_pipeline = build_pipeline(train, locked["continuous_main"], locked["categorical_main"], main_classes)

    attr_classes = ["Human-Phishing", "AI-Phishing"]
    phishing_train = train[train["class"].isin(attr_classes)].reset_index(drop=True)
    attr_pipeline = build_pipeline(phishing_train, locked["continuous_attribution"],
                                    locked["categorical_attribution"], attr_classes)

    def step1_step2(row):
        main_post = predict_posterior(row, main_pipeline, use_gate)
        p_phishing = main_post["Human-Phishing"] + main_post["AI-Phishing"]
        if p_phishing < 0.50:
            return "Legitimate"
        attr_post = predict_posterior(row, attr_pipeline, use_gate)
        p_ai = attr_post["AI-Phishing"] / (attr_post["AI-Phishing"] + attr_post["Human-Phishing"])
        return "AI-Phishing" if p_ai >= 0.50 else "Human-Phishing"

    holdout = holdout.copy()
    holdout["pred_class"] = holdout.apply(step1_step2, axis=1)

    true_phish = holdout["class"] != "Legitimate"
    pred_phish = holdout["pred_class"] != "Legitimate"
    cm1 = confusion(true_phish.map({True: "Phishing", False: "Legitimate"}),
                     pred_phish.map({True: "Phishing", False: "Legitimate"}), ["Legitimate", "Phishing"])
    acc1 = (true_phish == pred_phish).mean()
    fp = cm1.loc["Legitimate", "Phishing"]
    fpr = fp / cm1.loc["Legitimate"].sum()

    cm3 = confusion(holdout["class"], holdout["pred_class"], main_classes)
    acc3 = (holdout["class"] == holdout["pred_class"]).mean()

    return {"cm1": cm1, "acc1": acc1, "fpr": fpr, "cm3": cm3, "acc3": acc3, "holdout": holdout}


if __name__ == "__main__":
    train = pd.read_csv(f"{FEAT}/training_features.csv")
    holdout = pd.read_csv(f"{FEAT}/holdout_features.csv")
    with open(f"{FEAT}/locked_features.json") as f:
        locked = json.load(f)

    # bring word_count back in (feature CSVs don't carry it -- the original cleaned CSVs do)
    train_wc = pd.read_csv(f"{CLEAN}/training_set.csv")[["id", "word_count"]]
    holdout_wc = pd.read_csv(f"{CLEAN}/holdout_set.csv")[["id", "word_count"]]
    train = train.merge(train_wc, on="id")
    holdout = holdout.merge(holdout_wc, on="id")

    print("=" * 70)
    print("BEFORE (no length gate) -- reproducing 11_two_step_decision_and_evaluation.py")
    print("=" * 70)
    before = run_eval(train, holdout, locked, use_gate=False)
    print(before["cm1"])
    print(f"Step 1 accuracy: {before['acc1']:.4f}   False positive rate: {before['fpr']:.4f}")
    print(f"End-to-end 3-class accuracy: {before['acc3']:.4f}")

    print("\n" + "=" * 70)
    print(f"AFTER (length gate: skip {sorted(GATED_FEATURES)} for emails < {MIN_WORDS} words)")
    print("=" * 70)
    after = run_eval(train, holdout, locked, use_gate=True)
    print(after["cm1"])
    print(f"Step 1 accuracy: {after['acc1']:.4f}   False positive rate: {after['fpr']:.4f}")
    print(f"End-to-end 3-class accuracy: {after['acc3']:.4f}")

    print("\n" + "=" * 70)
    print("CHANGE")
    print("=" * 70)
    print(f"Step 1 accuracy:        {before['acc1']:.4f} -> {after['acc1']:.4f}  ({after['acc1']-before['acc1']:+.4f})")
    print(f"False positive rate:    {before['fpr']:.4f} -> {after['fpr']:.4f}  ({after['fpr']-before['fpr']:+.4f})")
    print(f"End-to-end accuracy:    {before['acc3']:.4f} -> {after['acc3']:.4f}  ({after['acc3']-before['acc3']:+.4f})")

    after["holdout"].to_csv(f"{FEAT}/holdout_predictions_gated.csv", index=False)
