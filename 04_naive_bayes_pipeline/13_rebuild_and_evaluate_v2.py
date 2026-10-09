"""
FOLLOW-UP to 12_length_gate_and_rerun.py, after adding 6 new candidate
features (see extract_new_features.py) and re-running the joint statistical
gate across all 26 features together (rerun_full_gate_with_new_features.py).

Gate result (26 features tested jointly, one FDR correction across all raw
p-values -- see rerun_full_gate_with_new_features.py for why this has to be
joint rather than gating the 6 new ones on their own):

    allcaps_word_flag    main=PASS (d/V=0.232)  attr=PASS (0.280)
    avg_word_length      main=PASS (1.627)       attr=PASS (-0.609)
    log_word_count       main=PASS (0.877)       attr=PASS (-0.877)
    multiple_urls_flag   main=PASS (0.416)       attr=FAIL (0.095, below V>=0.10... actually
                                                   0.095 just misses the 0.10 threshold)
    quoted_reply_flag    main=PASS (0.573)       attr=FAIL (p_adj=1.0, effect=0.000)
    signoff_flag         main=PASS (0.902)       attr=PASS (0.964)

quoted_reply_flag failing the attribution gate is exactly what the feature
was designed to do: it targets Legitimate-vs-Phishing (short real replies),
not Human-Phishing-vs-AI-Phishing, so near-zero signal once you restrict to
phishing-only rows is the expected, correct result, not a bug.
multiple_urls_flag similarly: it was designed to help flag Phishing overall,
and doesn't carry extra information about WHICH kind of phishing once you
already know it's phishing -- also expected, and it only just misses the
attribution threshold (V=0.095 vs 0.10 cutoff) rather than failing it badly.

This script now runs THREE evaluations side by side on the holdout set, to
separate out how much each change contributed:
  (A) BASELINE       -- original 20 features, no length gate (file 11's result)
  (B) +LENGTH GATE   -- original 20 features, length gate added (file 12's result)
  (C) +NEW FEATURES  -- all 26 gate-cleared features, length gate still applied

Requires: rerun_full_gate_with_new_features.py already run (reads
          locked_features.json, locked_features_v2.json,
          training_features_v2.csv, holdout_features_v2.csv)
Output:   prints A/B/C side by side; saves holdout_predictions_v2.csv
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

    # Step 2 scored in isolation on true-phishing rows (not contaminated by Step 1 errors)
    tp_rows = holdout[true_phish].copy()
    tp_rows["attr_pred"] = tp_rows.apply(
        lambda r: ("AI-Phishing" if (lambda ap: ap["AI-Phishing"] / (ap["AI-Phishing"] + ap["Human-Phishing"]))(
            predict_posterior(r, attr_pipeline, use_gate)) >= 0.50 else "Human-Phishing"), axis=1)
    cm2 = confusion(tp_rows["class"], tp_rows["attr_pred"], ["Human-Phishing", "AI-Phishing"])
    acc2 = (tp_rows["class"] == tp_rows["attr_pred"]).mean()

    cm3 = confusion(holdout["class"], holdout["pred_class"], main_classes)
    acc3 = (holdout["class"] == holdout["pred_class"]).mean()

    return {"cm1": cm1, "acc1": acc1, "fpr": fpr, "cm2": cm2, "acc2": acc2,
            "cm3": cm3, "acc3": acc3, "holdout": holdout}


if __name__ == "__main__":
    with open(f"{FEAT}/locked_features.json") as f:
        locked_v1 = json.load(f)
    with open(f"{FEAT}/locked_features_v2.json") as f:
        locked_v2 = json.load(f)

    train_v1 = pd.read_csv(f"{FEAT}/training_features.csv")
    holdout_v1 = pd.read_csv(f"{FEAT}/holdout_features.csv")
    train_v2 = pd.read_csv(f"{FEAT}/training_features_v2.csv")
    holdout_v2 = pd.read_csv(f"{FEAT}/holdout_features_v2.csv")

    # bring word_count back in (feature CSVs don't carry it -- the original cleaned CSVs do)
    train_wc = pd.read_csv(f"{CLEAN}/training_set.csv")[["id", "word_count"]]
    holdout_wc = pd.read_csv(f"{CLEAN}/holdout_set.csv")[["id", "word_count"]]
    train_v1 = train_v1.merge(train_wc, on="id")
    holdout_v1 = holdout_v1.merge(holdout_wc, on="id")
    train_v2 = train_v2.merge(train_wc, on="id")
    holdout_v2 = holdout_v2.merge(holdout_wc, on="id")

    print("=" * 78)
    print("(A) BASELINE -- original 20 features, no length gate")
    print("=" * 78)
    A = run_eval(train_v1, holdout_v1, locked_v1, use_gate=False)
    print(A["cm1"]); print(f"Step1 acc={A['acc1']:.4f}  FPR={A['fpr']:.4f}")
    print(A["cm2"]); print(f"Step2 acc={A['acc2']:.4f}")
    print(f"End-to-end acc={A['acc3']:.4f}\n")

    print("=" * 78)
    print(f"(B) +LENGTH GATE -- original 20 features, gate skips {sorted(GATED_FEATURES)} under {MIN_WORDS} words")
    print("=" * 78)
    B = run_eval(train_v1, holdout_v1, locked_v1, use_gate=True)
    print(B["cm1"]); print(f"Step1 acc={B['acc1']:.4f}  FPR={B['fpr']:.4f}")
    print(B["cm2"]); print(f"Step2 acc={B['acc2']:.4f}")
    print(f"End-to-end acc={B['acc3']:.4f}\n")

    print("=" * 78)
    print("(C) +NEW FEATURES -- all 26 gate-cleared features (16 cont/10 cat main;")
    print("    14 cont/6 cat attribution), length gate still applied")
    print("=" * 78)
    C = run_eval(train_v2, holdout_v2, locked_v2, use_gate=True)
    print(C["cm1"]); print(f"Step1 acc={C['acc1']:.4f}  FPR={C['fpr']:.4f}")
    print(C["cm2"]); print(f"Step2 acc={C['acc2']:.4f}")
    print(f"End-to-end acc={C['acc3']:.4f}\n")

    print("=" * 78)
    print("CHANGE SUMMARY (A -> B -> C)")
    print("=" * 78)
    print(f"Step 1 accuracy:      {A['acc1']:.4f} -> {B['acc1']:.4f} -> {C['acc1']:.4f}   (total {C['acc1']-A['acc1']:+.4f})")
    print(f"False positive rate:  {A['fpr']:.4f} -> {B['fpr']:.4f} -> {C['fpr']:.4f}   (total {C['fpr']-A['fpr']:+.4f})")
    print(f"Step 2 (attribution): {A['acc2']:.4f} -> {B['acc2']:.4f} -> {C['acc2']:.4f}   (total {C['acc2']-A['acc2']:+.4f})")
    print(f"End-to-end accuracy:  {A['acc3']:.4f} -> {B['acc3']:.4f} -> {C['acc3']:.4f}   (total {C['acc3']-A['acc3']:+.4f})")

    # diagnostic: how many of C's remaining false positives are still short emails?
    C_hold = C["holdout"]
    fp_mask = (C_hold["class"] == "Legitimate") & (C_hold["pred_class"] != "Legitimate")
    n_fp = fp_mask.sum()
    n_fp_short = (fp_mask & (C_hold["word_count"] < MIN_WORDS)).sum()
    print(f"\n(C) remaining false positives: {n_fp}, of which {n_fp_short} are still under {MIN_WORDS} words "
          f"({n_fp_short/n_fp:.1%})" if n_fp else "\n(C) no false positives remain")

    C["holdout"].to_csv(f"{FEAT}/holdout_predictions_v2.csv", index=False)
