"""
Re-evaluates the SAME accepted final model (locked_model_final.json -- no
retraining, no re-gating, nothing about the model changed) against the
enlarged holdout set that now includes the CLAIR fraud-corpus rows
(holdout_set_augmented.csv / holdout_features_v2_augmented.csv).

This isolates exactly one thing: does Human-Phishing attribution hold up
once it's measured on 3,441 rows instead of 42? Everything else (Legitimate,
AI-Phishing, the model itself) is identical to evaluate_final.py's run.
"""
import json
import importlib.util
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
FINAL = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/final"


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


def predict_posterior(row, pipeline, min_words, gated_features):
    classes, priors = pipeline["classes"], pipeline["priors"]
    skip = set(gated_features) if row["word_count"] < min_words else set()
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


def bootstrap_ci(values_correct, n_boot=2000, seed=42):
    """values_correct: 0/1 array. Returns (mean, lo95, hi95) via resampling."""
    import numpy as np
    rng = np.random.RandomState(seed)
    arr = np.asarray(values_correct)
    n = len(arr)
    boots = [arr[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    boots = np.sort(boots)
    return arr.mean(), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def main():
    with open(f"{FINAL}/locked_model_final.json") as f:
        locked_model = json.load(f)

    main_pipeline = locked_model["main_pipeline"]
    attr_pipeline = locked_model["attribution_pipeline"]
    min_words = locked_model["length_gate"]["min_words"]
    gated_features = locked_model["length_gate"]["gated_features"]
    main_classes = main_pipeline["classes"]

    holdout = pd.read_csv(f"{FEAT}/holdout_features_v2_augmented.csv")
    holdout_wc = pd.read_csv(f"{CLEAN}/holdout_set_augmented.csv")[["id", "word_count"]]
    holdout = holdout.merge(holdout_wc, on="id")
    assert holdout["id"].is_unique

    def step1_step2(row):
        main_post = predict_posterior(row, main_pipeline, min_words, gated_features)
        p_phishing = main_post["Human-Phishing"] + main_post["AI-Phishing"]
        if p_phishing < 0.50:
            return "Legitimate"
        attr_post = predict_posterior(row, attr_pipeline, min_words, gated_features)
        p_ai = attr_post["AI-Phishing"] / (attr_post["AI-Phishing"] + attr_post["Human-Phishing"])
        return "AI-Phishing" if p_ai >= 0.50 else "Human-Phishing"

    holdout["pred_class"] = holdout.apply(step1_step2, axis=1)

    true_phish = holdout["class"] != "Legitimate"
    pred_phish = holdout["pred_class"] != "Legitimate"
    cm1 = confusion(true_phish.map({True: "Phishing", False: "Legitimate"}),
                     pred_phish.map({True: "Phishing", False: "Legitimate"}), ["Legitimate", "Phishing"])
    acc1 = (true_phish == pred_phish).mean()
    fp, tn = cm1.loc["Legitimate", "Phishing"], cm1.loc["Legitimate", "Legitimate"]
    fpr = fp / (fp + tn)

    tp_rows = holdout[true_phish].copy()
    tp_rows["attr_pred"] = tp_rows.apply(
        lambda r: ("AI-Phishing" if (lambda ap: ap["AI-Phishing"] / (ap["AI-Phishing"] + ap["Human-Phishing"]))(
            predict_posterior(r, attr_pipeline, min_words, gated_features)) >= 0.50 else "Human-Phishing"), axis=1)
    cm2 = confusion(tp_rows["class"], tp_rows["attr_pred"], ["Human-Phishing", "AI-Phishing"])
    acc2 = (tp_rows["class"] == tp_rows["attr_pred"]).mean()

    # per-class attribution recall, with bootstrap 95% CI now that n is large enough to mean something
    hp_rows = tp_rows[tp_rows["class"] == "Human-Phishing"]
    ai_rows = tp_rows[tp_rows["class"] == "AI-Phishing"]
    hp_correct = (hp_rows["attr_pred"] == "Human-Phishing").values.astype(float)
    ai_correct = (ai_rows["attr_pred"] == "AI-Phishing").values.astype(float)
    hp_mean, hp_lo, hp_hi = bootstrap_ci(hp_correct)
    ai_mean, ai_lo, ai_hi = bootstrap_ci(ai_correct)

    cm3 = confusion(holdout["class"], holdout["pred_class"], main_classes)
    acc3 = (holdout["class"] == holdout["pred_class"]).mean()

    print("FINAL MODEL -- evaluated on ENLARGED holdout (Human-Phishing: 42 -> 3,441 rows)")
    print("=" * 78)
    print("Step 1 (Legitimate vs Phishing):")
    print(cm1)
    print(f"accuracy={acc1:.4f}  FPR={fpr:.4f}\n")
    print("Step 2 (attribution, scored on true-phishing holdout rows only):")
    print(cm2)
    print(f"accuracy={acc2:.4f}")
    print(f"Human-Phishing recall: {hp_mean:.4f}  (95% CI [{hp_lo:.4f}, {hp_hi:.4f}], n={len(hp_rows)})")
    print(f"AI-Phishing recall:    {ai_mean:.4f}  (95% CI [{ai_lo:.4f}, {ai_hi:.4f}], n={len(ai_rows)})\n")
    print("End-to-end 3-class:")
    print(cm3)
    print(f"accuracy={acc3:.4f}")

    summary = {
        "step1": {"accuracy": float(acc1), "fpr": float(fpr), "confusion_matrix": cm1.to_dict()},
        "step2": {
            "accuracy": float(acc2), "confusion_matrix": cm2.to_dict(),
            "human_phishing_recall": {"mean": float(hp_mean), "ci95": [float(hp_lo), float(hp_hi)], "n": int(len(hp_rows))},
            "ai_phishing_recall": {"mean": float(ai_mean), "ci95": [float(ai_lo), float(ai_hi)], "n": int(len(ai_rows))},
        },
        "end_to_end": {"accuracy": float(acc3), "confusion_matrix": cm3.to_dict()},
    }
    with open(f"{FINAL}/evaluation_summary_final_augmented.json", "w") as f:
        json.dump(summary, f, indent=2)
    holdout.to_csv(f"{FINAL}/holdout_predictions_final_augmented.csv", index=False)


if __name__ == "__main__":
    main()
