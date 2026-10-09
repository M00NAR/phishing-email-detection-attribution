"""
Evaluates the FINAL, accepted production model (locked_model_final.json --
26 features, length gate permanently applied) on the holdout set, and
reports the same metrics as configuration (C) in
13_rebuild_and_evaluate_v2.py, as an independent reproducibility check: this
script loads the already-fitted pipeline from disk rather than refitting it
inline, so a mismatch here would mean finalize_model.py's serialization lost
something.

Requires: locked_model_final.json (run finalize_model.py first)
Output:   evaluation_summary_final.json, holdout_predictions_final.csv
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


def main():
    with open(f"{FINAL}/locked_model_final.json") as f:
        locked_model = json.load(f)

    main_pipeline = locked_model["main_pipeline"]
    attr_pipeline = locked_model["attribution_pipeline"]
    min_words = locked_model["length_gate"]["min_words"]
    gated_features = locked_model["length_gate"]["gated_features"]

    holdout = pd.read_csv(f"{FEAT}/holdout_features_v2.csv")
    holdout_wc = pd.read_csv(f"{CLEAN}/holdout_set.csv")[["id", "word_count"]]
    holdout = holdout.merge(holdout_wc, on="id")
    assert holdout["id"].is_unique, "duplicate ids after word_count merge"

    main_classes = main_pipeline["classes"]

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
    tp, fn = cm1.loc["Phishing", "Phishing"], cm1.loc["Phishing", "Legitimate"]
    fp, tn = cm1.loc["Legitimate", "Phishing"], cm1.loc["Legitimate", "Legitimate"]
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    tp_rows = holdout[true_phish].copy()
    tp_rows["attr_pred"] = tp_rows.apply(
        lambda r: ("AI-Phishing" if (lambda ap: ap["AI-Phishing"] / (ap["AI-Phishing"] + ap["Human-Phishing"]))(
            predict_posterior(r, attr_pipeline, min_words, gated_features)) >= 0.50 else "Human-Phishing"), axis=1)
    cm2 = confusion(tp_rows["class"], tp_rows["attr_pred"], ["Human-Phishing", "AI-Phishing"])
    acc2 = (tp_rows["class"] == tp_rows["attr_pred"]).mean()

    cm3 = confusion(holdout["class"], holdout["pred_class"], main_classes)
    acc3 = (holdout["class"] == holdout["pred_class"]).mean()

    print("FINAL MODEL -- holdout evaluation (reproducibility check against config (C))")
    print("=" * 70)
    print("Step 1 (Legitimate vs Phishing):")
    print(cm1)
    print(f"accuracy={acc1:.4f}  FPR={fpr:.4f}  precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}\n")
    print("Step 2 (attribution, scored on true-phishing holdout rows only):")
    print(cm2)
    print(f"accuracy={acc2:.4f}\n")
    print("End-to-end 3-class:")
    print(cm3)
    print(f"accuracy={acc3:.4f}")

    expected = {"acc1": 0.8667, "fpr": 0.2169, "acc2": 0.9833, "acc3": 0.8607}
    got = {"acc1": round(acc1, 4), "fpr": round(fpr, 4), "acc2": round(acc2, 4), "acc3": round(acc3, 4)}
    match = all(abs(got[k] - expected[k]) < 0.001 for k in expected)
    print(f"\nMatches config (C) from 13_rebuild_and_evaluate_v2.py: {match}  (expected={expected}, got={got})")

    summary = {
        "step1": {"accuracy": acc1, "fpr": fpr, "precision": precision, "recall": recall, "f1": f1,
                   "confusion_matrix": cm1.to_dict()},
        "step2": {"accuracy": acc2, "confusion_matrix": cm2.to_dict()},
        "end_to_end": {"accuracy": acc3, "confusion_matrix": cm3.to_dict()},
        "reproducibility_check_passed": match,
    }
    with open(f"{FINAL}/evaluation_summary_final.json", "w") as f:
        json.dump(summary, f, indent=2)
    holdout.to_csv(f"{FINAL}/holdout_predictions_final.csv", index=False)


if __name__ == "__main__":
    main()
