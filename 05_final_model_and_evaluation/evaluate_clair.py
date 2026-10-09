"""
Builds and evaluates the CLAIR-only-Human-Phishing-training model: same
architecture (two Naive Bayes pipelines + Meta-NB, length gate at
MIN_WORDS=30) as the accepted final model, but trained on
training_features_clair.csv / locked_features_clair.json instead.

Reports the overall holdout numbers AND a breakdown of the Human-Phishing
holdout by source (CLAIR-leftover vs the 42 Nazario-crosscheck rows), since
that breakdown is the actual point of this experiment: does training
exclusively on CLAIR-style fraud letters generalize back to Nazario-style
credential-phishing it never saw in training?
"""
import json
import importlib.util
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
FINAL = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/final"

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


def predict_posterior(row, pipeline, use_gate=True):
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


def main():
    with open(f"{FEAT}/locked_features_clair.json") as f:
        locked = json.load(f)

    train = pd.read_csv(f"{FEAT}/training_features_clair.csv")
    holdout = pd.read_csv(f"{FEAT}/holdout_features_clair.csv")
    train_wc = pd.read_csv(f"{CLEAN}/training_set_clair.csv")[["id", "word_count"]]
    holdout_wc = pd.read_csv(f"{CLEAN}/holdout_set_clair.csv")[["id", "word_count", "source"]]
    train = train.merge(train_wc, on="id")
    holdout = holdout.merge(holdout_wc, on="id")

    main_classes = ["Legitimate", "Human-Phishing", "AI-Phishing"]
    main_pipeline = build_pipeline(train, locked["continuous_main"], locked["categorical_main"], main_classes)

    attr_classes = ["Human-Phishing", "AI-Phishing"]
    phishing_train = train[train["class"].isin(attr_classes)].reset_index(drop=True)
    attr_pipeline = build_pipeline(phishing_train, locked["continuous_attribution"],
                                    locked["categorical_attribution"], attr_classes)

    def step1_step2(row):
        main_post = predict_posterior(row, main_pipeline)
        p_phishing = main_post["Human-Phishing"] + main_post["AI-Phishing"]
        if p_phishing < 0.50:
            return "Legitimate"
        attr_post = predict_posterior(row, attr_pipeline)
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
            predict_posterior(r, attr_pipeline)) >= 0.50 else "Human-Phishing"), axis=1)
    cm2 = confusion(tp_rows["class"], tp_rows["attr_pred"], ["Human-Phishing", "AI-Phishing"])
    acc2 = (tp_rows["class"] == tp_rows["attr_pred"]).mean()

    cm3 = confusion(holdout["class"], holdout["pred_class"], main_classes)
    acc3 = (holdout["class"] == holdout["pred_class"]).mean()

    print("CLAIR-ONLY-TRAINED MODEL -- holdout evaluation")
    print("=" * 70)
    print("Step 1 (Legitimate vs Phishing):")
    print(cm1)
    print(f"accuracy={acc1:.4f}  FPR={fpr:.4f}\n")
    print("Step 2 (attribution):")
    print(cm2)
    print(f"accuracy={acc2:.4f}\n")
    print("End-to-end 3-class:")
    print(cm3)
    print(f"accuracy={acc3:.4f}\n")

    # the actual point of this experiment: break the Human-Phishing holdout down by source
    print("=" * 70)
    print("Human-Phishing holdout, by source (the generalization question)")
    print("=" * 70)
    hp_holdout = holdout[holdout["class"] == "Human-Phishing"]
    for src_label, mask in [
        ("CLAIR (same style as training)", hp_holdout["source"].str.contains("clair", case=False, na=False)),
        ("Nazario (NEVER seen in training)", hp_holdout["source"].str.contains("nazario", case=False, na=False)),
    ]:
        sub = hp_holdout[mask]
        caught_as_phishing = (sub["pred_class"] != "Legitimate").mean()
        correctly_attributed = (sub["pred_class"] == "Human-Phishing").mean()
        print(f"{src_label} (n={len(sub)}): caught-as-phishing={caught_as_phishing:.4f}  "
              f"correctly-labeled-Human-Phishing={correctly_attributed:.4f}")

    holdout.to_csv(f"{FINAL}/holdout_predictions_clair.csv", index=False)
    summary = {"step1": {"accuracy": float(acc1), "fpr": float(fpr)},
               "step2": {"accuracy": float(acc2)}, "end_to_end": {"accuracy": float(acc3)}}
    with open(f"{FINAL}/evaluation_summary_clair.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
