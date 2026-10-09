"""
FINAL STAGE: the two-step decision rule, end to end, evaluated on the
holdout set. This is what "running the trained model" actually means --
everything in files 01-10 built the pieces; this file assembles and scores
them.

Builds TWO full pipelines (Model1+Model2+Model3 each):
  - MAIN pipeline: 3-class (Legitimate/Human-Phishing/AI-Phishing), using the
    main-gate feature set, trained on the full training set. Produces Step 1.
  - ATTRIBUTION pipeline: 2-class (Human-Phishing/AI-Phishing), using the
    attribution-gate feature set, trained ONLY on phishing training rows.
    Produces Step 2, and only runs when Step 1 says "Phishing".

Step 1 (proposal.tex Eq. 10):
    P(Phishing) = P(Human-Phishing) + P(AI-Phishing)
    < 0.50 -> Legitimate; tie at exactly 0.50 -> Phishing (fail-safe)

Step 2 (proposal.tex Eq. 11), only if Step 1 said Phishing:
    P(AI | Phishing) = P(AI-Phishing) / (P(AI-Phishing) + P(Human-Phishing))
    >= 0.50 -> AI-Generated; tie at exactly 0.50 -> AI-Generated (conservative)

Requires: 07 already run. Builds its own models directly (imports the fit/
posterior functions from 08 and 09 rather than reusing locked_model.json),
so this one file is a complete, standalone trace from locked features to
final holdout numbers.
"""
import json
import importlib.util
import numpy as np
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"


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


def predict_posterior(row, pipeline):
    classes, priors = pipeline["classes"], pipeline["priors"]
    p1 = model1.posterior(row, pipeline["rb_params"], priors, classes) if pipeline["continuous_feats"] else priors
    p2 = model2.posterior(row, pipeline["cat_params"], priors, classes) if pipeline["categorical_feats"] else priors
    return meta_combine(p1, p2, priors, classes)


def confusion(y_true, y_pred, labels):
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(y_true, y_pred):
        cm.loc[t, p] += 1
    return cm


if __name__ == "__main__":
    train = pd.read_csv(f"{FEAT}/training_features.csv")
    holdout = pd.read_csv(f"{FEAT}/holdout_features.csv")
    with open(f"{FEAT}/locked_features.json") as f:
        locked = json.load(f)

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

    # ---- Step 1 scored alone ----
    true_phish = holdout["class"] != "Legitimate"
    pred_phish = holdout["pred_class"] != "Legitimate"
    cm1 = confusion(true_phish.map({True: "Phishing", False: "Legitimate"}),
                     pred_phish.map({True: "Phishing", False: "Legitimate"}), ["Legitimate", "Phishing"])
    acc1 = (true_phish == pred_phish).mean()
    print("STEP 1 (Legitimate vs Phishing):")
    print(cm1)
    print(f"accuracy: {acc1:.4f}\n")

    # ---- Step 2 scored in isolation on true-phishing rows, so Step 1 errors don't leak in ----
    tp_rows = holdout[true_phish].copy()
    tp_rows["attr_pred"] = tp_rows.apply(
        lambda r: ("AI-Phishing" if (lambda ap: ap["AI-Phishing"] / (ap["AI-Phishing"] + ap["Human-Phishing"]))(
            predict_posterior(r, attr_pipeline)) >= 0.50 else "Human-Phishing"), axis=1)
    cm2 = confusion(tp_rows["class"], tp_rows["attr_pred"], ["Human-Phishing", "AI-Phishing"])
    acc2 = (tp_rows["class"] == tp_rows["attr_pred"]).mean()
    print("STEP 2 (Attribution, scored on true-phishing rows only):")
    print(cm2)
    print(f"accuracy: {acc2:.4f}\n")

    # ---- end to end ----
    cm3 = confusion(holdout["class"], holdout["pred_class"], main_classes)
    acc3 = (holdout["class"] == holdout["pred_class"]).mean()
    print("END-TO-END 3-class:")
    print(cm3)
    print(f"accuracy: {acc3:.4f}")
