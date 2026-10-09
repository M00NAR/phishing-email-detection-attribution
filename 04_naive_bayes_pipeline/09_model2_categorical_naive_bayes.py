"""
MODEL 2 of 3 (classification stage): Categorical Naive Bayes.

Handles the categorical (flag) features that cleared the statistical gate.
"Training" = counting how often each category value (0 or 1) appears per
class. See proposal.tex Eq. 8:

    P(x_i = v | C_k) = (Count(x_i=v in C_k) + alpha) / (Total in C_k + alpha*K)

with Laplace smoothing alpha=1. An unseen category value at inference time
falls back to this same smoothed baseline rather than a zero probability.

This script fits and demonstrates Model 2 ALONE -- see
10_model3_meta_naive_bayes.py for where it's combined with Model 1.

Requires: 07_fdr_correction_and_gate.py already run (reads locked_features.json)
Output: categorical_params.json
"""
import json
import numpy as np
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
ALPHA = 1.0


def fit(df, features, classes, class_col="class"):
    """Returns {feature: {class: [P(value=0), P(value=1)]}}"""
    params = {}
    for feat in features:
        freq = {}
        for c in classes:
            vals = df.loc[df[class_col] == c, feat].values
            n0 = int(np.sum(vals == 0))
            n1 = int(np.sum(vals == 1))
            total = n0 + n1
            freq[c] = [(n0 + ALPHA) / (total + ALPHA * 2), (n1 + ALPHA) / (total + ALPHA * 2)]
        params[feat] = freq
    return params


def posterior(row, params, priors, classes):
    log_post = {c: np.log(priors[c]) for c in classes}
    for feat, freq in params.items():
        v = int(row[feat])
        for c in classes:
            log_post[c] += np.log(freq[c][v])
    m = max(log_post.values())
    exps = {c: np.exp(log_post[c] - m) for c in classes}
    s = sum(exps.values())
    return {c: exps[c] / s for c in classes}


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    with open(f"{FEAT}/locked_features.json") as f:
        locked = json.load(f)

    classes = ["Legitimate", "Human-Phishing", "AI-Phishing"]
    priors = {c: df["class"].eq(c).mean() for c in classes}
    params = fit(df, locked["categorical_main"], classes)

    with open(f"{FEAT}/categorical_params.json", "w") as f:
        json.dump({"features": locked["categorical_main"], "priors": priors, "params": params}, f, indent=2)

    print("Model 2 (Categorical NB) alone, posterior over classes for 5 example rows:")
    for i in range(5):
        row = df.iloc[i]
        post = posterior(row, params, priors, classes)
        print(f"  true={row['class']:16s}  " + "  ".join(f"{c}={p:.3f}" for c, p in post.items()))
