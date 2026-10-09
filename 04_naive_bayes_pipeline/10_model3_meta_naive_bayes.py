"""
MODEL 3 of 3 (classification stage): Meta-Naive Bayes combiner.

Combines Model 1's posterior (rank-binned, continuous features) and Model 2's
posterior (categorical, flag features) into one final posterior over the
three classes. See proposal.tex Eq. 9.

Design note (documented here because the proposal's equation doesn't fully
pin it down on its own): each sub-model's output P_m(C_k) is ALREADY a Bayes
posterior that has one copy of the class prior baked into it. Naively
multiplying P_1(C_k) * P_2(C_k) * P(C_k) together would count the prior
THREE times. The combiner used here instead computes

    P_meta(C_k) \propto P_1(C_k) * P_2(C_k) / P(C_k)

then renormalizes over k -- the self-consistent reading of Eq. 9's
"P(C_k) * prod_m P(P_m|C_k)" once P(P_m|C_k) is read as P_m(C_k)/P(C_k).

Requires: 08 and 09 already run (reads their saved params).
Output: prints Model 1 alone, Model 2 alone, and the combined Meta-NB
        posterior side by side for 5 example rows, so the effect of
        combining is visible directly.
"""
import json
import numpy as np
import pandas as pd
import importlib.util

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


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    with open(f"{FEAT}/rank_binned_params.json") as f:
        rb = json.load(f)
    with open(f"{FEAT}/categorical_params.json") as f:
        cat = json.load(f)

    classes = ["Legitimate", "Human-Phishing", "AI-Phishing"]
    priors = rb["priors"]  # same priors used by both sub-models

    print("Model 1 alone | Model 2 alone | Meta-NB combined   (for 5 example rows)\n")
    for i in range(5):
        row = df.iloc[i]
        p1 = model1.posterior(row, rb["params"], priors, classes)
        p2 = model2.posterior(row, cat["params"], priors, classes)
        pm = meta_combine(p1, p2, priors, classes)
        print(f"true = {row['class']}")
        print(f"  Model 1 (continuous):  " + "  ".join(f"{c}={p1[c]:.3f}" for c in classes))
        print(f"  Model 2 (categorical): " + "  ".join(f"{c}={p2[c]:.3f}" for c in classes))
        print(f"  Meta-NB (combined):    " + "  ".join(f"{c}={pm[c]:.3f}" for c in classes))
        print()
