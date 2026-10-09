"""
MODEL 1 of 3 (classification stage): Rank-Binned Naive Bayes.

Handles the continuous features that cleared the statistical gate (Test 1/2/5
above). "Training" = computing quantile bin edges from the training data,
then counting how many training emails of each class fall into each bin.
No iteration, no gradient descent -- see proposal.tex Eq. 7:

    P(x_i in Bin_b | C_k) = (N_{b,k} + alpha) / (N_k + alpha*B)

with Laplace smoothing alpha=1 so no bin/class combination ever hits exactly
zero probability.

This script fits and demonstrates Model 1 ALONE (its own posterior, not yet
combined with Model 2) -- see 10_model3_meta_naive_bayes.py for where the
two get combined.

Requires: 07_fdr_correction_and_gate.py already run (reads locked_features.json)
Output: rank_binned_params.json
"""
import json
import numpy as np
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
N_BINS = 8
ALPHA = 1.0


def make_bin_edges(values, n_bins=N_BINS):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges.tolist()


def assign_bin(value, edges):
    b = int(np.digitize([value], edges[1:-1])[0])
    return min(b, len(edges) - 2)


def fit(df, features, classes, class_col="class"):
    """Returns {feature: {'edges': [...], 'freq': {class: [P(bin0), P(bin1), ...]}}}"""
    params = {}
    for feat in features:
        edges = make_bin_edges(df[feat].values)
        n_bins = len(edges) - 1
        freq = {}
        for c in classes:
            vals = df.loc[df[class_col] == c, feat].values
            counts = np.zeros(n_bins)
            for v in vals:
                counts[assign_bin(v, edges)] += 1
            freq[c] = ((counts + ALPHA) / (counts.sum() + ALPHA * n_bins)).tolist()
        params[feat] = {"edges": edges, "freq": freq}
    return params


def posterior(row, params, priors, classes, skip_features=None):
    """skip_features: feature names to exclude from this row's posterior (the
    length gate uses this -- see 12_length_gate_and_rerun.py -- for features
    whose value is degenerate/uninformative on very short emails rather than
    genuinely predictive)."""
    skip_features = skip_features or set()
    log_post = {c: np.log(priors[c]) for c in classes}
    for feat, p in params.items():
        if feat in skip_features:
            continue
        b = assign_bin(row[feat], p["edges"])
        for c in classes:
            log_post[c] += np.log(p["freq"][c][b])
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
    params = fit(df, locked["continuous_main"], classes)

    with open(f"{FEAT}/rank_binned_params.json", "w") as f:
        json.dump({"features": locked["continuous_main"], "priors": priors, "params": params}, f, indent=2)

    # demonstrate on the first 5 training rows
    print("Model 1 (Rank-Binned NB) alone, posterior over classes for 5 example rows:")
    for i in range(5):
        row = df.iloc[i]
        post = posterior(row, params, priors, classes)
        print(f"  true={row['class']:16s}  " + "  ".join(f"{c}={p:.3f}" for c, p in post.items()))
