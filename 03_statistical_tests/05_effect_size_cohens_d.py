"""
TEST 5 of 7 (statistical screening stage): Cohen's d effect size.

Applies to: all 14 CONTINUOUS features, both tasks.

What it measures: practical significance, not just statistical significance
-- a p-value alone can be tiny just from having a lot of data, even when the
actual difference between groups is too small to matter. Cohen's d is
(difference in means) / (pooled standard deviation): how many standard
deviations apart the two groups' averages are.

    d = (x1_bar - x2_bar) / s_pooled
    s_pooled = sqrt( ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2) )

This is NOT a scipy built-in (scipy doesn't ship a Cohen's d function) --
implemented directly from the formula in proposal.tex Eq. 4.

For the 3-group MAIN task there isn't a single "the" d (3 classes = 3
possible pairs), so the reported effect size is the LARGEST of the three
pairwise |d| values -- if ANY pair is well separated, the feature is doing
real work. For the ATTRIBUTION task there are only 2 classes, so it's the
one, directly meaningful value.

Input:  training_features.csv
Output: cohens_d_results.csv
"""
import numpy as np
import pandas as pd

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

CONTINUOUS_FEATURES = [
    "avg_token_logprob", "sentence_perplexity_variance", "sentence_length_std",
    "max_parse_depth", "paragraph_density", "clause_density",
    "orthographic_error_density", "punctuation_entropy", "urgency_density",
    "pronoun_balance", "sentiment_variance", "imperative_density",
    "lexical_diversity", "pronoun_density",
]
CLASSES = ["Legitimate", "Human-Phishing", "AI-Phishing"]


def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / s_pooled)


def run_cohens_d(df):
    phishing_only = df[df["class"].isin(["Human-Phishing", "AI-Phishing"])]
    rows = []
    for feat in CONTINUOUS_FEATURES:
        # main task: largest pairwise |d| across the 3 classes
        pairs = [("Legitimate", "Human-Phishing"), ("Legitimate", "AI-Phishing"), ("Human-Phishing", "AI-Phishing")]
        pairwise_ds = {
            f"{a}_vs_{b}": cohens_d(df.loc[df["class"] == a, feat].values, df.loc[df["class"] == b, feat].values)
            for a, b in pairs
        }
        main_effect = max(abs(v) for v in pairwise_ds.values())

        # attribution task: Human-Phishing vs AI-Phishing, signed
        attr_effect = cohens_d(
            phishing_only.loc[phishing_only["class"] == "Human-Phishing", feat].values,
            phishing_only.loc[phishing_only["class"] == "AI-Phishing", feat].values,
        )
        rows.append({
            "feature": feat,
            "main_effect_size_d": main_effect,
            "attribution_effect_size_d": attr_effect,
            **{f"d_{k}": v for k, v in pairwise_ds.items()},
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    result = run_cohens_d(df)
    result.to_csv(f"{FEAT}/cohens_d_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(result[["feature", "main_effect_size_d", "attribution_effect_size_d"]].to_string(index=False))
