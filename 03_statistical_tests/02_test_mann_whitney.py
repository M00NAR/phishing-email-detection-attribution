"""
TEST 2 of 7 (statistical screening stage): Mann-Whitney U test.

Applies to: all 14 CONTINUOUS features, restricted to the ATTRIBUTION task
(Human-Phishing vs AI-Phishing only -- Legitimate rows excluded).

What it tests: the continuous counterpart to the 3-group Kruskal-Wallis test,
but pairwise and phishing-only -- whether a feature separates human-written
phishing from AI-generated phishing specifically. This is Step 2's screening
test, as opposed to Test 1 which screens for Step 1.

Uses scipy's built-in implementation (scipy.stats.mannwhitneyu).

Input:  training_features.csv
Output: mann_whitney_raw.csv (one row per feature: U statistic, raw p-value)
"""
import pandas as pd
from scipy import stats

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

CONTINUOUS_FEATURES = [
    "avg_token_logprob", "sentence_perplexity_variance", "sentence_length_std",
    "max_parse_depth", "paragraph_density", "clause_density",
    "orthographic_error_density", "punctuation_entropy", "urgency_density",
    "pronoun_balance", "sentiment_variance", "imperative_density",
    "lexical_diversity", "pronoun_density",
]


def run_mann_whitney(df):
    phishing_only = df[df["class"].isin(["Human-Phishing", "AI-Phishing"])]
    rows = []
    for feat in CONTINUOUS_FEATURES:
        x1 = phishing_only.loc[phishing_only["class"] == "Human-Phishing", feat].values
        x2 = phishing_only.loc[phishing_only["class"] == "AI-Phishing", feat].values
        U, p_raw = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        rows.append({"feature": feat, "test": "Mann-Whitney U", "statistic": U, "p_raw": p_raw})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    result = run_mann_whitney(df)
    result.to_csv(f"{FEAT}/mann_whitney_raw.csv", index=False)
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
