"""
TEST 1 of 7 (statistical screening stage): Kruskal-Wallis H-test.

Applies to: all 14 CONTINUOUS features, 3-group comparison
(Legitimate vs Human-Phishing vs AI-Phishing).

What it tests: whether the three classes' RANKS differ for a given feature --
a distribution-free alternative to one-way ANOVA. Used instead of ANOVA
because email-derived linguistic features routinely violate normality
(heavy skew, outliers, bounded counts) -- see proposal.tex Section 3.

Uses scipy's built-in implementation (scipy.stats.kruskal) -- this script
is about applying and reporting it correctly for every feature, not
reimplementing the math.

Input:  training_features.csv (from extract_features.py)
Output: kruskal_wallis_raw.csv (one row per feature: H statistic, raw p-value)
        -- raw p-values only. FDR correction happens later, jointly across
        ALL tests (see 05_fdr_correction.py), not here.
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
CLASSES = ["Legitimate", "Human-Phishing", "AI-Phishing"]


def run_kruskal_wallis(df):
    rows = []
    for feat in CONTINUOUS_FEATURES:
        groups = [df.loc[df["class"] == c, feat].values for c in CLASSES]
        H, p_raw = stats.kruskal(*groups)
        rows.append({"feature": feat, "test": "Kruskal-Wallis H", "statistic": H, "p_raw": p_raw})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    result = run_kruskal_wallis(df)
    result.to_csv(f"{FEAT}/kruskal_wallis_raw.csv", index=False)
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
