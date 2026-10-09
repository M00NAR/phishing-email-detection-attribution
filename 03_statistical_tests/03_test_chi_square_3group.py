"""
TEST 3 of 7 (statistical screening stage): Chi-Square test of independence,
3-group.

Applies to: all 6 CATEGORICAL (flag) features, 3-group comparison
(Legitimate vs Human-Phishing vs AI-Phishing).

What it tests: whether a feature's value (0/1 flag) is independent of class.
This is the categorical counterpart to Test 1 (Kruskal-Wallis) -- it's what
feeds Model 2 (Categorical Naive Bayes) for Step 1 screening.

Uses scipy's built-in implementation (scipy.stats.chi2_contingency).

Input:  training_features.csv
Output: chi_square_3group_raw.csv (one row per feature: chi2 statistic, dof,
        raw p-value, and the contingency table shape needed later for
        Cramer's V)
"""
import pandas as pd
from scipy import stats

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

CATEGORICAL_FEATURES = [
    "generic_greeting_flag", "subject_refw_flag", "attachment_mention_flag",
    "bracketed_cta_flag", "sensitive_info_flag", "threat_clause_flag",
]


def run_chi_square_3group(df):
    rows = []
    for feat in CATEGORICAL_FEATURES:
        ct = pd.crosstab(df["class"], df[feat])
        chi2, p_raw, dof, expected = stats.chi2_contingency(ct)
        rows.append({
            "feature": feat, "test": "Chi-Square (3-group)",
            "statistic": chi2, "dof": dof, "p_raw": p_raw,
            "n": int(ct.values.sum()), "n_rows": ct.shape[0], "n_cols": ct.shape[1],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    result = run_chi_square_3group(df)
    result.to_csv(f"{FEAT}/chi_square_3group_raw.csv", index=False)
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
