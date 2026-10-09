"""
TEST 6 of 7 (statistical screening stage): Cramer's V effect size.

Applies to: all 6 CATEGORICAL features, both tasks. The categorical
counterpart to Test 5 (Cohen's d) -- practical-significance check for
features screened by Chi-Square instead of a mean-difference test.

    V = sqrt( chi2 / (N * (min(r,c) - 1)) )

Derived directly from the SAME chi-square statistic already computed by
Test 3 (3-group) and Test 4 (2-group attribution) -- not a separate test,
a different summary of the same contingency table. Re-runs chi2_contingency
here rather than reading the earlier CSVs back in, so this script stays
runnable completely on its own.

Not a scipy built-in on its own (scipy gives you chi2, not V) -- implemented
directly from the formula in proposal.tex Eq. 6.

Input:  training_features.csv
Output: cramers_v_results.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

CATEGORICAL_FEATURES = [
    "generic_greeting_flag", "subject_refw_flag", "attachment_mention_flag",
    "bracketed_cta_flag", "sensitive_info_flag", "threat_clause_flag",
]


def cramers_v(chi2, n, r, c):
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def run_cramers_v(df):
    phishing_only = df[df["class"].isin(["Human-Phishing", "AI-Phishing"])]
    rows = []
    for feat in CATEGORICAL_FEATURES:
        ct_main = pd.crosstab(df["class"], df[feat])
        chi2_main, _, _, _ = stats.chi2_contingency(ct_main)
        v_main = cramers_v(chi2_main, ct_main.values.sum(), ct_main.shape[0], ct_main.shape[1])

        ct_attr = pd.crosstab(phishing_only["class"], phishing_only[feat])
        chi2_attr, _, _, _ = stats.chi2_contingency(ct_attr)
        v_attr = cramers_v(chi2_attr, ct_attr.values.sum(), ct_attr.shape[0], ct_attr.shape[1])

        rows.append({"feature": feat, "main_effect_size_V": v_main, "attribution_effect_size_V": v_attr})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    result = run_cramers_v(df)
    result.to_csv(f"{FEAT}/cramers_v_results.csv", index=False)
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
