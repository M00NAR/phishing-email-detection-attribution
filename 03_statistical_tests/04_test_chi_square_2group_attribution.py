"""
TEST 4 of 7 (statistical screening stage): Chi-Square test of independence,
2-group attribution variant.

Applies to: all 6 CATEGORICAL (flag) features, restricted to the ATTRIBUTION
task (Human-Phishing vs AI-Phishing only). The categorical counterpart to
Test 2 (Mann-Whitney U) -- closes the gap where categorical features
previously had no dedicated attribution-stage test (proposal.tex Section 3.1).

Uses scipy's built-in implementation (scipy.stats.chi2_contingency), applied
to a 2xc contingency table instead of 3xc.

Input:  training_features.csv
Output: chi_square_2group_raw.csv
"""
import pandas as pd
from scipy import stats

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

CATEGORICAL_FEATURES = [
    "generic_greeting_flag", "subject_refw_flag", "attachment_mention_flag",
    "bracketed_cta_flag", "sensitive_info_flag", "threat_clause_flag",
]


def run_chi_square_2group(df):
    phishing_only = df[df["class"].isin(["Human-Phishing", "AI-Phishing"])]
    rows = []
    for feat in CATEGORICAL_FEATURES:
        ct = pd.crosstab(phishing_only["class"], phishing_only[feat])
        chi2, p_raw, dof, expected = stats.chi2_contingency(ct)
        rows.append({
            "feature": feat, "test": "Chi-Square (2-group, attribution)",
            "statistic": chi2, "dof": dof, "p_raw": p_raw,
            "n": int(ct.values.sum()), "n_rows": ct.shape[0], "n_cols": ct.shape[1],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    result = run_chi_square_2group(df)
    result.to_csv(f"{FEAT}/chi_square_2group_raw.csv", index=False)
    pd.set_option("display.width", 150)
    print(result.to_string(index=False))
