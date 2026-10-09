"""
Re-runs the full statistical gate (Kruskal-Wallis, Mann-Whitney U, Chi-Square
x2, Cohen's d, Cramer's V, Benjamini-Hochberg FDR) across ALL 26 candidate
features together: the original 20 plus the 6 new ones from
extract_new_features.py.

Why rerun the ORIGINAL 20's tests too, instead of just gating the 6 new ones
on their own: Benjamini-Hochberg FDR correction has to see every test being
run as one family to control the false-discovery rate correctly. Running the
new 6 through their own separate correction and bolting the result onto the
already-locked 20 would be the FDR equivalent of p-hacking -- testing more
things without accounting for it.

Input:  training_features_v2.csv
Output: gate_results_v2.csv, locked_features_v2.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

CONTINUOUS = [
    "avg_token_logprob", "sentence_perplexity_variance", "sentence_length_std",
    "max_parse_depth", "paragraph_density", "clause_density",
    "orthographic_error_density", "punctuation_entropy", "urgency_density",
    "pronoun_balance", "sentiment_variance", "imperative_density",
    "lexical_diversity", "pronoun_density",
    "log_word_count", "avg_word_length",  # NEW
]
CATEGORICAL = [
    "generic_greeting_flag", "subject_refw_flag", "attachment_mention_flag",
    "bracketed_cta_flag", "sensitive_info_flag", "threat_clause_flag",
    "quoted_reply_flag", "signoff_flag", "allcaps_word_flag", "multiple_urls_flag",  # NEW
]
CLASSES = ["Legitimate", "Human-Phishing", "AI-Phishing"]
Q, D_THRESH, V_THRESH = 0.05, 0.20, 0.10


def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(x1) - np.mean(x2)) / sp) if sp else 0.0


def cramers_v(chi2, n, r, c):
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def main():
    df = pd.read_csv(f"{FEAT}/training_features_v2.csv")
    phishing_only = df[df["class"].isin(["Human-Phishing", "AI-Phishing"])]

    results, raw_pvals = [], []

    for feat in CONTINUOUS:
        groups = [df.loc[df["class"] == c, feat].values for c in CLASSES]
        H, p = stats.kruskal(*groups)
        pairs = [("Legitimate", "Human-Phishing"), ("Legitimate", "AI-Phishing"), ("Human-Phishing", "AI-Phishing")]
        main_d = max(abs(cohens_d(df.loc[df["class"]==a, feat].values, df.loc[df["class"]==b, feat].values)) for a, b in pairs)
        results.append({"feature": feat, "track": "continuous", "task": "3-group", "test": "Kruskal-Wallis H",
                         "statistic": float(H), "effect_size": main_d, "is_new": feat in ("log_word_count", "avg_word_length")})
        raw_pvals.append(p)

    for feat in CONTINUOUS:
        x1 = phishing_only.loc[phishing_only["class"] == "Human-Phishing", feat].values
        x2 = phishing_only.loc[phishing_only["class"] == "AI-Phishing", feat].values
        U, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        d = cohens_d(x1, x2)
        results.append({"feature": feat, "track": "continuous", "task": "attribution", "test": "Mann-Whitney U",
                         "statistic": float(U), "effect_size": d, "is_new": feat in ("log_word_count", "avg_word_length")})
        raw_pvals.append(p)

    NEW_CAT = {"quoted_reply_flag", "signoff_flag", "allcaps_word_flag", "multiple_urls_flag"}
    for feat in CATEGORICAL:
        ct = pd.crosstab(df["class"], df[feat])
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
        results.append({"feature": feat, "track": "categorical", "task": "3-group", "test": "Chi-Square",
                         "statistic": float(chi2), "effect_size": v, "is_new": feat in NEW_CAT})
        raw_pvals.append(p)

    for feat in CATEGORICAL:
        ct = pd.crosstab(phishing_only["class"], phishing_only[feat])
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
        results.append({"feature": feat, "track": "categorical", "task": "attribution", "test": "Chi-Square (2-group)",
                         "statistic": float(chi2), "effect_size": v, "is_new": feat in NEW_CAT})
        raw_pvals.append(p)

    reject, p_adj, _, _ = multipletests(raw_pvals, alpha=Q, method="fdr_bh")
    for r, pa, rej in zip(results, p_adj, reject):
        r["p_adj"] = float(pa)
        r["significant"] = bool(rej)
        thresh = D_THRESH if r["track"] == "continuous" else V_THRESH
        r["clears_gate"] = bool(r["significant"] and abs(r["effect_size"]) >= thresh)

    pd.DataFrame(results).to_csv(f"{FEAT}/gate_results_v2.csv", index=False)

    main_gate = {r["feature"] for r in results if r["task"] == "3-group" and r["clears_gate"]}
    attr_gate = {r["feature"] for r in results if r["task"] == "attribution" and r["clears_gate"]}
    locked = {
        "continuous_main": sorted(f for f in CONTINUOUS if f in main_gate),
        "continuous_attribution": sorted(f for f in CONTINUOUS if f in attr_gate),
        "categorical_main": sorted(f for f in CATEGORICAL if f in main_gate),
        "categorical_attribution": sorted(f for f in CATEGORICAL if f in attr_gate),
    }
    with open(f"{FEAT}/locked_features_v2.json", "w") as f:
        json.dump(locked, f, indent=2)

    print(f"{len(main_gate)}/26 clear main gate, {len(attr_gate)}/26 clear attribution gate\n")
    print("--- NEW FEATURES ONLY ---")
    new_feats = {"log_word_count", "avg_word_length", "quoted_reply_flag", "signoff_flag", "allcaps_word_flag", "multiple_urls_flag"}
    rows = []
    for feat in sorted(new_feats):
        mr = next(r for r in results if r["feature"] == feat and r["task"] == "3-group")
        ar = next(r for r in results if r["feature"] == feat and r["task"] == "attribution")
        rows.append({"feature": feat, "main_p_adj": round(mr["p_adj"], 4), "main_effect": round(mr["effect_size"], 3),
                      "main_gate": mr["clears_gate"], "attr_p_adj": round(ar["p_adj"], 4),
                      "attr_effect": round(ar["effect_size"], 3), "attr_gate": ar["clears_gate"]})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
