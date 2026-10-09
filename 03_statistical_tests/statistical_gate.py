"""
Phase 1, Stage 2: Statistical Screening -- run on the REAL extracted features
in training_features.csv (not the hand-built proxy features used earlier to
validate the approach). Mirrors proposal.tex Section 3 exactly:

  - Continuous features: Kruskal-Wallis H (3-group) + Mann-Whitney U
    (Human-Phishing vs AI-Phishing, attribution) + Cohen's d.
  - Categorical features: Chi-Square (3-group) + Chi-Square (2-group,
    attribution) + Cramer's V.
  - All raw p-values (both tracks, both tasks) corrected JOINTLY with
    Benjamini-Hochberg FDR (Q=0.05).
  - A feature clears the gate only if adjusted p < 0.05 AND effect size
    >= threshold (|d| >= 0.20 continuous, V >= 0.10 categorical).

Output: gate_results.json (full stats for every feature/task) and
locked_features.json (just the feature names that cleared BOTH the main
3-group gate and, separately, which also clear the attribution gate --
Phase 2 needs to know which features are usable for which decision step).
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
]
CATEGORICAL = [
    "generic_greeting_flag", "subject_refw_flag", "attachment_mention_flag",
    "bracketed_cta_flag", "sensitive_info_flag", "threat_clause_flag",
]

Q = 0.05
D_THRESH = 0.20
V_THRESH = 0.10


def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / s_pooled)


def cramers_v(chi2, n, r, c):
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def main():
    df = pd.read_csv(f"{FEAT}/training_features.csv")
    classes = ["Legitimate", "Human-Phishing", "AI-Phishing"]
    phishing_only = df[df["class"].isin(["Human-Phishing", "AI-Phishing"])]

    raw_pvals = []
    results = []  # list of dicts, filled in, matched index-for-index with raw_pvals

    # ---------- continuous: 3-group Kruskal-Wallis ----------
    for feat in CONTINUOUS:
        groups = [df.loc[df["class"] == c, feat].values for c in classes]
        H, p = stats.kruskal(*groups)
        results.append({"feature": feat, "track": "continuous", "task": "3-group",
                         "test": "Kruskal-Wallis H", "statistic": float(H)})
        raw_pvals.append(p)

    # ---------- continuous: attribution Mann-Whitney U ----------
    for feat in CONTINUOUS:
        x1 = phishing_only.loc[phishing_only["class"] == "Human-Phishing", feat].values
        x2 = phishing_only.loc[phishing_only["class"] == "AI-Phishing", feat].values
        U, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        d = cohens_d(x1, x2)
        results.append({"feature": feat, "track": "continuous", "task": "attribution",
                         "test": "Mann-Whitney U", "statistic": float(U), "effect_size": d})
        raw_pvals.append(p)

    # ---------- categorical: 3-group Chi-Square ----------
    for feat in CATEGORICAL:
        ct = pd.crosstab(df["class"], df[feat])
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
        results.append({"feature": feat, "track": "categorical", "task": "3-group",
                         "test": "Chi-Square", "statistic": float(chi2), "effect_size": v})
        raw_pvals.append(p)

    # ---------- categorical: attribution 2-group Chi-Square ----------
    for feat in CATEGORICAL:
        ct = pd.crosstab(phishing_only["class"], phishing_only[feat])
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
        results.append({"feature": feat, "track": "categorical", "task": "attribution",
                         "test": "Chi-Square (2-group)", "statistic": float(chi2), "effect_size": v})
        raw_pvals.append(p)

    # ---------- continuous 3-group also needs an effect size for the gate;
    #            use max pairwise Cohen's d across the 3 classes (Legit vs HP, Legit vs AI, HP vs AI) ----------
    for i, feat in enumerate(CONTINUOUS):
        pairs = [("Legitimate", "Human-Phishing"), ("Legitimate", "AI-Phishing"), ("Human-Phishing", "AI-Phishing")]
        ds = []
        for a, b in pairs:
            ds.append(abs(cohens_d(df.loc[df["class"] == a, feat].values, df.loc[df["class"] == b, feat].values)))
        results[i]["effect_size"] = float(max(ds))

    # ---------- joint Benjamini-Hochberg FDR across ALL raw p-values ----------
    reject, p_adj, _, _ = multipletests(raw_pvals, alpha=Q, method="fdr_bh")
    for r, pa, rej, p0 in zip(results, p_adj, reject, raw_pvals):
        r["p_raw"] = float(p0)
        r["p_adj"] = float(pa)
        r["significant"] = bool(rej)

    # ---------- apply the dual gate ----------
    # NOTE: Cohen's d for the attribution task is signed (direction is meaningful for
    # reporting -- which class has the higher value) but the GATE must compare magnitude,
    # since a large negative d is just as real a difference as a large positive one.
    # Cramer's V is non-negative by construction so abs() is a no-op there.
    for r in results:
        thresh = D_THRESH if r["track"] == "continuous" else V_THRESH
        r["effect_size_threshold"] = thresh
        r["clears_gate"] = bool(r["significant"] and abs(r.get("effect_size", 0)) >= thresh)

    with open(f"{FEAT}/gate_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------- build locked feature list: main-task and attribution-task separately ----------
    main_gate = {r["feature"] for r in results if r["task"] == "3-group" and r["clears_gate"]}
    attr_gate = {r["feature"] for r in results if r["task"] == "attribution" and r["clears_gate"]}

    locked = {
        "continuous_main": sorted(f for f in CONTINUOUS if f in main_gate),
        "continuous_attribution": sorted(f for f in CONTINUOUS if f in attr_gate),
        "categorical_main": sorted(f for f in CATEGORICAL if f in main_gate),
        "categorical_attribution": sorted(f for f in CATEGORICAL if f in attr_gate),
    }
    with open(f"{FEAT}/locked_features.json", "w") as f:
        json.dump(locked, f, indent=2)

    print(json.dumps(locked, indent=2))
    print(f"\n{len(main_gate)}/{len(CONTINUOUS)+len(CATEGORICAL)} features clear the main (Legit vs Phishing) gate")
    print(f"{len(attr_gate)}/{len(CONTINUOUS)+len(CATEGORICAL)} features clear the attribution (Human vs AI) gate")

    # print a readable summary table
    rows = []
    for feat in CONTINUOUS + CATEGORICAL:
        main_r = next(r for r in results if r["feature"] == feat and r["task"] == "3-group")
        attr_r = next(r for r in results if r["feature"] == feat and r["task"] == "attribution")
        rows.append({
            "feature": feat,
            "main_p_adj": round(main_r["p_adj"], 4), "main_effect": round(main_r["effect_size"], 3),
            "main_gate": main_r["clears_gate"],
            "attr_p_adj": round(attr_r["p_adj"], 4), "attr_effect": round(attr_r["effect_size"], 3),
            "attr_gate": attr_r["clears_gate"],
        })
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False))
    summary.to_csv(f"{FEAT}/gate_summary.csv", index=False)


if __name__ == "__main__":
    main()
