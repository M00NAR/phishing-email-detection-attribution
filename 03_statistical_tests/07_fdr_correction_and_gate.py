"""
TEST 7 of 7 (statistical screening stage): Benjamini-Hochberg FDR correction
+ the final dual-gate decision.

This is the step that ties Tests 1-6 together, and it HAS to run after all
four p-value tests (01-04), not standalone -- that's the whole point of FDR
correction: adjusting p-values JOINTLY across every test run, not one at a
time. Run 40 independent tests at raw p<0.05 and you'd expect ~2 false
positives by chance alone, even if nothing were really different between
classes. Correcting each test's p-value in isolation doesn't fix this; the
correction has to see the full set of 40 raw p-values at once.

    P_(i) <= (i/m) * Q      (Benjamini-Hochberg, Q = 0.05)

Uses statsmodels' built-in implementation (statsmodels.stats.multitest.
multipletests, method="fdr_bh") -- this script applies it correctly across
ALL 40 raw p-values together (14 Kruskal-Wallis + 14 Mann-Whitney + 6
Chi-Square-3group + 6 Chi-Square-2group), not reimplementing the correction
math by hand.

A feature clears the gate only if BOTH:
  - adjusted p < 0.05 (statistically real, after correction)
  - effect size >= threshold (|d| >= 0.20 continuous, V >= 0.10 categorical)
    -- practically real, not just real because of sample size.

Requires: 01-06 already run (reads their output CSVs).
Output: gate_results_full.csv (every feature/task/test with adjusted p,
        effect size, and the gate decision) and locked_features.json (the
        feature lists that actually feed the Naive Bayes models next).
"""
import json
import pandas as pd
from statsmodels.stats.multitest import multipletests

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
Q = 0.05
D_THRESH = 0.20
V_THRESH = 0.10


def main():
    kw = pd.read_csv(f"{FEAT}/kruskal_wallis_raw.csv")
    kw["track"], kw["task"] = "continuous", "3-group"
    mwu = pd.read_csv(f"{FEAT}/mann_whitney_raw.csv")
    mwu["track"], mwu["task"] = "continuous", "attribution"
    chi3 = pd.read_csv(f"{FEAT}/chi_square_3group_raw.csv")
    chi3["track"], chi3["task"] = "categorical", "3-group"
    chi2 = pd.read_csv(f"{FEAT}/chi_square_2group_raw.csv")
    chi2["track"], chi2["task"] = "categorical", "attribution"

    all_tests = pd.concat([kw, mwu, chi3, chi2], ignore_index=True, sort=False)

    # ---- joint FDR correction across all 40 raw p-values at once ----
    reject, p_adj, _, _ = multipletests(all_tests["p_raw"].values, alpha=Q, method="fdr_bh")
    all_tests["p_adj"] = p_adj
    all_tests["significant"] = reject

    # ---- attach effect sizes computed by scripts 05/06 ----
    d = pd.read_csv(f"{FEAT}/cohens_d_results.csv")[["feature", "main_effect_size_d", "attribution_effect_size_d"]]
    v = pd.read_csv(f"{FEAT}/cramers_v_results.csv")[["feature", "main_effect_size_V", "attribution_effect_size_V"]]

    def get_effect_size(row):
        if row["track"] == "continuous":
            col = "main_effect_size_d" if row["task"] == "3-group" else "attribution_effect_size_d"
            return abs(d.loc[d["feature"] == row["feature"], col].values[0])
        else:
            col = "main_effect_size_V" if row["task"] == "3-group" else "attribution_effect_size_V"
            return abs(v.loc[v["feature"] == row["feature"], col].values[0])

    all_tests["effect_size"] = all_tests.apply(get_effect_size, axis=1)
    all_tests["effect_threshold"] = all_tests["track"].map({"continuous": D_THRESH, "categorical": V_THRESH})
    all_tests["clears_gate"] = all_tests["significant"] & (all_tests["effect_size"] >= all_tests["effect_threshold"])

    all_tests.to_csv(f"{FEAT}/gate_results_full.csv", index=False)

    main_gate = set(all_tests.loc[(all_tests["task"] == "3-group") & all_tests["clears_gate"], "feature"])
    attr_gate = set(all_tests.loc[(all_tests["task"] == "attribution") & all_tests["clears_gate"], "feature"])
    continuous = kw["feature"].tolist()
    categorical = chi3["feature"].tolist()

    locked = {
        "continuous_main": sorted(f for f in continuous if f in main_gate),
        "continuous_attribution": sorted(f for f in continuous if f in attr_gate),
        "categorical_main": sorted(f for f in categorical if f in main_gate),
        "categorical_attribution": sorted(f for f in categorical if f in attr_gate),
    }
    with open(f"{FEAT}/locked_features.json", "w") as f:
        json.dump(locked, f, indent=2)

    print(f"{len(main_gate)}/20 features clear the main gate")
    print(f"{len(attr_gate)}/20 features clear the attribution gate")
    print(json.dumps(locked, indent=2))


if __name__ == "__main__":
    main()
