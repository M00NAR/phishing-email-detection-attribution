"""
Builds a publication-ready chart from kruskal_wallis_full_results.csv:
a grouped horizontal bar chart of each feature's rank-sum DEVIATION from its
null-hypothesis expectation (E[R_j] = n_j*(N+1)/2), one group of 3 bars per
feature (Legitimate / Human-Phishing / AI-Phishing), sorted by H statistic so
the most significant features are at the top.

Deviation (observed - expected) rather than raw rank sum is the clearer story:
it's centered at 0, so a bar's length and direction directly show which class
is pulled toward higher or lower ranks for that feature, and by how much.

Colors: Okabe-Ito colorblind-safe triplet, fixed category order (never
cycled/reassigned): Legitimate=blue, Human-Phishing=orange, AI-Phishing=green.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FEAT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"

df = pd.read_csv(f"{FEAT}/kruskal_wallis_full_results.csv")

CLASSES = ["Legitimate", "Human-Phishing", "AI-Phishing"]
COLORS = {"Legitimate": "#0072B2", "Human-Phishing": "#E69F00", "AI-Phishing": "#009E73"}

# sort features by H statistic (most significant first), keep class order fixed within each
feat_order = (df.drop_duplicates("feature")
                .sort_values("H_statistic", ascending=True)["feature"].tolist())

n_feat = len(feat_order)
bar_h = 0.25
y_positions = {feat: i for i, feat in enumerate(feat_order)}

fig, ax = plt.subplots(figsize=(10, 9))

for i, cls in enumerate(CLASSES):
    sub = df[df["class"] == cls].set_index("feature").loc[feat_order]
    y = [y_positions[f] + (i - 1) * bar_h for f in feat_order]
    ax.barh(y, sub["deviation"], height=bar_h, color=COLORS[cls], label=cls,
            edgecolor="white", linewidth=0.3)

ax.axvline(0, color="#333333", linewidth=1)
ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(feat_order, fontsize=9)
ax.set_xlabel("Rank-sum deviation from null expectation  (observed $R_j$  $-$  expected $R_j$)")
ax.set_title("Kruskal-Wallis rank-sum deviation by class, all 16 continuous features\n"
             "(training set, $n=1{,}497$ per class, $N=4{,}491$; sorted by $H$ statistic)",
             fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x/1e3:.0f}K"))
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{FEAT}/kruskal_rank_sum_deviation.png", dpi=200)
plt.savefig(f"{FEAT}/kruskal_rank_sum_deviation.pdf")
print("Saved kruskal_rank_sum_deviation.png and .pdf")
