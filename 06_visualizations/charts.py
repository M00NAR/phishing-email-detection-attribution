import pandas as pd, numpy as np, re, email
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP = "/root/.claude/uploads/68f35678-ae4c-5abf-a779-d10b92f77209"
NAZ = f"{UP}/c05bddcb-1791215605517_Nazario_5.csv"
ENR = f"{UP}/3c0650ed-1791215605517_emails_sample_5000.csv"
OUT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/eda"

BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "ytick.left": False, "xtick.bottom": False,
})

naz = pd.read_csv(NAZ)
naz["label_name"] = naz["label"].map({0: "Legitimate", 1: "Phishing"})
naz["word_count"] = naz["body"].fillna("").apply(lambda s: len(str(s).split()))

enr = pd.read_csv(ENR)
def parse_body(raw):
    try:
        msg = email.message_from_string(str(raw))
        if msg.is_multipart():
            return "\n".join(p.get_payload(decode=False) or "" for p in msg.walk() if p.get_content_type()=="text/plain")
        return msg.get_payload(decode=False) or ""
    except Exception:
        return ""
enr["word_count"] = enr["message"].apply(parse_body).apply(lambda s: len(str(s).split()))

# ---------- Chart A: class balance ----------
fig, ax = plt.subplots(figsize=(5, 3.8))
counts = naz["label_name"].value_counts().reindex(["Legitimate", "Phishing"])
bars = ax.bar(counts.index, counts.values, color=[BLUE, ORANGE], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v+15, f"{v:,}", ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
ax.set_ylim(0, max(counts.values)*1.15)
ax.set_title("Nazario_5.csv — class balance", color=INK, fontsize=13, fontweight="bold", loc="left")
ax.set_ylabel("Emails")
fig.tight_layout()
fig.savefig(f"{OUT}/chart_class_balance.png", dpi=180)
plt.close(fig)

# ---------- Chart B: word count distribution (log scale), 3 groups, artifacts excluded ----------
naz_clean = naz[naz["word_count"] < 20000]  # drop the 2 corrupted mbox-artifact rows
groups = {
    "Enron sample\n(legit, raw)": enr.loc[enr["word_count"]>0, "word_count"].values,
    "Nazario — Legitimate": naz_clean.loc[naz_clean.label==0, "word_count"].values,
    "Nazario — Phishing": naz_clean.loc[naz_clean.label==1, "word_count"].values,
}
fig, ax = plt.subplots(figsize=(7, 4.6))
bp = ax.boxplot(list(groups.values()), vert=True, patch_artist=True, showfliers=True,
                 widths=0.45,
                 flierprops=dict(marker='o', markersize=3, markerfacecolor=MUTED, markeredgecolor='none', alpha=0.4),
                 medianprops=dict(color=INK, linewidth=2))
colors = [AQUA, BLUE, ORANGE]
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c)
for element in ['whiskers', 'caps']:
    for line in bp[element]:
        line.set_color(MUTED)
ax.set_yscale("log")
ax.set_xticks([1,2,3], list(groups.keys()))
ax.set_ylabel("Body word count (log scale)")
ax.set_title("Email length by source and class", color=INK, fontsize=13, fontweight="bold", loc="left")
ax.text(0, 1.10, "2 corrupted mbox-artifact rows excluded from Nazario — Phishing", transform=ax.transAxes, fontsize=9, color=MUTED, ha="left", va="bottom")
fig.tight_layout()
fig.savefig(f"{OUT}/chart_word_counts.png", dpi=180)
plt.close(fig)

# ---------- Chart C: urls-column schema mismatch ----------
urls_is_list = naz["urls"].astype(str).str.startswith("[")
naz["urls_schema"] = np.where(urls_is_list, "Real list (parseable)", "Flag only ('0'/'1' string)")
tab = naz.groupby(["label_name","urls_schema"]).size().unstack(fill_value=0).reindex(["Legitimate","Phishing"])
fig, ax = plt.subplots(figsize=(6, 3.8))
x = np.arange(len(tab.index)); width = 0.35
cols = tab.columns.tolist()
for i, col in enumerate(cols):
    color = BLUE if "list" in col.lower() else ORANGE
    bars = ax.bar(x + (i-0.5)*width, tab[col].values, width, label=col, color=color)
    for b, v in zip(bars, tab[col].values):
        if v>0:
            ax.text(b.get_x()+b.get_width()/2, v+15, f"{v:,}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.set_xticks(x, tab.index)
ax.set_ylabel("Emails")
ax.set_title("'urls' column: schema differs by class", color=INK, fontsize=13, fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/chart_urls_schema.png", dpi=180)
plt.close(fig)

print("done")
