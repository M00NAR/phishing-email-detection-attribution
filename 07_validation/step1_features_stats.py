import pandas as pd, numpy as np, re
from scipy import stats

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
t = pd.read_csv(f"{CLEAN}/training_set.csv")

def tokens(s):
    return re.findall(r"[A-Za-z']+", str(s).lower())

def sentences(s):
    parts = re.split(r'[.!?]+', str(s))
    return [p for p in parts if p.strip()]

def feats(row):
    body = str(row["body"])
    toks = tokens(body)
    n = len(toks) if toks else 1
    sents = sentences(body)
    sent_lens = [len(tokens(s)) for s in sents] if sents else [0]
    uniq = len(set(toks))
    upper_words = len(re.findall(r'\b[A-Z]{2,}\b', body))
    excl = body.count('!')
    first_person = sum(1 for w in toks if w in {"i","me","my","mine","we","us","our"})
    second_person = sum(1 for w in toks if w in {"you","your","yours"})
    return pd.Series({
        "ttr": uniq/n,
        "avg_sent_len": np.mean(sent_lens),
        "sent_len_std": np.std(sent_lens),
        "upper_word_count": upper_words,
        "excl_count": excl,
        "first_person_ratio": first_person/n,
        "second_person_ratio": second_person/n,
        "avg_word_len": np.mean([len(w) for w in toks]) if toks else 0,
    })

extra = t.apply(feats, axis=1)
t = pd.concat([t, extra], axis=1)

feature_cols = ["word_count","url_count","ttr","avg_sent_len","sent_len_std",
                 "upper_word_count","excl_count","first_person_ratio","second_person_ratio","avg_word_len"]

print("=== Kruskal-Wallis H test per feature across the 3 classes (training_set.csv) ===")
groups = {c: g for c, g in t.groupby("class")}
rows = []
for f in feature_cols:
    vals = [groups[c][f].values for c in ["Legitimate","Human-Phishing","AI-Phishing"]]
    H, p = stats.kruskal(*vals)
    rows.append((f, H, p))
res = pd.DataFrame(rows, columns=["feature","H","p"]).sort_values("H", ascending=False)
pd.set_option("display.width", 120)
print(res.to_string(index=False))

print()
print("=== per-class means for top features ===")
print(t.groupby("class")[feature_cols].mean().T)

t.to_csv(f"{CLEAN}/training_set_with_quickfeats.csv", index=False)
