import pandas as pd, numpy as np, re, email, hashlib

UP = "/root/.claude/uploads/68f35678-ae4c-5abf-a779-d10b92f77209"
NAZ = f"{UP}/c05bddcb-1791215605517_Nazario_5.csv"
ENR = f"{UP}/3c0650ed-1791215605517_emails_sample_5000.csv"
AI  = f"{UP}/0f9f26f7-1791216010277_llm_corpus_sampled.csv"
OUT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"

RNG = 42
URL_RE = re.compile(r'https?://[^\s\'"<>\)\]]+', re.IGNORECASE)
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')
RECIPIENT_RE = re.compile(r'\bmonkey\.org\b|\bjose\b', re.IGNORECASE)

def make_id(prefix, text):
    h = hashlib.md5(str(text).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}_{h}"

def scrub_recipient_identity(s):
    # Delete (not placeholder-replace) the corpus-collection-mailbox identity
    # (jose@monkey.org) and any other email address. A visible placeholder like
    # "[RECIPIENT]" was tried first and rejected: its own frequency is itself
    # class-correlated (it would appear almost only in Human-Phishing rows),
    # so it just relocates the leak instead of removing it. Deleting the span
    # and collapsing whitespace removes the signal instead of renaming it.
    s = EMAIL_RE.sub(" ", str(s))
    s = RECIPIENT_RE.sub(" ", s)
    s = re.sub(r" {2,}", " ", s).strip()
    return s

# ============================================================
# 1. Clean Nazario_5.csv -> Legitimate + Human-Phishing
# ============================================================
naz = pd.read_csv(NAZ)
naz["word_count"] = naz["body"].fillna("").apply(lambda s: len(str(s).split()))

before = len(naz)
naz = naz[naz["word_count"] < 20000].copy()   # drop the 2 corrupted mbox-artifact rows
dropped_corrupted = before - len(naz)

naz["subject"] = naz["subject"].fillna("[no subject]")
naz["body"] = naz["body"].fillna("")
naz["url_count"] = naz["body"].apply(lambda s: len(URL_RE.findall(str(s))))  # re-derived, ignores broken 'urls' col
# scrub the corpus-collection recipient identity (jose@monkey.org) + any other email
# address, then recompute word_count from the scrubbed text so every downstream
# feature is derived from the cleaned body, not the original
naz["subject"] = naz["subject"].apply(scrub_recipient_identity)
naz["body"] = naz["body"].apply(scrub_recipient_identity)
naz["word_count"] = naz["body"].apply(lambda s: len(str(s).split()))
# some emails were template phishing blasted to many recipients and were only
# distinguishable by the now-deleted recipient address; once that's stripped
# they collapse into true duplicates, so dedupe AFTER scrubbing, not before
before_dedup = len(naz)
naz = naz.drop_duplicates(subset=["body"]).copy()
dropped_collapsed_dupes = before_dedup - len(naz)
naz["class"] = naz["label"].map({0: "Legitimate", 1: "Human-Phishing"})
naz["model"] = np.nan
naz["category"] = np.nan
naz["source"] = "nazario_corpus (mixed legit senders + human phishing)"
naz["id"] = naz["body"].apply(lambda s: make_id("naz", s))
naz_clean = naz[["id","class","subject","body","word_count","url_count","model","category","source"]].copy()

naz_clean = naz_clean[naz_clean["word_count"] > 0].copy()  # drop degenerate empty-body rows (breaks per-word feature math)
legit_pool   = naz_clean[naz_clean["class"]=="Legitimate"].reset_index(drop=True)
humphish_pool = naz_clean[naz_clean["class"]=="Human-Phishing"].reset_index(drop=True)

# ============================================================
# 2. Clean llm_corpus_sampled.csv -> AI-Phishing
# ============================================================
ai = pd.read_csv(AI)
ai = ai.dropna(subset=["body"]).drop_duplicates(subset=["body"]).copy()
ai["word_count"] = ai["body"].apply(lambda s: len(str(s).split()))
ai["url_count"] = ai["body"].apply(lambda s: len(URL_RE.findall(str(s))))
ai["class"] = "AI-Phishing"
ai["source"] = ai["model"]
ai["id"] = ai["body"].apply(lambda s: make_id("ai", s))
ai_clean = ai[["id","class","subject","body","word_count","url_count","model","category","source"]].copy()
ai_clean = ai_clean[ai_clean["word_count"] > 0].copy()
aiphish_pool = ai_clean.reset_index(drop=True)

# ============================================================
# 3. Clean emails_sample_5000.csv (raw Enron) -> held-out Legitimate
# ============================================================
enr = pd.read_csv(ENR)

def parse_msg(raw):
    try:
        msg = email.message_from_string(str(raw))
        subj = msg.get("Subject", "[no subject]")
        if msg.is_multipart():
            body = "\n".join(p.get_payload(decode=False) or "" for p in msg.walk() if p.get_content_type()=="text/plain")
        else:
            body = msg.get_payload(decode=False) or ""
        return subj, body
    except Exception:
        return "[no subject]", ""

parsed = enr["message"].apply(parse_msg)
enr["subject"] = parsed.apply(lambda t: t[0])
enr["body"] = parsed.apply(lambda t: t[1])
enr = enr[enr["body"].str.strip().str.len() > 0].copy()  # drop empty-body rows
enr["word_count"] = enr["body"].apply(lambda s: len(str(s).split()))
enr["url_count"] = enr["body"].apply(lambda s: len(URL_RE.findall(str(s))))
enr["class"] = "Legitimate"
enr["model"] = np.nan
enr["category"] = np.nan
enr["source"] = "enron_raw_sample"
enr["id"] = enr["body"].apply(lambda s: make_id("enr", s))
enr_clean = enr[["id","class","subject","body","word_count","url_count","model","category","source"]].copy()
enr_clean = enr_clean[enr_clean["word_count"] > 0].drop_duplicates(subset=["id"]).reset_index(drop=True)  # drop exact-duplicate Enron emails (common: same email saved in multiple mailboxes)

# ============================================================
# 4. Equalize class sizes for the TRAINING set
#    target N = size of the smallest of the 3 core classes
# ============================================================
target_n = min(len(legit_pool), len(humphish_pool), aiphish_pool["class"].eq("AI-Phishing").sum())
# -> bounded by Legitimate (1500)

legit_train = legit_pool.sample(n=target_n, random_state=RNG) if len(legit_pool) > target_n else legit_pool
humphish_train = humphish_pool.sample(n=target_n, random_state=RNG)

# stratified sample of AI-Phishing across (model, category) to preserve its balance
ai_pool_idx = aiphish_pool.copy()
strata = ai_pool_idx.groupby(["model","category"], group_keys=False)
n_cells = ai_pool_idx.groupby(["model","category"]).ngroups
per_cell = target_n // n_cells
remainder = target_n - per_cell * n_cells

sampled_parts = []
cells = list(ai_pool_idx.groupby(["model","category"]))
rng = np.random.RandomState(RNG)
cell_order = rng.permutation(len(cells))
for rank, idx in enumerate(cell_order):
    _, g = cells[idx]
    n_take = per_cell + (1 if rank < remainder else 0)
    n_take = min(n_take, len(g))
    sampled_parts.append(g.sample(n=n_take, random_state=RNG))
ai_train = pd.concat(sampled_parts, ignore_index=True)
# top up if stratified rounding left us short (small cells)
shortfall = target_n - len(ai_train)
if shortfall > 0:
    remaining = ai_pool_idx.drop(ai_train.index.intersection(ai_pool_idx.index), errors="ignore")
    leftover_pool = ai_pool_idx[~ai_pool_idx["id"].isin(ai_train["id"])]
    ai_train = pd.concat([ai_train, leftover_pool.sample(n=shortfall, random_state=RNG)], ignore_index=True)

training_set = pd.concat([legit_train, humphish_train, ai_train], ignore_index=True)
training_set = training_set.sample(frac=1, random_state=RNG).reset_index(drop=True)  # shuffle

# ============================================================
# 5. Build holdout set: everything NOT used in training
# ============================================================
holdout_parts = []
holdout_parts.append(legit_pool[~legit_pool["id"].isin(legit_train["id"])].assign(holdout_reason="unused Nazario-legit (rare, only if pool>target)"))
holdout_parts.append(humphish_pool[~humphish_pool["id"].isin(humphish_train["id"])].assign(holdout_reason="downsampled out of Human-Phishing for balance"))
holdout_parts.append(aiphish_pool[~aiphish_pool["id"].isin(ai_train["id"])].assign(holdout_reason="downsampled out of AI-Phishing for balance"))
holdout_parts.append(enr_clean.assign(holdout_reason="raw Enron sample, held out to test Legitimate-class generalization beyond Nazario's legit mix"))
holdout_set = pd.concat(holdout_parts, ignore_index=True)

training_set.to_csv(f"{OUT}/training_set.csv", index=False)
holdout_set.to_csv(f"{OUT}/holdout_set.csv", index=False)

# ============================================================
# Report
# ============================================================
report = {
    "dropped_corrupted_nazario_rows": dropped_corrupted,
    "dropped_collapsed_dupes_after_redaction": dropped_collapsed_dupes,
    "pool_sizes": {
        "Legitimate (Nazario)": len(legit_pool),
        "Human-Phishing (Nazario)": len(humphish_pool),
        "AI-Phishing (llm corpus, deduped)": len(aiphish_pool),
        "Legitimate (raw Enron, held out)": len(enr_clean),
    },
    "target_n_per_class": int(target_n),
    "training_set_shape": training_set.shape,
    "training_class_balance": training_set["class"].value_counts().to_dict(),
    "training_ai_model_balance": ai_train["model"].value_counts().to_dict(),
    "training_ai_category_balance": ai_train["category"].value_counts().to_dict(),
    "holdout_set_shape": holdout_set.shape,
    "holdout_class_balance": holdout_set["class"].value_counts().to_dict(),
    "holdout_reason_counts": holdout_set["holdout_reason"].value_counts().to_dict(),
}
import json
print(json.dumps(report, indent=2, default=str))
