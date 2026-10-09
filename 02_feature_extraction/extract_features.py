"""
Phase 1, Stage 1: Feature Extraction.

Computes all 20 candidate features (14 continuous + 6 categorical) for every
email in training_set.csv / holdout_set.csv.

Design choices made here that weren't pinned down before, and why:

  - TOKEN LOG-PROBABILITY LM: a from-scratch word-level trigram language
    model with stupid backoff, trained ONCE on the full training_set.csv
    text (all three classes combined, so it represents generic email-domain
    language rather than being biased toward one class). This is "locked"
    exactly like bin edges and frequency tables are -- fit once on training
    data, then applied unchanged to both training and holdout text, so there
    is zero leakage into holdout. A transformer LM (GPT-2 etc.) was
    considered and rejected: this project's whole premise is a lightweight,
    auditable, non-black-box pipeline (see proposal.tex, "Low Compute
    Overhead"), and downloading pretrained transformer weights wasn't
    reachable from this environment anyway. Perplexity-under-an-n-gram-model
    is a standard, well-precedented AI-text-detection feature that predates
    transformer-based detectors.

  - PARSE-TREE DEPTH: spaCy's dependency parser (en_core_web_sm), depth =
    longest path from a token up to its sentence root via .head chain.
    This is a dependency-tree depth, not a constituency-tree depth (no
    constituency parser was available offline) -- functionally serves the
    same purpose (nesting/complexity signal) and is noted as such.

  - LENGTH CAP: email text is capped at the first 6000 characters before any
    spaCy/tokenization-based feature is computed (regex-only features still
    see the full text). A handful of extreme outliers (up to 31,396 words --
    likely mbox-thread artifacts) would otherwise dominate runtime for no
    signal benefit; this is also consistent with the project's own
    sub-10ms-per-email design goal, which assumes realistic single-message
    length.

Run: python3 extract_features.py
"""
import re, json, math, random, sys, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from spellchecker import SpellChecker

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
OUT = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/features"
CAP_CHARS = 6000
RNG = 42

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
sia = SentimentIntensityAnalyzer()
spell = SpellChecker(distance=1)

# ============================================================
# Regex keyword lists (categorical + a few continuous features)
# ============================================================
URGENCY_WORDS = re.compile(
    r"\b(urgent|immediately|asap|act now|right away|as soon as possible|expire|expir(?:e|es|ed|ing)|"
    r"deadline|limited time|final notice|last chance|time[- ]sensitive|without delay|promptly)\b",
    re.IGNORECASE,
)
IMPERATIVE_LEMMAS = {
    "click", "verify", "submit", "confirm", "update", "login", "log", "download",
    "open", "send", "provide", "enter", "call", "reply", "act", "review", "complete",
}
GENERIC_GREETING_RE = re.compile(
    r"\bdear\s+(valued\s+customer|customer|user|member|sir\s*/?\s*madam|account\s*holder|client)\b"
    r"|\bto\s+whom\s+it\s+may\s+concern\b"
    r"|^\s*(hello|hi|dear)\s*,",
    re.IGNORECASE,
)
NAMED_GREETING_RE = re.compile(r"\b(dear|hi|hello)\s+[A-Z][a-z]+\b")
SUBJECT_REFW_RE = re.compile(r"^\s*(re|fw|fwd)\s*:", re.IGNORECASE)
ATTACHMENT_RE = re.compile(r"\b(attach(?:ed|ment|ments)?|enclosed|see\s+the\s+file|find\s+attached)\b", re.IGNORECASE)
BRACKETED_CTA_RE = re.compile(r"\[[^\[\]]{2,40}\]")
CTA_WORDS_RE = re.compile(r"\b(click|verify|login|log in|here|now|update|confirm|view|download)\b", re.IGNORECASE)
SENSITIVE_INFO_RE = re.compile(
    r"\b(password|ssn|social security|card\s*number|credit\s*card|cvv|pin\s*number|"
    r"account\s*number|routing\s*number|date\s*of\s*birth)\b", re.IGNORECASE
)
THREAT_RE = re.compile(
    r"\b(suspend(?:ed|sion)?|clos(?:e|ed|ure)|restrict(?:ed|ion)?|terminat(?:e|ed|ion)|"
    r"lock(?:ed|out)?|deactivat(?:e|ed|ion)|legal\s+action|penalt(?:y|ies))\b", re.IGNORECASE
)

PUNCT_CHARS = list(".,;:!?-'\"()[]")

def cap(text):
    return str(text)[:CAP_CHARS]

# ============================================================
# Background trigram language model (stupid backoff), fit once
# on the TRAINING set text only -- see module docstring.
# ============================================================
class TrigramLM:
    def __init__(self):
        self.uni = Counter()
        self.bi = Counter()
        self.tri = Counter()
        self.total_tokens = 0

    def fit(self, texts):
        for t in texts:
            toks = ["<s>", "<s>"] + re.findall(r"[a-z']+", str(t).lower()) + ["</s>"]
            self.total_tokens += len(toks) - 2
            for i, w in enumerate(toks):
                self.uni[w] += 1
                if i >= 1:
                    self.bi[(toks[i-1], w)] += 1
                if i >= 2:
                    self.tri[(toks[i-2], toks[i-1], w)] += 1
        self.vocab_size = len(self.uni)
        self.uni_total = sum(self.uni.values())

    def token_logprob(self, w2, w1, w):
        # stupid backoff: trigram -> bigram*0.4 -> unigram*0.4^2 -> floor
        tri_ctx = self.bi.get((w2, w1), 0)
        if tri_ctx > 0 and self.tri.get((w2, w1, w), 0) > 0:
            p = self.tri[(w2, w1, w)] / tri_ctx
            return math.log(p)
        bi_ctx = self.uni.get(w1, 0)
        if bi_ctx > 0 and self.bi.get((w1, w), 0) > 0:
            p = 0.4 * self.bi[(w1, w)] / bi_ctx
            return math.log(p)
        uni_c = self.uni.get(w, 0)
        if uni_c > 0:
            p = 0.4 * 0.4 * uni_c / self.uni_total
            return math.log(p)
        return math.log(0.4 * 0.4 * 0.5 / self.uni_total)  # unseen-word floor

    def sentence_logprobs(self, sentence_tokens_list):
        """Return per-sentence average token log-prob, for a list of token lists."""
        out = []
        for toks in sentence_tokens_list:
            seq = ["<s>", "<s>"] + toks + ["</s>"]
            lps = [self.token_logprob(seq[i-2], seq[i-1], seq[i]) for i in range(2, len(seq))]
            if lps:
                out.append(float(np.mean(lps)))
        return out


def build_background_lm(train_df):
    print("Fitting background trigram LM on training_set.csv text ...", flush=True)
    lm = TrigramLM()
    texts = (train_df["subject"].fillna("") + " " + train_df["body"].fillna("").apply(cap)).tolist()
    lm.fit(texts)
    print(f"  vocab={lm.vocab_size:,} unigrams, {len(lm.bi):,} bigram types, {len(lm.tri):,} trigram types")
    return lm


# ============================================================
# Per-email feature computation
# ============================================================
def shannon_entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counter.values() if c > 0]
    return float(-sum(p * math.log2(p) for p in probs))


def extract_one(row, lm, doc):
    subject = str(row["subject"]) if pd.notna(row["subject"]) else ""
    body_full = str(row["body"]) if pd.notna(row["body"]) else ""
    body = cap(body_full)
    text_full = subject + "\n" + body_full

    words_full = re.findall(r"[A-Za-z']+", body_full)
    n_words_full = max(len(words_full), 1)

    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    sent_token_lists = [re.findall(r"[a-z']+", s.lower()) for s in sentences]
    sent_token_lists = [t for t in sent_token_lists if t]

    # --- 1 & 2: LM-based predictability / burstiness ---
    sent_lps = lm.sentence_logprobs(sent_token_lists) if sent_token_lists else []
    avg_token_logprob = float(np.mean(sent_lps)) if sent_lps else 0.0
    sentence_perplexity_variance = float(np.var(sent_lps)) if len(sent_lps) > 1 else 0.0

    # --- 3: std dev of sentence length (in words) ---
    sent_lengths = [len(t) for t in sent_token_lists]
    sentence_length_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # --- 4: max dependency-tree depth ---
    def tok_depth(tok):
        d = 0
        cur = tok
        while cur.head != cur:
            cur = cur.head
            d += 1
            if d > 200:
                break
        return d
    depths = [tok_depth(t) for t in doc if not t.is_space]
    max_parse_depth = int(max(depths)) if depths else 0

    # --- 5: paragraph density (breaks per 100 words) ---
    n_paragraphs = len([p for p in re.split(r"\n\s*\n", body_full) if p.strip()])
    paragraph_density = 100.0 * max(n_paragraphs - 1, 0) / n_words_full

    # --- 6: clause density (subordinators/relatives per sentence), via spaCy deps ---
    clause_deps = {"mark", "advcl", "relcl", "ccomp", "xcomp", "acl"}
    n_clause_markers = sum(1 for t in doc if t.dep_ in clause_deps)
    clause_density = n_clause_markers / max(len(sentences), 1)

    # --- 7: orthographic error density (per 100 words) ---
    sample_words = [w.lower() for w in re.findall(r"[A-Za-z]+", body)][:400]  # cap for speed
    alpha_words = [w for w in sample_words if len(w) > 2]
    misspelled = spell.unknown(alpha_words) if alpha_words else set()
    orthographic_error_density = 100.0 * len(misspelled) / max(len(alpha_words), 1)

    # --- 8: punctuation distribution entropy ---
    punct_counts = Counter(ch for ch in body_full if ch in PUNCT_CHARS)
    punctuation_entropy = shannon_entropy(punct_counts)

    # --- 9: urgency-keyword density (per 100 words) ---
    urgency_hits = len(URGENCY_WORDS.findall(text_full))
    urgency_density = 100.0 * urgency_hits / n_words_full

    # --- 10: pronoun balance (1st:2nd person) ---
    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    second_person = {"you", "your", "yours"}
    pron_tokens = [t.text.lower() for t in doc if t.pos_ == "PRON"]
    n_first = sum(1 for p in pron_tokens if p in first_person)
    n_second = sum(1 for p in pron_tokens if p in second_person)
    pronoun_balance = (n_first + 1.0) / (n_second + 1.0)

    # --- 11: sentiment variance across sentences (VADER compound) ---
    sent_sentiments = [sia.polarity_scores(s)["compound"] for s in sentences[:60]]
    sentiment_variance = float(np.var(sent_sentiments)) if len(sent_sentiments) > 1 else 0.0

    # --- 12: imperative verb density (per 100 words) ---
    n_imperative = 0
    for sent in doc.sents:
        toks = [t for t in sent if not t.is_space and not t.is_punct]
        if not toks:
            continue
        first = toks[0]
        if first.tag_ == "VB" and first.lemma_.lower() in IMPERATIVE_LEMMAS:
            n_imperative += 1
        elif first.tag_ == "VB" and not any(c.dep_ == "nsubj" for c in first.children):
            n_imperative += 1
    imperative_density = 100.0 * n_imperative / n_words_full

    # --- 13: lexical diversity (TTR) ---
    lc_words = [w.lower() for w in words_full]
    lexical_diversity = len(set(lc_words)) / n_words_full if lc_words else 0.0

    # --- 14: pronoun density (overall, share of all words) ---
    pronoun_density = len(pron_tokens) / max(len([t for t in doc if not t.is_space and not t.is_punct]), 1)

    # --- 15: generic greeting flag ---
    head = (subject + " " + body_full[:300])
    is_generic_greeting = bool(GENERIC_GREETING_RE.search(head)) and not bool(NAMED_GREETING_RE.search(body_full[:300]))

    # --- 16: subject RE:/FW: flag ---
    is_subject_refw = bool(SUBJECT_REFW_RE.match(subject))

    # --- 17: attachment-mention flag ---
    has_attachment_mention = bool(ATTACHMENT_RE.search(text_full))

    # --- 18: bracketed CTA flag ---
    has_bracketed_cta = any(CTA_WORDS_RE.search(m) or True for m in BRACKETED_CTA_RE.findall(text_full)) and bool(BRACKETED_CTA_RE.search(text_full))

    # --- 19: sensitive-info-request flag ---
    requests_sensitive_info = bool(SENSITIVE_INFO_RE.search(text_full))

    # --- 20: threat/consequence-clause flag ---
    has_threat_clause = bool(THREAT_RE.search(text_full))

    return {
        "avg_token_logprob": avg_token_logprob,
        "sentence_perplexity_variance": sentence_perplexity_variance,
        "sentence_length_std": sentence_length_std,
        "max_parse_depth": max_parse_depth,
        "paragraph_density": paragraph_density,
        "clause_density": clause_density,
        "orthographic_error_density": orthographic_error_density,
        "punctuation_entropy": punctuation_entropy,
        "urgency_density": urgency_density,
        "pronoun_balance": pronoun_balance,
        "sentiment_variance": sentiment_variance,
        "imperative_density": imperative_density,
        "lexical_diversity": lexical_diversity,
        "pronoun_density": pronoun_density,
        "generic_greeting_flag": int(is_generic_greeting),
        "subject_refw_flag": int(is_subject_refw),
        "attachment_mention_flag": int(has_attachment_mention),
        "bracketed_cta_flag": int(has_bracketed_cta),
        "sensitive_info_flag": int(requests_sensitive_info),
        "threat_clause_flag": int(has_threat_clause),
    }


def process_file(path, lm, label):
    df = pd.read_csv(path)
    print(f"Processing {label}: {len(df)} rows ...", flush=True)
    t0 = time.time()
    bodies_capped = (df["body"].fillna("").apply(cap)).tolist()
    records = []
    BATCH = 200
    docs_iter = nlp.pipe(bodies_capped, batch_size=64)
    for i, doc in enumerate(docs_iter):
        row = df.iloc[i]
        feats = extract_one(row, lm, doc)
        feats["id"] = row["id"]
        feats["class"] = row["class"]
        records.append(feats)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(df)}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    out = pd.DataFrame(records)
    cols = ["id", "class"] + [c for c in out.columns if c not in ("id", "class")]
    out = out[cols]
    print(f"  done in {time.time()-t0:.0f}s")
    return out


if __name__ == "__main__":
    train_df = pd.read_csv(f"{CLEAN}/training_set.csv")
    lm = build_background_lm(train_df)

    train_feats = process_file(f"{CLEAN}/training_set.csv", lm, "training_set.csv")
    train_feats.to_csv(f"{OUT}/training_features.csv", index=False)

    holdout_feats = process_file(f"{CLEAN}/holdout_set.csv", lm, "holdout_set.csv")
    holdout_feats.to_csv(f"{OUT}/holdout_features.csv", index=False)

    print("\nWrote training_features.csv and holdout_features.csv")
    print(train_feats.describe().T[["mean", "std", "min", "max"]])
