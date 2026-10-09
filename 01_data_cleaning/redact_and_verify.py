import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')
RECIPIENT_RE = re.compile(r'\bmonkey\.org\b|\bjose\b', re.IGNORECASE)

def redact(s):
    s = EMAIL_RE.sub(' [EMAIL] ', str(s))
    s = RECIPIENT_RE.sub('[RECIPIENT]', s)
    return s

for fname in ["training_set.csv", "holdout_set.csv"]:
    df = pd.read_csv(f"{CLEAN}/{fname}")
    before_count = df["body"].astype(str).str.contains(r'monkey|jose', case=False, regex=True).sum()
    df["subject"] = df["subject"].apply(redact)
    df["body"] = df["body"].apply(redact)
    after_count = df["body"].astype(str).str.contains(r'monkey|jose', case=False, regex=True).sum()
    print(f"{fname}: rows mentioning monkey/jose  before={before_count}  after={after_count}")
    df.to_csv(f"{CLEAN}/{fname}", index=False)

# re-verify with the same baseline classifier used before, on the redacted training set
t = pd.read_csv(f"{CLEAN}/training_set.csv")
X_text = t["subject"].fillna("") + " " + t["body"].fillna("")
y = t["class"]
vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, stop_words="english")
X = vec.fit_transform(X_text)
clf = LogisticRegression(max_iter=1000, C=1.0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=skf, method="predict")
print()
print("=== Post-redaction 5-fold CV, TF-IDF + Logistic Regression ===")
print(classification_report(y, preds, digits=3))

clf.fit(X, y)
feat_names = np.array(vec.get_feature_names_out())
for i, c in enumerate(clf.classes_):
    coefs = clf.coef_[i]
    top_idx = np.argsort(coefs)[-10:][::-1]
    print(f"\nTop tokens for {c}:")
    for idx in top_idx:
        print(f"   {feat_names[idx]:25s} {coefs[idx]:.3f}")
