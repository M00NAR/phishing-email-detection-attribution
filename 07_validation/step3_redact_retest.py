import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
t = pd.read_csv(f"{CLEAN}/training_set_with_quickfeats.csv")

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+')

def redact(s):
    s = EMAIL_RE.sub(' [EMAIL] ', str(s))
    return s

X_text = (t["subject"].fillna("").apply(redact) + " " + t["body"].fillna("").apply(redact))
y = t["class"]

vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, stop_words="english")
X = vec.fit_transform(X_text)

clf = LogisticRegression(max_iter=1000, C=1.0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=skf, method="predict")

print("=== After redacting email addresses (removes the jose@monkey.org recipient leak) ===")
print(classification_report(y, preds, digits=3))

clf.fit(X, y)
feat_names = np.array(vec.get_feature_names_out())
for i, c in enumerate(clf.classes_):
    coefs = clf.coef_[i]
    top_idx = np.argsort(coefs)[-12:][::-1]
    print(f"\nTop tokens for {c}:")
    for idx in top_idx:
        print(f"   {feat_names[idx]:25s} {coefs[idx]:.3f}")
