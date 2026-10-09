import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

CLEAN = "/tmp/claude-0/-home-claude/68f35678-ae4c-5abf-a779-d10b92f77209/scratchpad/clean"
t = pd.read_csv(f"{CLEAN}/training_set_with_quickfeats.csv")

X_text = t["subject"].fillna("") + " " + t["body"].fillna("")
y = t["class"]

vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2, stop_words="english")
X = vec.fit_transform(X_text)

clf = LogisticRegression(max_iter=1000, C=1.0)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
preds = cross_val_predict(clf, X, y, cv=skf, method="predict")

print("=== 5-fold CV, TF-IDF(1-2gram) + Logistic Regression, full training_set.csv ===")
print(classification_report(y, preds, digits=3))
print("Confusion matrix (rows=true, cols=pred), classes:", sorted(y.unique()))
print(confusion_matrix(y, preds, labels=sorted(y.unique())))

# fit on full data to inspect top tokens per class (what is it actually keying on?)
clf.fit(X, y)
feat_names = np.array(vec.get_feature_names_out())
classes = clf.classes_
print()
print("=== Top 15 tokens most predictive of each class (logistic regression coefficients) ===")
for i, c in enumerate(classes):
    coefs = clf.coef_[i]
    top_idx = np.argsort(coefs)[-15:][::-1]
    print(f"\n{c}:")
    for idx in top_idx:
        print(f"   {feat_names[idx]:30s} {coefs[idx]:.3f}")
