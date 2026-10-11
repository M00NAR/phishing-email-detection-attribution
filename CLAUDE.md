# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Research code, data and write-up for a phishing email **detection and attribution** paper
(Mannan Bhardwaj & Bhoumik Singh Rajput). Emails are classified in two steps:

1. **Detection** — Legitimate vs Phishing
2. **Attribution** — Human-Phishing vs AI-Phishing (only for emails flagged as phishing)

The classifier is a hand-built two-stage Naive Bayes: a rank-binned NB over continuous
features + a categorical NB over flag features, combined by a meta-NB (renormalized posterior).
Features are selected by a statistical gate (Kruskal-Wallis, Mann-Whitney, chi-square,
Cohen's d, Cramer's V, Benjamini-Hochberg FDR).

Data sources: Nazario phishing corpus, Enron (legitimate), AI-generated phishing, and the
CLAIR advance-fee-fraud corpus.

## Layout

Folders are numbered in pipeline order; each holds its scripts **and** the data/outputs they
produce. There are no nested subfolders — keep it that way.

| Folder | Purpose |
|---|---|
| `01_data_cleaning/` | Build training/holdout CSVs from the raw corpora |
| `02_feature_extraction/` | Compute the 26 linguistic/statistical features per email |
| `03_statistical_tests/` | Hypothesis-test feature selection gate → `locked_features_*.json` |
| `04_naive_bayes_pipeline/` | Development of the NB models (scripts 08–13) |
| `05_final_model_and_evaluation/` | Accepted model (`locked_model_final.json`) + evaluations |
| `06_visualizations/` | Rank-sum deviation chart and early EDA charts |
| `07_validation/` | Collision checks, baseline classifier, redaction retest |
| `08_proposal_and_writeup/` | LaTeX proposal/abstract, figures, PDFs, training algorithm doc |

See `README.md` for a per-file description.

## Naming conventions

- `v2` — the accepted 26-feature set with the length gate (variance features skipped for
  emails under 30 words).
- `final` — the accepted production model and its results.
- `clair` — a **separate** generalization experiment (model trained on CLAIR only). Never mix
  its files with the `final`/`v2` results.
- `augmented` — the original holdout enlarged with CLAIR rows (training set unchanged).

## Important caveats

- **Hard-coded paths:** scripts set directory constants such as
  `CLEAN`, `FEAT`, `FINAL`, `OUT`, `UP` to absolute paths from the original sandbox
  (`/tmp/claude-0/.../scratchpad/...`, `/root/.claude/uploads/...`). They will not run until
  these are pointed at the repo folders (e.g. paths relative to `Path(__file__).parent.parent`).
- **Raw corpora not included:** scripts that read from `UP` (raw Nazario/Enron/LLM uploads)
  need the original source CSVs, which are not in the repo. The cleaned CSVs in
  `01_data_cleaning/` are the starting point for everything downstream.
- **Some intermediate files are missing:** a few scripts reference outputs not committed
  (e.g. `training_features.csv`, `rank_binned_params.json`, `*_raw.csv`). Regenerate them by
  running the earlier pipeline steps rather than assuming they exist.
- **Write-up status:** `proposal_v3.tex` is the authoritative proposal. `abstract.tex` and the
  flowchart/figure PDFs still describe the older 14-feature design.
- Do not edit the locked results (`locked_model_final.json`, `locked_features_*.json`,
  `evaluation_summary_*.json`) by hand — they are outputs reported in the paper.

## Dependencies

Python 3 with: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, nltk, spacy,
vaderSentiment, pyspellchecker.

## Git

- Large CSVs (up to ~27 MB) are committed directly; keep every file under GitHub's 100 MB limit.
- Zips and LaTeX build artifacts are ignored via `.gitignore`.
