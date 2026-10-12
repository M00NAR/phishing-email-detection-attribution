# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, Copilot, etc.) working on
this repository. Human collaborators should start with `HANDOUT.md` in this folder.

## Project summary

This is research code for a phishing email paper with a two-step classifier:
1. Legitimate vs Phishing
2. Human-Phishing vs AI-Phishing

The model is a hand-built Naive Bayes:
- Rank-binned NB handles continuous features, and categorical NB handles flag features.
- A meta-NB combines the two.
- Features are chosen by a statistical gate.
- The accepted model is `05_final_model_and_evaluation/locked_model_final.json`, with
  feature set `v2_26_features_length_gated`.

## Repository map

| Path | Contents |
|---|---|
| `00_onboarding/` | This file and `HANDOUT.md` (onboarding docs) |
| `01_data_cleaning/` | Dataset building scripts + cleaned CSVs (`id, class, subject, body, ...`) |
| `02_feature_extraction/` | Feature scripts + feature tables |
| `03_statistical_tests/` | Feature-selection tests, `locked_features_*.json` |
| `04_naive_bayes_pipeline/` | Model development scripts `08_`–`13_` |
| `05_final_model_and_evaluation/` | Final model, evaluation scripts, predictions, summaries |
| `06_visualizations/` | Plots |
| `07_validation/` | Sanity/validation scripts |
| `08_proposal_and_writeup/` | LaTeX sources and PDFs |
| `README.md` | Per-file description |
| `CLAUDE.md` | Same conventions, for Claude Code |

The `class` column values are exactly `Legitimate`, `Human-Phishing` and `AI-Phishing`.

## Rules

1. **Keep the layout flat.** Do not create subfolders inside the numbered folders. New
   files go in the numbered folder for their pipeline stage.
2. **Never modify locked outputs:** `locked_model_final.json`, `locked_features_*.json`,
   `evaluation_summary_*.json`, `holdout_predictions_*.csv`. The paper reports these. Write
   new experiments to new files with a distinct suffix.
3. **Never train on holdout data.** Holdout CSVs are for evaluation only.
4. **Keep experiments separate:** `clair` experiment files are separate from `final`/`v2`
   files. Do not mix their inputs or outputs.
5. **Hard-coded paths:** scripts define `CLEAN`, `FEAT`, `FINAL`, `OUT`, `UP` as absolute
   sandbox paths. If you need to run a script, change these to repo-relative paths, e.g.
   `ROOT = Path(__file__).resolve().parent.parent`. Don't add new absolute paths.
6. **Missing inputs:** raw corpora (`UP`) and some intermediates (`training_features.csv`,
   `rank_binned_params.json`, `*_raw.csv`) are not committed. Say so instead of inventing
   data. Regenerate intermediates by running the earlier steps.
7. **File size:** keep every file under 100 MB. Don't commit zips or LaTeX build artifacts;
   `.gitignore` already covers them.
8. **Reporting numbers:** take metrics from the `evaluation_summary_*.json` files. Never
   quote numbers from memory or from older drafts. `abstract.tex` and the flowchart PDFs
   describe an outdated 14-feature design.

## Environment

- Python 3 with pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, nltk, spacy
  (`en_core_web_sm`), vaderSentiment and pyspellchecker.
- There is no test suite or build step yet. To check a change, re-run the affected script
  and compare its output with the existing CSV/JSON.

## Git workflow

- Work on a feature branch and open a pull request against `main`. Don't push directly to
  `main` unless the maintainers ask.
- Write clear commit messages that say which pipeline stage changed.
