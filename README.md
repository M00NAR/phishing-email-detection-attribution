Phishing Detection & Attribution Paper — Project Files
Mannan Bhardwaj (25BYB0013) & Bhoumik Singh Rajput (25BYB0120)

Folders are in pipeline order:

01_data_cleaning/
  Scripts that build the training/holdout datasets (Nazario + Enron + AI-generated +
  CLAIR advance-fee-fraud corpus). Includes the cleaned CSVs themselves.
  - clean_and_merge.py        -> builds the ORIGINAL training_set.csv / holdout_set.csv
  - add_fraud_corpus.py       -> cleans CLAIR corpus, adds it to holdout ONLY
  - rebuild_clair_only_training.py -> builds the CLAIR-only-trained experiment's datasets
  - parse_fraud_corpus.py / redact_and_verify.py -> supporting cleaning utilities

02_feature_extraction/
  Scripts that compute the 26 linguistic/statistical features per email.
  - extract_features.py        -> original 20 features (spaCy, trigram LM, VADER, spellcheck)
  - extract_new_features.py    -> the 6 new regex-based features
  - extract_features_for_new_holdout_rows.py -> features for the added CLAIR holdout rows
  - recompute_lm_features_clair.py -> LM-dependent features for the CLAIR-only experiment
  (CSV outputs: training/holdout feature tables for each experiment variant)

03_statistical_tests/
  The hypothesis-testing feature-selection pipeline.
  - 01-07: Kruskal-Wallis, Mann-Whitney, Chi-square (3-group & 2-group), Cohen's d,
    Cramer's V, FDR (Benjamini-Hochberg) correction + dual gate
  - rerun_full_gate_with_new_features.py / rerun_gate_clair.py -> re-run gate on 26 features
  - gate_results_*.csv, locked_features_*.json -> which features passed, per experiment
  - kruskal_wallis_full_results.csv -> full tidy rank-sum/expected/deviation/H-stat table

04_naive_bayes_pipeline/
  The two-stage Naive Bayes classifier itself.
  - 08: Rank-Binned Naive Bayes (continuous features)
  - 09: Categorical Naive Bayes (categorical/flag features)
  - 10: Meta-Naive Bayes combination (renormalized posterior)
  - 11: Two-step decision logic (legit-vs-phishing, then human-vs-AI) + evaluation
  - 12: Length-robustness gate (skip variance features under 30 words)
  - 13: Full A/B/C rebuild-and-compare across feature-set versions

05_final_model_and_evaluation/
  The ACCEPTED production model and its evaluations.
  - finalize_model.py       -> builds locked_model_final.json (26 features + permanent length gate)
  - evaluate_final.py       -> independent reproducibility check on original holdout
  - evaluate_final_augmented.py -> same model, re-evaluated on enlarged (CLAIR-added) holdout
  - evaluate_clair.py       -> the CLAIR-only-trained model (generalization study, Test 2)
  - locked_model_final.json -> the actual locked model parameters
  - evaluation_summary_*.json, holdout_predictions_*.csv -> results per experiment

06_visualizations/
  - plot_kruskal_rank_sums.py + kruskal_rank_sum_deviation.png/.pdf -> rank-sum deviation chart
  - charts.py / eda.py + chart_*.png -> early exploratory data analysis charts

07_validation/
  - check_collisions.py -> cross-corpus/id collision checks
  - step1_features_stats.py / step2_baseline_clf.py / step3_redact_retest.py -> early
    validation scripts (feature sanity checks, baseline classifier, redaction test)

08_proposal_and_writeup/
  - proposal_v3.tex/.pdf  -> CURRENT authoritative proposal (26 features, length gate,
    two-pipeline architecture, generalization robustness study)
  - proposal.txt          -> an earlier plain-text draft (superseded)
  - abstract.tex/.pdf     -> shorter congress abstract (NOT yet updated to match proposal_v3)
  - feature_list.tex/.pdf, project_workflow.tex/.pdf, training_results.tex/.pdf,
    workflow.tex/.pdf, test_reference.tex/.pdf -> supporting reference documents
  - flowchart.pdf, flowchart2.pdf, figure1_pipeline.pdf -> pipeline diagrams (NOTE: these
    still show the older 14-feature design; proposal_v3's figure caption flags this)
  - training_algorithm.docx -> Naive Bayes training algorithm write-up

Notes:
- "v2" in filenames = the 26-feature + length-gate feature set (the accepted one).
- "clair" in filenames = the CLAIR advance-fee-fraud corpus generalization experiment
  (a SEPARATE, parallel experiment — NOT the production model).
- "final" in filenames = the accepted production model.
