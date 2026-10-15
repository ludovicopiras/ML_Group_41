# Kaggle submissions log

This table is written by `src/submissions.py`: do not edit it by hand.

**How to submit** (at the end of your notebook, after fitting the final model):

```python
from src.submissions import submit, publish, record_kaggle_score

submission = pd.DataFrame({"ID": test.index, "repeat_purchase_90d": test_pred})  # 0/1 labels, threshold applied
version = submit(submission, notebook="my_notebook", model="short model description",
                 cv_f1=honest_cv_f1, notes="threshold, features, anything worth knowing")
publish(version)                        # commit + push: the version number is now taken
```

Then upload `submissions/ML_Group_41_VersionXX.csv` on Kaggle and record the public score:

```python
record_kaggle_score(version, 0.77033)
publish(version)
```

- **Version numbers, date and author are automatic.** Re-running a notebook with the same predictions reuses its version.
- `submit` stops if GitHub has commits you have not pulled: run `git pull`, then run the cell again.
- `cv_f1` is the honest validation estimate (time CV + 90-day gap). Kaggle should land close to it.

| Version | Date | Author | Notebook / model | CV F1 | Kaggle public F1 | Notes |
|---|---|---|---|---|---|---|
| 01 | 2026-10-11 | Ludovico Piras | submission_baseline_ludovico · logistic regression (median impute + scaling + one-hot) | 0.771 | 0.77033 | threshold 0.34 from time CV + gap; CV ROC AUC 0.778; 80% predicted 1; no feature work |
| 02 | 2026-10-15 | Ludovico Piras | submission_v02_ludovico · logistic regression + linear recipe (median/mode, log1p, robust scaler, one-hot min 50) | 0.781 | 0.77511 | threshold 0.38 from time CV + gap; CV ROC AUC 0.783; 77% predicted 1; same model as V01, only preprocessing changed |
| 03 | 2026-10-15 | Ludovico Piras | submission_v03_v04_ludovico · logistic regression + linear recipe + month + L1 selection (C=0.05) | 0.802 | 0.80397 | threshold 0.43 from time CV + gap; CV ROC AUC 0.796; 74% predicted 1 |
| 04 | 2026-10-15 | Ludovico Piras | submission_v03_v04_ludovico · gradient boosting (default) + tree recipe + month, no selection | 0.777 | 0.79611 | threshold 0.35 from time CV + gap; CV ROC AUC 0.774; 78% predicted 1 |
