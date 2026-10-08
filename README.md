# ML_Group_41 · Customer Repurchase Prediction

Group project for **Machine Learning, NOVA IMS (Lisbon), 2026/27** — Group 41, Kaggle team `ML_Group_41`.

**Task.** For the wholesaler *Everything & Then Some, Ltd.*, predict `repeat_purchase_90d`: whether a customer
buys again within 90 days of a purchase occasion (binary classification).
**Metric.** F1-score on the positive class, so the decision threshold is tuned, not left at 0.5.
**Data.** `train.csv` (6,534 purchase occasions + target), `test.csv` (986 *later* occasions, no target),
optional `clientes.csv` (1,143 customers, join on `customer_id`), `sample_submission.csv`.
Customers repeat across rows, chronology matters, some columns have Portuguese names, and `random_noise`
is a deliberate decoy. Source: [RicardoSantos0/ML_DSAA_Practicals](https://github.com/RicardoSantos0/ML_DSAA_Practicals) (`project/`).

Full instructions are in [`docs/brief/`](docs/brief/).

## Deadlines

| Deliverable | What | Deadline |
|---|---|---|
| Homework | `Homework_Group41` notebook + 2-page PDF | **26 Oct 2026, 17:59** |
| Final project | One clean notebook (zip with `src/` and data allowed) | **22 Dec 2026, 17:59** — target delivery **15 Dec** (early bonus 0.15 pts/day, max 1 pt) |

## Team

| Name | GitHub | Main area |
|---|---|---|
| _Member 1_ | `@username1` | _e.g. EDA + cleaning_ |
| _Member 2_ | `@username2` | _e.g. validation + preprocessing_ |
| _Member 3_ | `@username3` | _e.g. features + models_ |

## Quick start

```bash
git clone git@github.com:<owner>/ML_Group_41.git
cd ML_Group_41
conda env create -f environment.yml
conda activate machine_learning
```

Then:

1. Open the `ML_Group_41` folder in **VS Code** (File → Open Folder).
2. Install the **Python** and **Jupyter** extensions (Microsoft).
3. Open any notebook, click *Select Kernel* (top right) and choose **`machine_learning`**.
4. Run the first cell: it should print the project root and `Raw data folder exists: True`.

The raw data is committed in `data/raw/`, so notebooks run right after cloning.

## Daily git workflow

Start of a session:

```bash
git pull
```

Work, then save your progress (clear notebook outputs first):

```bash
git add notebooks/lab/01_eda.ipynb src/cleaning.py docs/decisions.md
git commit -m "EDA: missing values and target balance"
git push
```

Changes to the **final notebook** always go through a branch + Pull Request:

```bash
git switch main
git pull
git switch -c final/section-iii-models
# ... edit notebooks/ML_Group_41_final.ipynb ...
git add notebooks/ML_Group_41_final.ipynb
git commit -m "Final notebook: section III model comparison"
git push -u origin final/section-iii-models
gh pr create --fill
```

After the PR is merged: `git switch main && git pull`.

## Repository map

```
ML_Group_41/
├── README.md
├── environment.yml          # course conda env (machine_learning), unchanged
├── data/
│   ├── raw/                 # original Kaggle CSVs, never modified
│   └── processed/           # generated files (gitignored)
├── docs/
│   ├── brief/               # course PDFs (project + homework)
│   └── decisions.md         # decision log
├── notebooks/
│   ├── lab/                 # personal working notebooks, one owner each
│   │   ├── 01_eda.ipynb
│   │   ├── 02_cleaning.ipynb
│   │   ├── 03_validation.ipynb
│   │   ├── 04_preprocessing.ipynb
│   │   ├── 05_features.ipynb
│   │   └── 06_models.ipynb
│   ├── homework/
│   │   └── Homework_Group41.ipynb
│   └── ML_Group_41_final.ipynb   # the graded notebook
├── src/                     # reusable functions imported by notebooks
│   ├── paths.py             # PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, SUBMISSIONS, REPORTS
│   ├── cleaning.py          # deterministic rules (train and test alike)
│   ├── preprocessing.py     # learned steps, inside sklearn Pipelines
│   ├── features.py          # feature engineering + selected feature lists
│   └── validation.py        # time-/customer-aware CV, F1 + threshold helpers
├── reports/                 # homework PDF, figures
└── submissions/             # ML_Group_41_VersionXX.csv
```

## Workflow rules

- **Lab notebooks are personal**: one owner each (see the notebook header), never edited by two people at once.
- **The final notebook is edited by one person at a time**, via a branch + Pull Request.
- **Reusable, stable code is moved (not copied)** from lab notebooks into `src/` and imported back.
- **`src/` holds the "how"** (functions); **notebooks hold the "what" and "why"** (decisions, results, plots).
- **Always `git pull` before starting**, and commit + push at the end of every session.
- **Clear all notebook outputs before committing lab notebooks** (VS Code: *Clear All Outputs*).
- **Never fit anything on `test.csv`.** Preprocessing that learns from data (imputation, encoding, scaling,
  learned outlier thresholds) lives inside sklearn Pipelines, fitted on training folds only.
  Deterministic rule-based cleaning may be applied to train and test alike. Rows are never dropped from test.
- **Validation respects time order and repeated customers** (time-aware and/or group-aware splits).
- **Models**: vanilla scikit-learn (or TensorFlow / Keras / PyTorch) only. No XGBoost / LightGBM / CatBoost, no AutoML.
- **Every non-trivial decision gets a row in [`docs/decisions.md`](docs/decisions.md).**
- **Submission naming**: `submissions/ML_Group_41_VersionXX.csv` (two-digit version: `Version01`, `Version02`, ...),
  columns `ID,repeat_purchase_90d`, 986 rows, no index.
- **The repo stays private until the final deadline**, then it is made public (GitHub bonus point).
- Experiments that do not end up in the final notebook are submitted through the course form, so keep lab notebooks tidy.
