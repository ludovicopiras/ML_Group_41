# Predicting 90-Day Repeat Purchases

Machine Learning 2026/27 · NOVA IMS · Group 41

Binary classification of `repeat_purchase_90d` (whether a customer buys again within 90 days of a purchase)
for the wholesaler *Everything & Then Some, Ltd.*, evaluated with F1-score.
Data: [RicardoSantos0/ML_DSAA_Practicals](https://github.com/RicardoSantos0/ML_DSAA_Practicals) (`project/data/`).

## Structure

```
data/raw/          original competition data
notebooks/         final notebook, homework, lab notebooks
src/               helper functions used by the notebooks
submissions/       Kaggle submission files
docs/              decision log, Kaggle submissions log
environment.yml    conda environment (machine_learning)
```

## How to run

Create the environment with `conda env create -f environment.yml`, then run
`notebooks/ML_Group_41_final.ipynb` with the `machine_learning` kernel.
