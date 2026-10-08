"""Feature engineering and feature selection helpers.

This module will hold the features designed in notebooks/lab/05_features.ipynb
(e.g. ratios between monetary columns, recency/frequency combinations, optional
joins with clientes.csv) and the selected feature lists.

Identifier-like columns (ID, customer_id, document_series) and the decoy column
random_noise must be justified explicitly before being used as predictors.
Any selection step that learns from the target must run inside the training folds.
"""
