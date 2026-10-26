# Phase 2 current audit

The seven archives and Phase 1 arrays are present locally. Full-data Logistic Regression, Random Forest, XGBoost, and ANN runs are saved and independently reloaded on held-out rows.
XGBoost uses bounded CUDA histogram training and validation early stopping. The ANN uses CPU because native Windows TensorFlow 2.21 has no CUDA GPU support.
A separate validation-only XGBoost depth and ANN width sensitivity study is in `validation_sensitivity.json`; it does not alter held-out scores.
17 verified Phase 2 configurations are in `classical_model_metrics.csv`.
