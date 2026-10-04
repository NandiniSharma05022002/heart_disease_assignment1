# EDA and Modeling Notebook Guide

Run the equivalent scripts from the repository root:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/eda.py
python -m src.heart_disease_mlops.train
```

The generated figures are saved under `reports/figures/`, and model comparison results are saved under `artifacts/model_comparison.csv`.

For an actual submitted notebook, use this as the execution outline and add your own interpretation of distributions, class balance, correlations, model selection and tuning results.
