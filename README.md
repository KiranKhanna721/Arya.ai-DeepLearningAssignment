# Arya.ai Deep Learning Assignment

A binary classification assignment: train a model on an anonymised tabular dataset (features `X1` to `X57`, target `Y`) and predict labels for a held-out test set.

## Contents

| File | Purpose |
|------|---------|
| `EDA.ipynb` | Exploratory data analysis: distributions, correlations, feature selection |
| `Full_Assignment.ipynb` | Full modelling workflow and comparison of approaches |
| `Performance_Of_model.py` | Final script: drops low-value features, tunes a Random Forest with `GridSearchCV` (10-fold CV), reports accuracy, F1 and the confusion matrix, then writes the test predictions |
| `training_set.csv` / `test_set.csv` | Input data |
| `test_output.csv` | Model predictions for the test set |

## Getting started

```bash
pip install -r requirements.txt
python Performance_Of_model.py
```
