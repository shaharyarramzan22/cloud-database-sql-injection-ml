# cloud-database-sql-injection-ml
# ML-Based Cloud Database SQL Injection Detection

Implementation companion for the MS thesis **“A Modern Machine Learning-Based Approach to Cloud Computing Database Security: Enhanced Techniques for Analyzing Challenges and Opportunities.”**

The project implements the experimental pipeline described in the thesis:

- SQL-query cleaning and normalization
- lexical/structural feature engineering: query length, keyword frequencies, suspicious operators/functions, entropy
- TF-IDF text representation
- one-hot encoding for categorical SQL metadata
- stratified 80/20 train-test split
- optional SMOTE balancing on the training set only
- 10-fold stratified cross-validation
- Naive Bayes, Decision Tree, Random Forest, SVM, ANN
- hybrid ANN→SVM model, where the ANN penultimate layer is used as the feature generator
- accuracy, precision, recall, F1, false-positive rate, confusion matrices, ROC-AUC
- training and inference timing
- single-query prediction CLI
- saved preprocessing/model artifacts for reproducible inference

## Important reproducibility note

The thesis document describes the dataset construction and reports results, but the actual underlying SQL-query dataset is not included with the document. Therefore this repository **does not fabricate the thesis dataset or silently claim exact reproduction of the reported numbers**. Put your original CSV in `data/sql_queries.csv` and run the pipeline. The expected CSV schema is:

```csv
query,label
"SELECT * FROM users WHERE id = 10",0
"SELECT * FROM users WHERE id = 10 OR 1=1",1
```

`label=0` means benign and `label=1` means malicious.

## Environment

Recommended: Python 3.10–3.12.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the complete experiment

```bash
python -m src.train --data data/sql_queries.csv --output results --models models
```

If your CSV uses different column names:

```bash
python -m src.train --data data/my_dataset.csv --query-column sql --label-column class
```

## Predict a new query

After training:

```bash
python -m src.predict --models models --query "SELECT * FROM users WHERE id = 10 OR 1=1"
```

The CLI returns predictions from all trained models and the hybrid ANN-SVM model.

## Project structure

```text
.
├── data/
│   └── sql_queries.csv          # add the thesis dataset here; not included
├── docs/
│   └── thesis_alignment.md
├── models/                      # generated .joblib/.keras artifacts
├── results/                     # generated CSV/PNG/JSON reports
├── src/
│   ├── __init__.py
│   ├── features.py              # SQL feature engineering + TF-IDF
│   ├── models.py                # six classifiers + hybrid ANN-SVM
│   ├── metrics.py               # metrics and plots
│   ├── train.py                 # complete experiment
│   └── predict.py               # online/single-query inference
├── requirements.txt
└── README.md
```

## Research-design alignment

The thesis uses six models—Naive Bayes, Decision Tree, Random Forest, SVM, ANN, and hybrid ANN-SVM—and describes a modular input → preprocessing → classification → output/monitoring architecture. It also specifies feature engineering with keyword frequency, entropy, TF-IDF and one-hot encoding, an 80/20 stratified split, 10-fold cross-validation, and security-focused metrics. This repository implements those elements as executable modules.

### Security note

This is a defensive research implementation. It classifies SQL text; it does **not** execute supplied SQL against a live database. For production prevention, use parameterized queries/prepared statements and least-privilege database access in addition to ML detection.
