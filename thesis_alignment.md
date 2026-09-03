# Thesis-to-code alignment

| Thesis element | Repository implementation |
|---|---|
| Binary classes: benign/malicious | `label=0/1` validation in `train.py` |
| Data cleaning | `normalize_sql()` in `features.py` |
| Query length / keyword frequency / suspicious operators / functions | `handcrafted_features()` |
| Entropy | `shannon_entropy()` |
| TF-IDF | `TfidfVectorizer` |
| One-hot encoding | `OneHotEncoder` for query-shape metadata |
| Normalization/scaling | `StandardScaler` for engineered numeric features |
| 80/20 stratified split | `train_test_split(... test_size=0.20, stratify=y)` |
| Class balancing | SMOTE on training data only |
| Naive Bayes | `MultinomialNB` |
| Decision Tree | `DecisionTreeClassifier` |
| Random Forest | `RandomForestClassifier` |
| SVM | RBF `SVC` |
| ANN | Keras dense network with batch normalization, dropout and early stopping |
| Hybrid ANN-SVM | ANN penultimate layer → RBF SVM |
| 10-fold cross-validation | `StratifiedKFold(n_splits=10)` for classical models |
| Accuracy / precision / recall / F1 | `metrics.py` |
| False positive rate | computed from confusion matrix |
| Confusion matrix | PNG per model |
| ROC-AUC / ROC curve | `metrics.py` |
| Computational efficiency | training and inference timing |
| Runtime monitoring concept | `predict.py` provides non-executing online query classification |

## Why the code may not produce exactly the thesis' reported numbers

The thesis document reports test support of 614 benign and 226 malicious examples for several result tables, but the actual query corpus is not embedded in the thesis document. Exact numerical reproduction therefore requires the original dataset, the exact cleaning rules, random state(s), and the exact hyperparameters used in the original experiment. This repository implements the documented methodology without fabricating those missing inputs.
