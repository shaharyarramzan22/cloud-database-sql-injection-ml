from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ANNConfig:
    hidden1: int = 128
    hidden2: int = 64
    embedding: int = 32
    dropout: float = 0.25
    epochs: int = 30
    batch_size: int = 32


def build_ann(input_dim: int, cfg: ANNConfig | None = None) -> keras.Model:
    cfg = cfg or ANNConfig()
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(cfg.hidden1, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.dropout)(x)
    x = layers.Dense(cfg.hidden2, activation="relu", name="deep_features")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs, outputs, name="sql_ann")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def make_classical_models():
    return {
        "Naive_Bayes": Pipeline([
            ("scale", MaxAbsScaler()),
            ("model", MultinomialNB(alpha=0.1)),
        ]),
        "Decision_Tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, class_weight="balanced", n_jobs=-1, random_state=42),
        "SVM": Pipeline([
            ("scale", MaxAbsScaler()),
            ("model", SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        ]),
    }


def train_ann(X_train, y_train, X_val, y_val, cfg: ANNConfig | None = None):
    cfg = cfg or ANNConfig()
    model = build_ann(X_train.shape[1], cfg)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def ann_features(model: keras.Model, X):
    extractor = keras.Model(model.input, model.get_layer("deep_features").output)
    return extractor.predict(X, verbose=0)


def train_hybrid_ann_svm(ann_model, X_train, y_train, X_test):
    z_train = ann_features(ann_model, X_train)
    z_test = ann_features(ann_model, X_test)
    svm = SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)
    svm.fit(z_train, y_train)
    return svm, z_test
