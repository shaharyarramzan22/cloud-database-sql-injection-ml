from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SQL_KEYWORDS = [
    "select", "insert", "update", "delete", "drop", "alter", "create",
    "union", "from", "where", "join", "having", "group", "order", "limit",
    "and", "or", "not", "into", "values", "set", "exec", "execute",
    "declare", "cast", "convert", "sleep", "benchmark", "waitfor",
]
SUSPICIOUS_TOKENS = [
    "--", "#", "/*", "*/", ";", "'", '"', "=", "<>", "!=", "||", "&&",
    "@@", "@", "xp_", "information_schema", "concat", "char(", "ascii(",
]
FUNCTIONS = ["sleep", "benchmark", "waitfor", "load_file", "outfile", "concat", "substr", "substring", "ascii", "char"]


def normalize_sql(sql: object) -> str:
    text = "" if sql is None or (isinstance(sql, float) and math.isnan(sql)) else str(sql)
    text = text.replace("\x00", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def keyword_count(text: str, keyword: str) -> int:
    return len(re.findall(rf"\b{re.escape(keyword)}\b", text, flags=re.I))


def handcrafted_features(sql: object) -> dict[str, float]:
    text = normalize_sql(sql)
    features: dict[str, float] = {
        "query_length": len(text),
        "token_count": len(re.findall(r"\b\w+\b", text)),
        "entropy": shannon_entropy(text),
        "digit_count": sum(ch.isdigit() for ch in text),
        "alpha_count": sum(ch.isalpha() for ch in text),
        "special_char_count": sum(not ch.isalnum() and not ch.isspace() for ch in text),
        "quote_count": text.count("'") + text.count('"'),
        "semicolon_count": text.count(";"),
        "comment_marker_count": text.count("--") + text.count("#") + text.count("/*"),
        "equals_count": text.count("="),
    }
    for kw in SQL_KEYWORDS:
        features[f"kw_{kw}"] = keyword_count(text, kw)
    for token in SUSPICIOUS_TOKENS:
        features[f"tok_{re.sub(r'[^a-z0-9]+', '_', token).strip('_') or 'special'}"] = text.count(token)
    for fn in FUNCTIONS:
        features[f"fn_{fn}"] = len(re.findall(rf"\b{re.escape(fn)}\s*\(", text, flags=re.I))
    return features


@dataclass
class SQLFeatureTransformer:
    max_tfidf_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self):
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[\w$@#-]+\b",
            ngram_range=self.ngram_range,
            min_df=1,
            max_features=self.max_tfidf_features,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numeric_columns: list[str] = []
        self.fitted = False

    def _handcrafted_frame(self, queries: Iterable[object]) -> pd.DataFrame:
        rows = [handcrafted_features(q) for q in queries]
        return pd.DataFrame(rows).fillna(0.0)

    def fit(self, queries: Iterable[object]):
        queries = [normalize_sql(q) for q in queries]
        self.tfidf.fit(queries)
        frame = self._handcrafted_frame(queries)
        self.numeric_columns = list(frame.columns)
        self.scaler.fit(frame[self.numeric_columns])
        # A small categorical representation of query shape/type supports the thesis' one-hot stage.
        categories = self._categorical_frame(queries)
        self.encoder.fit(categories)
        self.fitted = True
        return self

    def _categorical_frame(self, queries: Iterable[str]) -> pd.DataFrame:
        values = []
        for q in queries:
            first = q.split(" ", 1)[0] if q else "empty"
            values.append({
                "first_keyword": first,
                "has_where": "where" in q,
                "has_join": "join" in q,
                "has_comment": ("--" in q or "#" in q or "/*" in q),
            })
        return pd.DataFrame(values)

    def transform(self, queries: Iterable[object]):
        if not self.fitted:
            raise RuntimeError("Transformer must be fitted before transform().")
        queries = [normalize_sql(q) for q in queries]
        text = self.tfidf.transform(queries)
        frame = self._handcrafted_frame(queries)
        numeric = self.scaler.transform(frame[self.numeric_columns])
        categorical = self.encoder.transform(self._categorical_frame(queries))
        return hstack([text, csr_matrix(numeric), csr_matrix(categorical)], format="csr")

    def fit_transform(self, queries: Iterable[object]):
        self.fit(queries)
        return self.transform(queries)

    def transform_dense(self, queries: Iterable[object]) -> np.ndarray:
        return self.transform(queries).toarray().astype(np.float32)
