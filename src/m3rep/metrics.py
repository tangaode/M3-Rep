"""Ranking metrics used by M3-Rep."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _rows(scores, truth, exclude):
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    exclude = np.asarray(exclude, dtype=bool)
    if scores.shape != truth.shape or scores.shape != exclude.shape:
        raise ValueError("scores, truth and exclude must have the same shape")
    for q in range(scores.shape[1]):
        available = ~exclude[:, q] & np.isfinite(scores[:, q])
        y = truth[available, q]
        s = scores[available, q]
        if y.sum() == 0:
            continue
        order = np.argsort(-s, kind="stable")
        yield q, y, s, order


def evaluate(scores, truth, exclude, ks=(1, 5, 10, 20, 50, 100)) -> pd.DataFrame:
    records = []
    for q, y, s, order in _rows(scores, truth, exclude):
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(1, len(order) + 1)
        ap = float(average_precision_score(y, s))
        auroc = float(roc_auc_score(y, s)) if y.any() and (~y).any() else np.nan
        row = {"disease_index": q, "positives": int(y.sum()),
               "candidates": int(len(y)), "AP": ap, "AUROC": auroc,
               "MRR": float(1.0 / rank[np.flatnonzero(y)].min())}
        for k in ks:
            top = order[:min(k, len(order))]
            row[f"Recall@{k}"] = float(y[top].sum() / y.sum())
            gains = (2.0 ** y[order[:min(k, len(order))]] - 1.0)
            discount = np.log2(np.arange(2, len(gains) + 2))
            dcg = float((gains / discount).sum())
            ideal_len = min(k, int(y.sum()))
            ideal = float((np.ones(ideal_len) / np.log2(np.arange(2, ideal_len + 2))).sum())
            row[f"NDCG@{k}"] = dcg / ideal if ideal else 0.0
        records.append(row)
    return pd.DataFrame(records)


def macro_summary(per_disease: pd.DataFrame) -> pd.Series:
    if per_disease.empty:
        raise ValueError("No disease has a positive test label")
    numeric = [c for c in per_disease.columns if c not in {"disease_index", "positives", "candidates"}]
    return per_disease[numeric].mean(numeric_only=True)
