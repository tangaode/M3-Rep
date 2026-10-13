"""M3-Rep training and inference."""
from __future__ import annotations

from pathlib import Path
import hashlib
import pickle
from typing import Mapping
import lightgbm as lgb
import numpy as np

from .features import FEATURE_NAMES, build_feature_planes, rows_from_planes


class M3RepRanker:
    """LightGBM scorer trained with disease-level pseudo-holdouts."""

    def __init__(self, seed: int = 42, n_estimators: int = 600):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.model: lgb.LGBMClassifier | None = None
        self.best_iteration: int | None = None
        self.feature_names = list(FEATURE_NAMES)

    @staticmethod
    def _folds(positive: np.ndarray, heldout: np.ndarray) -> np.ndarray:
        train_diseases = np.flatnonzero(positive.any(axis=0) & ~heldout)
        order = sorted(train_diseases, key=lambda q: hashlib.sha256(
            f"m3rep:{int(q)}".encode()).hexdigest())
        folds = np.full(positive.shape[1], -1, dtype=np.int8)
        load = np.zeros(5, dtype=np.int64)
        for q in order:
            fold = int(np.argmin(load))
            folds[q] = fold
            load[fold] += int(positive[:, q].sum())
        return folds

    def _pseudo_training(self, train_labels, all_labels, heldout, kernels, static):
        folds = self._folds(train_labels, heldout)
        rng = np.random.default_rng(self.seed + 30107)
        x_parts, y_parts, fold_parts = [], [], []
        for fold in range(5):
            q_idx = np.flatnonzero(folds == fold)
            if not len(q_idx):
                continue
            visible = train_labels.copy()
            visible[:, q_idx] = False
            planes = build_feature_planes(visible, kernels, static)
            pos_d, pos_local_q = np.where(train_labels[:, q_idx])
            pos_q = q_idx[pos_local_q]
            d_parts, q_parts = [pos_d], [pos_q]
            for q in q_idx:
                n_pos = int(train_labels[:, q].sum())
                if n_pos == 0:
                    continue
                available = np.flatnonzero(~all_labels[:, q])
                size = min(len(available), max(20 * n_pos, 1))
                if size:
                    chosen = rng.choice(available, size=size, replace=False)
                    d_parts.append(chosen)
                    q_parts.append(np.full(size, q, dtype=np.int32))
            d = np.concatenate(d_parts)
            q = np.concatenate(q_parts)
            y = np.zeros(len(d), dtype=np.int8)
            y[:len(pos_d)] = 1
            x_parts.append(rows_from_planes(planes, d, q))
            y_parts.append(y)
            fold_parts.append(np.full(len(y), fold, dtype=np.int8))
        if not x_parts:
            raise ValueError("No pseudo-training examples were generated")
        return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(fold_parts)

    def fit(self, train_labels, all_labels, heldout, kernels: Mapping[str, np.ndarray],
            static: Mapping[str, np.ndarray]) -> "M3RepRanker":
        train_labels = np.asarray(train_labels, dtype=bool)
        all_labels = np.asarray(all_labels, dtype=bool)
        heldout = np.asarray(heldout, dtype=bool)
        x, y, folds = self._pseudo_training(train_labels, all_labels, heldout, kernels, static)
        params = dict(learning_rate=0.035, num_leaves=15, max_depth=5,
                      min_child_samples=50, colsample_bytree=0.9, subsample=0.9,
                      reg_lambda=10, verbosity=-1, n_jobs=4, random_state=self.seed)
        train_mask, valid_mask = folds != 0, folds == 0
        provisional = lgb.LGBMClassifier(n_estimators=self.n_estimators, **params)
        if valid_mask.any() and train_mask.any() and len(np.unique(y[valid_mask])) > 1:
            provisional.fit(x[train_mask], y[train_mask], feature_name=self.feature_names,
                            eval_set=[(x[valid_mask], y[valid_mask])],
                            eval_metric="average_precision",
                            callbacks=[lgb.early_stopping(35, verbose=False)])
            rounds = max(int(provisional.best_iteration_), 30)
        else:
            rounds = self.n_estimators
        self.model = lgb.LGBMClassifier(n_estimators=rounds, **params)
        self.model.fit(x, y, feature_name=self.feature_names)
        self.best_iteration = int(rounds)
        return self

    def predict_scores(self, visible_labels, kernels, static, disease_indices=None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit before predict_scores")
        visible_labels = np.asarray(visible_labels, dtype=bool)
        n_drugs, n_diseases = visible_labels.shape
        if disease_indices is None:
            disease_indices = np.arange(n_diseases, dtype=np.int64)
        disease_indices = np.asarray(disease_indices, dtype=np.int64)
        planes = build_feature_planes(visible_labels, kernels, static)
        d = np.tile(np.arange(n_drugs), len(disease_indices))
        q = np.repeat(disease_indices, n_drugs)
        values = self.model.predict_proba(rows_from_planes(planes, d, q))[:, 1]
        return values.reshape(len(disease_indices), n_drugs).T

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted ranker")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "M3RepRanker":
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, cls):
            raise TypeError("The saved object is not an M3RepRanker")
        return value

    def metadata(self) -> dict:
        return {"model": "M3-Rep", "seed": self.seed,
                "best_iteration": self.best_iteration,
                "features": self.feature_names,
                "training": "disease-level pseudo-holdout with LightGBM"}
