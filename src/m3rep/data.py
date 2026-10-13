"""Input bundle validation and serialization."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import numpy as np

KERNEL_KEYS = (
    "disease_gene", "disease_phenotype", "disease_name", "disease_ontology",
    "disease_definition", "disease_clinical", "drug_gene", "drug_chem",
    "disease_llm", "disease_causal", "disease_causal_tfidf",
)
STATIC_KEYS = (
    "direct_target", "disease_gene_count", "disease_phenotype_count",
    "max_gene_neighbor", "max_name_neighbor", "definition_available",
    "clinical_available", "physchem_available",
)


def _bool_matrix(x, name):
    value = np.asarray(x, dtype=bool)
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return value


def load_bundle(path: str | Path) -> dict:
    """Load a prepared bundle saved by ``save_bundle``."""
    with np.load(path, allow_pickle=False) as z:
        required = {"drug_ids", "disease_ids", "train_labels", "all_labels", "test_labels"}
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"Missing bundle arrays: {', '.join(missing)}")
        bundle = {key: z[key] for key in z.files}
    bundle["train_labels"] = _bool_matrix(bundle["train_labels"], "train_labels")
    bundle["all_labels"] = _bool_matrix(bundle["all_labels"], "all_labels")
    bundle["test_labels"] = _bool_matrix(bundle["test_labels"], "test_labels")
    shape = bundle["train_labels"].shape
    if bundle["all_labels"].shape != shape or bundle["test_labels"].shape != shape:
        raise ValueError("label matrices must have the same shape")
    if len(bundle["drug_ids"]) != shape[0] or len(bundle["disease_ids"]) != shape[1]:
        raise ValueError("ID arrays do not match label dimensions")
    try:
        bundle["kernels"] = {key: np.asarray(bundle[f"kernel__{key}"], dtype=np.float32)
                             for key in KERNEL_KEYS}
        bundle["static"] = {key: np.asarray(bundle[f"static__{key}"], dtype=np.float32)
                            for key in STATIC_KEYS}
    except KeyError as error:
        raise ValueError(f"Missing feature array: {error.args[0]}") from error
    return bundle


def save_bundle(path: str | Path, *, drug_ids, disease_ids, train_labels,
                all_labels, test_labels, kernels: Mapping[str, np.ndarray],
                static: Mapping[str, np.ndarray]) -> None:
    payload = {
        "drug_ids": np.asarray(drug_ids).astype(str),
        "disease_ids": np.asarray(disease_ids).astype(str),
        "train_labels": np.asarray(train_labels, dtype=bool),
        "all_labels": np.asarray(all_labels, dtype=bool),
        "test_labels": np.asarray(test_labels, dtype=bool),
    }
    for key in KERNEL_KEYS:
        payload[f"kernel__{key}"] = np.asarray(kernels[key], dtype=np.float32)
    for key in STATIC_KEYS:
        payload[f"static__{key}"] = np.asarray(static[key], dtype=np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
