"""Feature construction for M3-Rep."""
from __future__ import annotations

from typing import Mapping
import numpy as np

GRAPH_FEATURES = (
    "direct_target", "gene_transfer", "phenotype_transfer", "name_transfer",
    "ontology_transfer", "drug_gene_transfer", "drug_degree",
    "disease_gene_count", "disease_phenotype_count", "max_gene_neighbor",
    "max_name_neighbor",
)
MULTIMODAL_FEATURES = (
    "definition_transfer", "clinical_transfer", "chem_name_transfer",
    "chem_gene_transfer", "definition_available", "clinical_available",
    "physchem_available",
)
MECHANISM_FEATURES = (
    "mechanism_full_transfer", "mechanism_causal_transfer",
    "causal_text_transfer",
)
FEATURE_NAMES = GRAPH_FEATURES + MULTIMODAL_FEATURES + MECHANISM_FEATURES


def _as_float(x: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def topk_kernel(similarity: np.ndarray, k: int, include_self: bool = False) -> np.ndarray:
    """Keep the k strongest non-negative neighbours in each row."""
    matrix = _as_float(similarity, "similarity").copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if not include_self:
        np.fill_diagonal(matrix, 0)
    matrix[matrix < 1e-7] = 0
    k = max(1, min(int(k), matrix.shape[1]))
    order = np.argpartition(matrix, -k, axis=1)[:, -k:]
    weights = np.take_along_axis(matrix, order, axis=1)
    weights = np.maximum(weights, 0) ** 2
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    result = np.zeros_like(matrix)
    np.put_along_axis(result, order, weights, axis=1)
    return result


def _check_pair(name: str, matrix: np.ndarray, n_drugs: int, n_diseases: int) -> np.ndarray:
    value = _as_float(matrix, name)
    if value.shape != (n_drugs, n_diseases):
        raise ValueError(f"{name} must have shape {(n_drugs, n_diseases)}, got {value.shape}")
    return value


def validate_kernels(kernels: Mapping[str, np.ndarray], static: Mapping[str, np.ndarray],
                     n_drugs: int, n_diseases: int) -> None:
    disease_keys = ("disease_gene", "disease_phenotype", "disease_name", "disease_ontology",
                    "disease_definition", "disease_clinical", "disease_llm",
                    "disease_causal", "disease_causal_tfidf")
    for key in disease_keys:
        matrix = _as_float(kernels[key], key)
        if matrix.shape != (n_diseases, n_diseases):
            raise ValueError(f"{key} must have shape {(n_diseases, n_diseases)}, got {matrix.shape}")
    for key in ("drug_gene", "drug_chem"):
        matrix = _as_float(kernels[key], key)
        if matrix.shape != (n_drugs, n_drugs):
            raise ValueError(f"{key} must have shape {(n_drugs, n_drugs)}, got {matrix.shape}")
    _check_pair("direct_target", static["direct_target"], n_drugs, n_diseases)
    for key in ("disease_gene_count", "disease_phenotype_count", "max_gene_neighbor", "max_name_neighbor"):
        value = _as_float(static[key], key)
        if value.shape != (n_diseases,):
            raise ValueError(f"{key} must have shape {(n_diseases,)}, got {value.shape}")
    for key in ("definition_available", "clinical_available"):
        value = _as_float(static[key], key)
        if value.shape != (n_diseases,):
            raise ValueError(f"{key} must have shape {(n_diseases,)}, got {value.shape}")
    value = _as_float(static["physchem_available"], "physchem_available")
    if value.shape != (n_drugs,):
        raise ValueError(f"physchem_available must have shape {(n_drugs,)}, got {value.shape}")


def build_feature_planes(positive: np.ndarray, kernels: Mapping[str, np.ndarray],
                         static: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Build one matrix per model feature from visible drug--disease labels."""
    positive = np.asarray(positive, dtype=np.float32)
    if positive.ndim != 2:
        raise ValueError("positive must be a two-dimensional drug-by-disease matrix")
    n_drugs, n_diseases = positive.shape
    validate_kernels(kernels, static, n_drugs, n_diseases)
    count = positive.sum(axis=1)
    scaled = positive / np.maximum(count[:, None], 1) ** 0.5
    gene = scaled @ kernels["disease_gene"].T
    phenotype = scaled @ kernels["disease_phenotype"].T
    name = scaled @ kernels["disease_name"].T
    ontology = scaled @ kernels["disease_ontology"].T
    drug_gene = kernels["drug_gene"] @ scaled @ kernels["disease_gene"].T
    definition = scaled @ kernels["disease_definition"].T
    clinical = scaled @ kernels["disease_clinical"].T
    chem_name = (kernels["drug_chem"] @ scaled) @ kernels["disease_name"].T
    chem_gene = (kernels["drug_chem"] @ scaled) @ kernels["disease_gene"].T
    mechanism_full = scaled @ kernels["disease_llm"].T
    mechanism_causal = scaled @ kernels["disease_causal"].T
    causal_text = scaled @ kernels["disease_causal_tfidf"].T
    shape = positive.shape
    planes = [
        _check_pair("direct_target", static["direct_target"], n_drugs, n_diseases),
        gene, phenotype, name, ontology, drug_gene,
        np.broadcast_to(np.log1p(count)[:, None], shape),
        np.broadcast_to(np.log1p(static["disease_gene_count"])[None, :], shape),
        np.broadcast_to(np.log1p(static["disease_phenotype_count"])[None, :], shape),
        np.broadcast_to(static["max_gene_neighbor"][None, :], shape),
        np.broadcast_to(static["max_name_neighbor"][None, :], shape),
        definition, clinical, chem_name, chem_gene,
        np.broadcast_to(static["definition_available"][None, :], shape),
        np.broadcast_to(static["clinical_available"][None, :], shape),
        np.broadcast_to(static["physchem_available"][:, None], shape),
        mechanism_full, mechanism_causal, causal_text,
    ]
    return [_as_float(x, FEATURE_NAMES[i]) for i, x in enumerate(planes)]


def rows_from_planes(planes: list[np.ndarray], drug_index: np.ndarray,
                     disease_index: np.ndarray) -> np.ndarray:
    drug_index = np.asarray(drug_index, dtype=np.int64)
    disease_index = np.asarray(disease_index, dtype=np.int64)
    if drug_index.shape != disease_index.shape:
        raise ValueError("drug_index and disease_index must have the same shape")
    return np.column_stack([plane[drug_index, disease_index] for plane in planes]).astype(np.float32)
