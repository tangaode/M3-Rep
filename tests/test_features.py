import numpy as np
from m3rep.features import FEATURE_NAMES, build_feature_planes, rows_from_planes


def bundle(n_drugs=5, n_diseases=4):
    rng = np.random.default_rng(7)
    def disease():
        x = rng.random((n_diseases, n_diseases), dtype=np.float32)
        np.fill_diagonal(x, 0)
        return x
    kernels = {key: disease() for key in (
        "disease_gene", "disease_phenotype", "disease_name", "disease_ontology",
        "disease_definition", "disease_clinical", "disease_llm", "disease_causal",
        "disease_causal_tfidf")}
    kernels["drug_gene"] = rng.random((n_drugs, n_drugs), dtype=np.float32)
    kernels["drug_chem"] = rng.random((n_drugs, n_drugs), dtype=np.float32)
    static = {
        "direct_target": rng.random((n_drugs, n_diseases), dtype=np.float32),
        "disease_gene_count": rng.random(n_diseases, dtype=np.float32),
        "disease_phenotype_count": rng.random(n_diseases, dtype=np.float32),
        "max_gene_neighbor": rng.random(n_diseases, dtype=np.float32),
        "max_name_neighbor": rng.random(n_diseases, dtype=np.float32),
        "definition_available": np.ones(n_diseases, dtype=np.float32),
        "clinical_available": np.ones(n_diseases, dtype=np.float32),
        "physchem_available": np.ones(n_drugs, dtype=np.float32),
    }
    return rng.integers(0, 2, (n_drugs, n_diseases), dtype=np.int8), kernels, static


def test_feature_count_and_rows():
    labels, kernels, static = bundle()
    planes = build_feature_planes(labels, kernels, static)
    assert len(planes) == len(FEATURE_NAMES) == 21
    rows = rows_from_planes(planes, np.array([0, 2]), np.array([1, 3]))
    assert rows.shape == (2, 21)
    assert np.isfinite(rows).all()
