import numpy as np
from m3rep.data import load_bundle, save_bundle
from tests.test_features import bundle


def test_bundle_roundtrip(tmp_path):
    labels, kernels, static = bundle()
    path = tmp_path / "bundle.npz"
    save_bundle(path, drug_ids=["d0", "d1", "d2", "d3", "d4"],
                disease_ids=["q0", "q1", "q2", "q3"],
                train_labels=labels.astype(bool), all_labels=labels.astype(bool),
                test_labels=labels.astype(bool), kernels=kernels, static=static)
    value = load_bundle(path)
    assert value["train_labels"].shape == labels.shape
    assert set(value["kernels"]) == set(kernels)
