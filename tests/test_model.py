import numpy as np
from m3rep.model import M3RepRanker
from tests.test_features import bundle


def test_fit_and_score():
    labels, kernels, static = bundle(n_drugs=24, n_diseases=12)
    train = labels.copy(); test = np.zeros_like(train)
    train[:, 8:] = False
    test[:, 8:] = labels[:, 8:]
    all_labels = train | test
    ranker = M3RepRanker(seed=3, n_estimators=12)
    ranker.fit(train, all_labels, test.any(axis=0), kernels, static)
    scores = ranker.predict_scores(train, kernels, static, np.arange(8, 12))
    assert scores.shape == (24, 4)
    assert np.isfinite(scores).all()
