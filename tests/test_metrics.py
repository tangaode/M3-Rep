import numpy as np
from m3rep.metrics import evaluate


def test_ranking_metrics():
    scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.3]])
    truth = np.array([[True, False], [False, True], [False, False]])
    exclude = np.zeros_like(truth, dtype=bool)
    result = evaluate(scores, truth, exclude)
    assert len(result) == 2
    assert (result["Recall@1"] == 1).all()
    assert (result["MRR"] == 1).all()
