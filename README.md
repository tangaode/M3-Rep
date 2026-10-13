# M3-Rep

M3-Rep is a multimodal drug-disease prioritization model for drug repurposing. For each disease, it ranks candidate drugs by combining four information sources:

1. biological graph relationships, including drug-target and disease-gene or phenotype links;
2. molecular similarity between drugs;
3. disease definitions and clinical descriptions;
4. treatment-blinded mechanism representations and a causal-text similarity channel.

A LightGBM classifier converts the resulting drug-disease features into a ranking score. Training uses disease-level pseudo-holdouts: treatment records for a subset of training diseases are hidden while the model learns transfer patterns from the remaining diseases. Evaluation then ranks drugs for diseases whose held-out treatment records were not used to construct the features.

The public repository contains the model implementation, input contract, command-line entry points and tests. Source databases, generated embeddings, checkpoints and study-specific output files are kept outside the code repository and should be downloaded from their original providers.

## Installation

```bash
git clone https://github.com/tangaode/M3-Rep.git
cd M3-Rep
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Prepare an input bundle

Create a compressed NumPy archive following [docs/data_format.md](docs/data_format.md). The helper below shows the required Python interface:

```python
import numpy as np
from m3rep.data import save_bundle

save_bundle(
    "data/m3rep_bundle.npz",
    drug_ids=np.array(["DB00001", "DB00002"]),
    disease_ids=np.array(["D1", "D2"]),
    train_labels=train_labels,
    all_labels=all_labels,
    test_labels=test_labels,
    kernels=kernels,
    static=static,
)
```

The graph, chemistry, text and mechanism matrices must be aligned to the same IDs. A mechanism matrix is a similarity matrix, not a treatment label; treatment records are supplied only through the label matrices.

## Train, rank and evaluate

```bash
python -m m3rep train --data data/m3rep_bundle.npz --output outputs/m3rep --seed 42
python -m m3rep predict --data data/m3rep_bundle.npz --model outputs/m3rep/ranker.pkl --output outputs/m3rep/predictions.npz
python -m m3rep evaluate --data data/m3rep_bundle.npz --predictions outputs/m3rep/predictions.npz --output outputs/m3rep/per_disease.csv
```

The prediction file contains a score for each drug and each held-out disease. The evaluation command reports disease-level AP, Recall@K, NDCG@K, MRR and AUROC, followed by macro averages.

## Reproducibility

All random sampling is controlled by the `--seed` argument. The code records the selected tree count and feature names in `metadata.json`. To compare models, keep the disease split, candidate-drug pool, label masks and metric implementation fixed, then change only the feature group or comparator.

## Code layout

- `src/m3rep/features.py` builds graph, chemistry, text and mechanism transfer features.
- `src/m3rep/model.py` implements pseudo-holdout training, scoring and serialization.
- `src/m3rep/metrics.py` implements the ranking metrics.
- `src/m3rep/data.py` validates and serializes prepared input bundles.
- `tests/` contains deterministic checks for feature construction and ranking metrics.

## References

The graph and disease-area evaluation protocol follows the public TxGNN implementation. M3-Rep adds aligned chemistry, disease-text and treatment-blinded mechanism channels before ranking candidates with LightGBM. Cite the accompanying M3-Rep manuscript together with the data sources used in a particular analysis.
