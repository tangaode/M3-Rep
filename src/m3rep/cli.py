"""Command-line entry points for M3-Rep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .data import load_bundle
from .metrics import evaluate, macro_summary
from .model import M3RepRanker


def train(args):
    bundle = load_bundle(args.data)
    heldout = bundle["test_labels"].any(axis=0)
    ranker = M3RepRanker(seed=args.seed)
    ranker.fit(bundle["train_labels"], bundle["all_labels"], heldout,
               bundle["kernels"], bundle["static"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ranker.save(out / "ranker.pkl")
    (out / "metadata.json").write_text(json.dumps(ranker.metadata(), indent=2), encoding="utf-8")
    print(json.dumps(ranker.metadata(), indent=2))


def predict(args):
    bundle = load_bundle(args.data)
    ranker = M3RepRanker.load(args.model)
    disease_indices = np.flatnonzero(bundle["test_labels"].any(axis=0))
    scores = ranker.predict_scores(bundle["train_labels"], bundle["kernels"],
                                   bundle["static"], disease_indices)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, disease_indices=disease_indices, scores=scores,
                        drug_ids=bundle["drug_ids"], disease_ids=bundle["disease_ids"][disease_indices])
    print(f"saved {out} ({scores.shape[0]} drugs x {scores.shape[1]} diseases)")


def evaluate_predictions(args):
    bundle = load_bundle(args.data)
    with np.load(args.predictions, allow_pickle=False) as z:
        disease_indices = z["disease_indices"]
        scores = z["scores"]
    truth = bundle["test_labels"][:, disease_indices]
    exclude = (bundle["train_labels"] | (bundle["all_labels"] & ~bundle["test_labels"]))[:, disease_indices]
    result = evaluate(scores, truth, exclude)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))
    print("\nmacro summary\n", macro_summary(result).to_string())


def parser():
    p = argparse.ArgumentParser(prog="m3rep")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("train")
    a.add_argument("--data", required=True); a.add_argument("--output", required=True)
    a.add_argument("--seed", type=int, default=42); a.set_defaults(func=train)
    a = sub.add_parser("predict")
    a.add_argument("--data", required=True); a.add_argument("--model", required=True)
    a.add_argument("--output", required=True); a.set_defaults(func=predict)
    a = sub.add_parser("evaluate")
    a.add_argument("--data", required=True); a.add_argument("--predictions", required=True)
    a.add_argument("--output", required=True); a.set_defaults(func=evaluate_predictions)
    return p


def main():
    args = parser().parse_args()
    args.func(args)
