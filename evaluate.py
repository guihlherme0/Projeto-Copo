"""Auditoria do CSV e avaliação repetida; nunca usa o ajuste final para medir métricas."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from water_classifier.data import CLASSES, audit_dataset, load_dataset
from water_classifier.model import candidates


def evaluate(csv_path, repeats=5, folds=5):
    features, labels = load_dataset(csv_path)
    counts = {label: int((labels == label).sum()) for label in CLASSES}
    if min(counts.values()) < folds:
        raise ValueError("Há menos registros em uma classe que dobras solicitadas.")
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    splits = list(splitter.split(features, labels))
    results = {}
    for name, candidate in candidates().items():
        true, predicted, fold_scores = [], [], []
        for train, valid in splits:
            model = clone(candidate).fit(features[train], labels[train])
            predictions = model.predict(features[valid])
            true.extend(labels[valid].tolist())
            predicted.extend(predictions.tolist())
            fold_scores.append({
                "accuracy": float(accuracy_score(labels[valid], predictions)),
                "macro_f1": float(f1_score(labels[valid], predictions, labels=CLASSES, average="macro", zero_division=0)),
            })
        report = classification_report(true, predicted, labels=CLASSES, output_dict=True, zero_division=0)
        results[name] = {
            "accuracy": float(accuracy_score(true, predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {label: {metric: float(report[label][metric]) for metric in ("precision", "recall", "f1-score", "support")} for label in CLASSES},
            "confusion_matrix": confusion_matrix(true, predicted, labels=CLASSES).tolist(),
            "fold_accuracy_mean": float(np.mean([s["accuracy"] for s in fold_scores])),
            "fold_accuracy_std": float(np.std([s["accuracy"] for s in fold_scores])),
            "fold_macro_f1_mean": float(np.mean([s["macro_f1"] for s in fold_scores])),
            "fold_macro_f1_std": float(np.std([s["macro_f1"] for s in fold_scores])),
            "fold_macro_f1_min": float(np.min([s["macro_f1"] for s in fold_scores])),
            "fold_macro_f1_max": float(np.max([s["macro_f1"] for s in fold_scores])),
            "fold_scores": fold_scores,
        }
    # Média por dobra pondera igualmente cada validação. Desempate: recall da minoria.
    eligible = [name for name in results if name != "referencia_maioria"]
    winner = max(eligible, key=lambda name: (
        results[name]["fold_macro_f1_mean"],
        results[name]["per_class"]["limpo"]["recall"],
    ))
    return {
        "data_audit": audit_dataset(features, labels),
        "evaluation": {"method": "RepeatedStratifiedKFold", "folds": folds, "repeats": repeats, "random_state": 42, "class_order": list(CLASSES), "groups_available": False},
        "algorithms": results,
        "winner": winner,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="res.csv")
    parser.add_argument("--output", default="reports/evaluation.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    result = evaluate(args.csv, args.repeats, args.folds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Avaliação salva em {output}; vencedor: {result['winner']}")
    for name, metrics in result["algorithms"].items():
        print(f"{name:22} acurácia={metrics['accuracy']:.3f} F1 macro={metrics['macro_f1']:.3f} "
              f"F1 macro/dobra={metrics['fold_macro_f1_mean']:.3f} ± {metrics['fold_macro_f1_std']:.3f} "
              f"recall limpo={metrics['per_class']['limpo']['recall']:.3f} "
              f"matriz={metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
