"""Compara SVM, Random Forest e KNN no CSV antes do treinamento final."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from water_classifier.data import CLASSES, audit_dataset, load_dataset
from water_classifier.image_training import load_image_dataset
from water_classifier.model import candidates


def evaluate(csv_path="res.csv", photos_manifest="water_regions.json"):
    features, labels = load_dataset(csv_path)
    if min(np.count_nonzero(labels == name) for name in CLASSES) < 5:
        raise ValueError("Cada classe precisa de pelo menos cinco registros.")

    splits = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=42
    ).split(features, labels))
    results = {}
    for name, candidate in candidates().items():
        actual, predicted, fold_f1 = [], [], []
        for train, test in splits:
            model = clone(candidate).fit(features[train], labels[train])
            guess = model.predict(features[test])
            actual.extend(labels[test])
            predicted.extend(guess)
            fold_f1.append(f1_score(labels[test], guess, labels=CLASSES,
                                    average="macro", zero_division=0))
        report = classification_report(actual, predicted, labels=CLASSES,
                                       output_dict=True, zero_division=0)
        results[name] = {
            "fold_macro_f1_mean": float(np.mean(fold_f1)),
            "fold_macro_f1_std": float(np.std(fold_f1)),
            "accuracy": float(accuracy_score(actual, predicted)),
            "per_class": {
                label: {metric: float(report[label][metric])
                        for metric in ("precision", "recall", "f1-score", "support")}
                for label in CLASSES
            },
            "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
        }

    winner = max(results, key=lambda name: (
        results[name]["fold_macro_f1_mean"],
        results[name]["per_class"]["limpo"]["recall"],
    ))
    result = {
        "source_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "data_audit": audit_dataset(features, labels),
        "evaluation": {
            "method": "RepeatedStratifiedKFold", "folds": 5, "repeats": 5,
            "random_state": 42, "class_order": list(CLASSES),
            "selection_metric": "fold_macro_f1_mean",
        },
        "algorithms": results,
        "winner": winner,
    }
    if photos_manifest:
        photo_features, photo_labels, paths, _ = load_image_dataset(
            photos_manifest, use_crop=False
        )
        result["photo_check"] = {
            "region": "full_image",
            "note": "Fotos externas ao CSV; não participam da escolha do vencedor.",
            "algorithms": {},
        }
        for name, candidate in candidates().items():
            guess = clone(candidate).fit(features, labels).predict(photo_features)
            result["photo_check"]["algorithms"][name] = {
                "correct": int(np.sum(guess == photo_labels)),
                "clean_correct": int(np.sum((guess == photo_labels) & (photo_labels == "limpo"))),
                "photos": [
                    {"path": path, "actual": str(actual), "predicted": str(prediction)}
                    for path, actual, prediction in zip(paths, photo_labels, guess)
                ],
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="res.csv")
    parser.add_argument("--photos", default="water_regions.json")
    parser.add_argument("--output", default="reports/csv_model_selection.json")
    args = parser.parse_args()
    result = evaluate(args.csv, args.photos)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, metrics in result["algorithms"].items():
        print(f"{name}: F1 macro/dobra={metrics['fold_macro_f1_mean']:.3f}; "
              f"acurácia={metrics['accuracy']:.3f}; "
              f"recall limpo={metrics['per_class']['limpo']['recall']:.3f}")
    print(f"Vencedor: {result['winner']}; relatório: {output}")


if __name__ == "__main__":
    main()
