"""Reavalia métodos dos notebooks do professor usando o res.csv deste projeto.

O notebook de seleção exibe 61 registros; esta reprodução usa somente as 50
linhas disponíveis e ajusta a seleção de atributos dentro de cada dobra.
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from water_classifier.data import CLASSES, load_dataset
from water_classifier.image_training import load_image_dataset


def candidates():
    return {
        "arvore_contagens_brutas": DecisionTreeClassifier(random_state=42),
        "knn_5_contagens_brutas": KNeighborsClassifier(n_neighbors=5),
        "chi2_200_arvore": make_pipeline(
            SelectKBest(chi2, k=200), DecisionTreeClassifier(random_state=42)
        ),
        "anova_500_arvore": make_pipeline(
            SelectKBest(f_classif, k=500), DecisionTreeClassifier(random_state=42)
        ),
    }


def evaluate(csv_path, manifest_path):
    features, labels = load_dataset(csv_path)
    photo_features, photo_labels, names, photo_hash = load_image_dataset(manifest_path, use_crop=False)
    splits = list(RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=42
    ).split(features, labels))
    results = {}
    for name, candidate in candidates().items():
        scores, actual, predicted = [], [], []
        warning_count = 0
        for train, test in splits:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = clone(candidate).fit(features[train], labels[train])
            warning_count += len(caught)
            fold_predicted = fitted.predict(features[test])
            scores.append(f1_score(labels[test], fold_predicted, labels=CLASSES,
                                   average="macro", zero_division=0))
            actual.extend(labels[test])
            predicted.extend(fold_predicted)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = clone(candidate).fit(features, labels)
        warning_count += len(caught)
        photo_predicted = fitted.predict(photo_features)
        results[name] = {
            "fold_macro_f1_mean": float(np.mean(scores)),
            "fold_macro_f1_std": float(np.std(scores)),
            "repeated_cv_accuracy": float(accuracy_score(actual, predicted)),
            "repeated_cv_confusion_matrix": confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
            "photos_correct": int(np.sum(photo_predicted == photo_labels)),
            "clean_photos_correct": int(np.sum((photo_predicted == photo_labels) & (photo_labels == "limpo"))),
            "photo_predictions": [
                {"path": path, "actual": str(actual), "predicted": str(prediction)}
                for path, actual, prediction in zip(names, photo_labels, photo_predicted)
            ],
            "fit_warning_count": warning_count,
        }
    return {
        "csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "csv_records": len(labels),
        "photo_manifest_sha256": photo_hash,
        "evaluation": "RepeatedStratifiedKFold, 5 dobras × 5 repetições, random_state=42; seleção ajustada somente no treino de cada dobra.",
        "photo_check": "Nove fotos inteiras; teste adicional, não usado para escolher o vencedor no CSV.",
        "class_order": list(CLASSES),
        "algorithms": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="res.csv")
    parser.add_argument("--images", default="water_regions.json")
    parser.add_argument("--output", default="reports/notebook_methods_evaluation.json")
    args = parser.parse_args()
    result = evaluate(args.csv, args.images)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, metrics in result["algorithms"].items():
        print(f"{name}: F1 macro/dobra={metrics['fold_macro_f1_mean']:.3f}; "
              f"fotos={metrics['photos_correct']}/9; "
              f"limpas={metrics['clean_photos_correct']}/4")
    print(f"Relatório salvo em {output}")


if __name__ == "__main__":
    main()
