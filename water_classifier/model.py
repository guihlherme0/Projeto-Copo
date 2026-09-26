"""Pipelines candidatos e persistência do modelo final."""

import hashlib
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import CLASSES, FEATURE_COLUMNS, load_dataset
from .features import NormalizeHistograms
from .image_training import image_pipeline, load_image_dataset


def candidates():
    return {
        "svm_rbf": Pipeline([
            ("histograma", NormalizeHistograms()),
            ("escala", StandardScaler()),
            ("classificador", SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=42)),
        ]),
        "floresta_aleatoria": Pipeline([
            ("histograma", NormalizeHistograms()),
            ("classificador", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)),
        ]),
        "knn_3": Pipeline([
            ("histograma", NormalizeHistograms()),
            ("escala", StandardScaler()),
            ("classificador", KNeighborsClassifier(n_neighbors=3)),
        ]),
    }


def train_final(csv_path, model_path, algorithm):
    available = candidates()
    if algorithm not in available:
        raise ValueError("Algoritmo final inválido.")
    features, labels = load_dataset(csv_path)
    pipeline = available[algorithm].fit(features, labels)
    bundle = {
        "pipeline": pipeline,
        "algorithm": algorithm,
        "feature_columns": FEATURE_COLUMNS,
        "classes": tuple(pipeline.classes_),
        "source_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "training_records": len(labels),
        "region": "full_image",
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def train_from_images(manifest_path, model_path):
    features, labels, _, source_sha256 = load_image_dataset(manifest_path, include_focus=True)
    pipeline = image_pipeline().fit(features, labels)
    bundle = {
        "pipeline": pipeline,
        "algorithm": "intervalo_cor_rg",
        "feature_columns": FEATURE_COLUMNS,
        "classes": tuple(pipeline.classes_),
        "source_sha256": source_sha256,
        "training_records": len(labels) // 2,
        "region": "water_crop",
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load_model(model_path):
    bundle = joblib.load(model_path)
    if (tuple(bundle["feature_columns"]) != FEATURE_COLUMNS
            or set(bundle["classes"]) != set(CLASSES)
            or bundle.get("region") not in {"full_image", "water_crop"}):
        raise ValueError("Modelo incompatível com o contrato de características.")
    return bundle
