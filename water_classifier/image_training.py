"""Treino experimental a partir das fotos rotuladas e regiões de água."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from .data import CLASSES
from .features import MeanRedGreen, NormalizeHistograms, extract_histogram


class ColorIntervalClassifier(ClassifierMixin, BaseEstimator):
    """Aprende a faixa de cor das fotos limpas; cores fora dela são sujas."""

    def fit(self, X, y):
        values = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y)
        if values.ndim != 2 or values.shape[1] != 1 or not np.isfinite(values).all():
            raise ValueError("Esperada uma diferença de cor finita por foto.")
        if set(labels) != set(CLASSES) or (labels == "limpo").sum() < 2:
            raise ValueError("São necessárias pelo menos duas fotos limpas e uma suja.")
        clean = values[labels == "limpo", 0]
        dirty = values[labels == "sujo", 0]
        self.center_ = float(clean.mean())
        # Água sem tonalidade deve ser aceita mesmo se as fotos limpas de
        # treinamento compartilham um pequeno desvio de iluminação.
        clean_radius = float(max(np.max(np.abs(clean - self.center_)), abs(self.center_)))
        dirty_radius = float(np.min(np.abs(dirty - self.center_)))
        if dirty_radius <= clean_radius:
            raise ValueError("As cores das fotos limpas e sujas se sobrepõem neste modelo.")
        self.threshold_ = (clean_radius + dirty_radius) / 2
        self.classes_ = np.asarray(CLASSES)
        self.n_features_in_ = 1
        return self

    def predict(self, X):
        if not hasattr(self, "threshold_"):
            raise ValueError("O classificador ainda não foi treinado.")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 1 or not np.isfinite(values).all():
            raise ValueError("Esperada uma diferença de cor finita por foto.")
        return np.where(np.abs(values[:, 0] - self.center_) <= self.threshold_, "limpo", "sujo")


def image_pipeline():
    return Pipeline([
        ("histograma", NormalizeHistograms()),
        ("diferenca_rg", MeanRedGreen()),
        ("classificador", ColorIntervalClassifier()),
    ])


def load_image_dataset(manifest_path):
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("O manifesto precisa listar as fotos de treino.")
    features, labels, names = [], [], []
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "label", "crop"}:
            raise ValueError("Entrada inválida no manifesto de fotos.")
        if entry["label"] not in CLASSES or not isinstance(entry["path"], str):
            raise ValueError("Foto com caminho ou classe inválida.")
        path = manifest_path.parent / entry["path"]
        if not path.is_file():
            raise ValueError(f"Foto de treino não encontrada: {path}")
        with Image.open(path) as image:
            features.append(extract_histogram(image, entry["crop"]))
        labels.append(entry["label"])
        names.append(entry["path"])
        digest.update(entry["path"].encode("utf-8"))
        digest.update(path.read_bytes())
    if len(set(names)) != len(names):
        raise ValueError("A mesma foto aparece mais de uma vez no manifesto.")
    if set(labels) != set(CLASSES):
        raise ValueError("O manifesto precisa conter as duas classes.")
    return np.asarray(features), np.asarray(labels), names, digest.hexdigest()
