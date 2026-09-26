"""Leitura e validação do contrato observado em res.csv."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = tuple(f"{channel}{level}" for channel in "rgb" for level in range(256))
CLASSES = ("limpo", "sujo")


def load_dataset(path: str | Path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        expected = [*FEATURE_COLUMNS, "class"]
        if reader.fieldnames != expected:
            raise ValueError("Cabeçalho inesperado: esperado r0..r255, g0..g255, b0..b255, class.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV vazio.")
    if any(None in row or any(value is None or value == "" for value in row.values()) for row in rows):
        raise ValueError("CSV contém campos ausentes ou linhas incompletas.")
    try:
        features = np.asarray([[int(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("As contagens RGB devem ser inteiros.") from exc
    if not np.isfinite(features).all() or (features < 0).any() or (features != np.floor(features)).any():
        raise ValueError("As contagens RGB devem ser inteiros finitos não negativos.")
    labels = np.asarray([row["class"] for row in rows])
    if set(labels) != set(CLASSES):
        raise ValueError(f"Classes inesperadas: {sorted(set(labels))}.")
    sums = features.reshape(-1, 3, 256).sum(axis=2)
    if (sums <= 0).any() or not (sums == sums[:, [0]]).all():
        raise ValueError("Cada canal deve contar o mesmo número positivo de pixels.")
    return features, labels


def audit_dataset(features, labels):
    sums = features.reshape(-1, 3, 256).sum(axis=2)
    duplicate_features = len(features) - len(np.unique(features, axis=0))
    return {
        "records": len(features),
        "features": features.shape[1],
        "classes": dict(Counter(labels)),
        "missing": 0,
        "duplicate_feature_rows": int(duplicate_features),
        "pixel_counts": dict(Counter(str(int(n)) for n in sums[:, 0])),
        "pixel_counts_by_class": {
            label: dict(Counter(str(int(n)) for n in sums[labels == label, 0]))
            for label in CLASSES
        },
        "min_count": int(features.min()),
        "max_count": int(features.max()),
    }
