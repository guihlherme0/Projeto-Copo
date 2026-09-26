"""Compara as fotos disponíveis com o contrato RGB observado em res.csv.

Este diagnóstico não participa da escolha nem do treinamento do modelo.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from water_classifier.data import CLASSES, load_dataset
from water_classifier.features import extract_histogram
from water_classifier.model import load_model


def _distances_to_csv(csv_features, photo_features, pipeline):
    """Distâncias euclidianas no espaço normalizado e escalado do SVM."""
    preprocessing = pipeline[:-1]
    train = preprocessing.transform(csv_features)
    photos = preprocessing.transform(photo_features)
    train_distances = np.linalg.norm(train[:, None, :] - train[None, :, :], axis=2)
    np.fill_diagonal(train_distances, np.inf)
    photo_distances = np.linalg.norm(photos[:, None, :] - train[None, :, :], axis=2)
    return float(np.percentile(train_distances.min(axis=1), 95)), photo_distances.min(axis=1)


def audit(csv_path, manifest_path, model_path):
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    csv_features, _ = load_dataset(csv_path)
    bundle = load_model(model_path)
    if bundle["source_sha256"] != hashlib.sha256(csv_path.read_bytes()).hexdigest():
        raise ValueError("O modelo salvo não foi treinado com este CSV.")
    if bundle["algorithm"] != "svm_rbf":
        raise ValueError("Este diagnóstico de distância exige o SVM padrão.")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not entries or any(entry["label"] not in CLASSES for entry in entries):
        raise ValueError("Manifesto de fotos inválido.")

    full, cropped, resized, photos = [], [], [], []
    for entry in entries:
        path = manifest_path.parent / entry["path"]
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            full_histogram = extract_histogram(oriented)
            full.append(full_histogram)
            cropped.append(extract_histogram(oriented, entry["crop"]))
            # Hipótese de outra implementação encontrada; não há evidência de
            # que o CSV original tenha sido produzido com esse redimensionamento.
            resized.append(extract_histogram(oriented.resize((4000, 3000), Image.Resampling.BILINEAR)))
            photos.append({
                "path": entry["path"],
                "label": entry["label"],
                "dimensions_after_exif": [oriented.width, oriented.height],
                "exact_csv_match": bool(np.any(np.all(csv_features == full_histogram, axis=1))),
            })

    full, cropped, resized = (np.asarray(values) for values in (full, cropped, resized))
    pipeline = bundle["pipeline"]
    reference_95, nearest = _distances_to_csv(csv_features, full, pipeline)
    for photo, distance, prediction in zip(photos, nearest, pipeline.predict(full)):
        photo["nearest_csv_distance"] = round(float(distance), 2)
        photo["prediction_full_image"] = str(prediction)

    variants = {
        "full_image": full,
        "water_crop": cropped,
        "resize_full_image_4000x3000_bilinear": resized,
    }
    for order in itertools.permutations(range(3)):
        if order != (0, 1, 2):
            variants["full_image_channels_" + "".join("rgb"[index] for index in order)] = (
                full.reshape(-1, 3, 256)[:, order, :].reshape(-1, 768)
            )
    predictions_by_variant = {}
    labels = [photo["label"] for photo in photos]
    for name, features in variants.items():
        predictions = pipeline.predict(features).tolist()
        predictions_by_variant[name] = {
            "correct": sum(actual == predicted for actual, predicted in zip(labels, predictions)),
            "clean_predicted_clean": sum(actual == predicted == "limpo" for actual, predicted in zip(labels, predictions)),
            "predictions": predictions,
        }
    return {
        "csv_sha256": bundle["source_sha256"],
        "model_algorithm": bundle["algorithm"],
        "csv_records": len(csv_features),
        "photos_count": len(photos),
        "distance_definition": "Distância euclidiana após normalização RGB e StandardScaler do SVM; diagnóstico descritivo, não limiar de rejeição calibrado.",
        "csv_nearest_neighbor_distance_p95": round(reference_95, 2),
        "photos": photos,
        "variants": predictions_by_variant,
        "limitation": "Sem as 50 imagens ou o script que gerou res.csv, não é possível confirmar a extração RGB original ou conferir os rótulos do CSV.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="res.csv")
    parser.add_argument("--images", default="water_regions.json")
    parser.add_argument("--model", default="models/water_model.joblib")
    parser.add_argument("--output", default="reports/feature_contract_audit.json")
    args = parser.parse_args()
    result = audit(args.csv, args.images, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Auditoria de {result['photos_count']} fotos salva em {output}")


if __name__ == "__main__":
    main()
