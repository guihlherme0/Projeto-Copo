"""Validação deixando uma foto inteira fora do treino a cada rodada."""

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from water_classifier.data import CLASSES
from water_classifier.image_training import image_pipeline, load_image_dataset


def evaluate(manifest_path):
    features, labels, names, source_sha256 = load_image_dataset(manifest_path, include_focus=True)
    predictions = []
    for held_out in range(len(labels) // 2):
        train = [index for index in range(len(labels)) if index // 2 != held_out]
        model = image_pipeline().fit(features[train], labels[train])
        # A interface usa a metade central da seleção; as duas versões da
        # foto deixada de fora ficam fora do ajuste desta rodada.
        predictions.append(model.predict(features[2 * held_out + 1:2 * held_out + 2])[0])
    actual = labels[::2]
    paths = names[::2]
    return {
        "method": "leave-one-image-out",
        "source_sha256": source_sha256,
        "class_order": list(CLASSES),
        "accuracy": float(accuracy_score(actual, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=CLASSES).tolist(),
        "photos": [
            {"path": path, "actual": actual, "predicted": predicted}
            for path, actual, predicted in zip(paths, actual, predictions)
        ],
        "limitation": "As duas versões de cada foto são mantidas juntas fora do treino; a característica e as regiões foram escolhidas após inspecionar estas fotos, portanto a avaliação não mede generalização externa.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="water_regions.json")
    parser.add_argument("--output", default="reports/image_evaluation.json")
    args = parser.parse_args()
    result = evaluate(args.images)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['accuracy']:.3f} acurácia em {len(result['photos'])} fotos; relatório em {path}")


if __name__ == "__main__":
    main()
