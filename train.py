"""Treina o modelo visual ou um algoritmo do CSV em todos os registros."""

import argparse
import hashlib
import json
from pathlib import Path

from water_classifier.model import candidates, train_final, train_from_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("csv", "images"), default="images")
    parser.add_argument("--csv", default="res.csv")
    parser.add_argument("--images", default="water_regions.json")
    parser.add_argument("--model", help="Destino do modelo; padrão depende da fonte.")
    parser.add_argument("--report", default="reports/csv_model_selection.json")
    parser.add_argument("--algorithm", choices=tuple(candidates()),
                        help="Testar outro algoritmo do CSV em um arquivo separado.")
    args = parser.parse_args()
    model_path = args.model or ("models/water_image_interval.joblib" if args.source == "images" else "models/water_model.joblib")
    if args.source == "images":
        if args.algorithm:
            parser.error("--algorithm só pode ser usado com --source csv.")
        bundle = train_from_images(args.images, model_path)
    else:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        csv_hash = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
        if report.get("source_sha256") != csv_hash:
            parser.error("O relatório de avaliação não corresponde ao CSV; execute evaluate.py novamente.")
        if args.algorithm and not args.model:
            parser.error("Use --model para não substituir o SVM vencedor ao testar outro algoritmo.")
        bundle = train_final(args.csv, model_path, args.algorithm or report["winner"])
    print(f"Modelo {bundle['algorithm']} treinado com {bundle['training_records']} registros e salvo em {model_path}.")
    print(f"SHA-256 das fontes: {bundle['source_sha256']}")


if __name__ == "__main__":
    main()
