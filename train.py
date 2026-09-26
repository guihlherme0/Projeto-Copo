"""Treina o modelo das fotos; opcionalmente, o modelo legado do CSV."""

import argparse

from water_classifier.model import train_final, train_from_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="res.csv")
    parser.add_argument("--images", default="water_regions.json")
    parser.add_argument("--model", default="models/water_model.joblib")
    parser.add_argument("--algorithm", help="Algoritmo legado do CSV; sem esta opção, usa as fotos.")
    args = parser.parse_args()
    bundle = (train_final(args.csv, args.model, args.algorithm) if args.algorithm
              else train_from_images(args.images, args.model))
    print(f"Modelo {bundle['algorithm']} treinado com {bundle['training_records']} registros e salvo em {args.model}.")
    print(f"SHA-256 das fontes: {bundle['source_sha256']}")


if __name__ == "__main__":
    main()
