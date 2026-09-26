import hashlib
import json
from pathlib import Path

from water_classifier.model import load_model


def test_csv_model_matches_csv_selection():
    report = json.loads(Path("reports/csv_model_selection.json").read_text(encoding="utf-8"))
    bundle = load_model("models/water_model.joblib")
    assert set(report["algorithms"]) == {"svm_rbf", "floresta_aleatoria", "knn_3"}
    assert report["winner"] == bundle["algorithm"] == "svm_rbf"
    assert bundle["training_records"] == report["data_audit"]["records"] == 50
    assert bundle["region"] == "full_image"
    assert bundle["source_sha256"] == report["source_sha256"] == hashlib.sha256(Path("res.csv").read_bytes()).hexdigest()
