from evaluate_images import evaluate
from water_classifier.image_training import image_pipeline, load_image_dataset
from water_classifier.model import load_model


def test_image_level_validation_and_final_predictions():
    result = evaluate("water_regions.json")
    assert result["confusion_matrix"] == [[4, 0], [0, 5]]
    features, labels, _, _ = load_image_dataset("water_regions.json", include_focus=True)
    model = image_pipeline().fit(features, labels)
    assert list(model.predict(features)) == list(labels)
    saved = load_model("models/water_image_interval.joblib")
    assert saved["region"] == "water_crop"
    assert saved["training_records"] == 9
    assert saved["source_sha256"] == result["source_sha256"]
    assert list(saved["pipeline"].predict(features[1::2])) == list(labels[1::2])
