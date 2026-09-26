from evaluate_images import evaluate
from water_classifier.image_training import image_pipeline, load_image_dataset


def test_image_level_validation_and_final_predictions():
    result = evaluate("water_regions.json")
    assert result["confusion_matrix"] == [[4, 0], [0, 5]]
    features, labels, _, _ = load_image_dataset("water_regions.json")
    model = image_pipeline().fit(features, labels)
    assert list(model.predict(features)) == list(labels)
