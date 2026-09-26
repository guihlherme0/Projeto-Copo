import numpy as np
import pytest
from PIL import Image

from water_classifier.data import FEATURE_COLUMNS, load_dataset
from water_classifier.features import MeanRedGreen, NormalizeHistograms, extract_histogram


def test_extraction_order_scale_and_pipeline_contract():
    pixels = np.array([[[0, 10, 255], [0, 20, 30]],
                       [[255, 10, 30], [5, 10, 0]]], dtype=np.uint8)
    histogram = extract_histogram(Image.fromarray(pixels, mode="RGB"))
    assert histogram.shape == (len(FEATURE_COLUMNS),)
    assert histogram[0] == 2
    assert histogram[5] == 1
    assert histogram[255] == 1
    assert histogram[256 + 10] == 3
    assert histogram[512 + 30] == 2
    assert histogram[512 + 255] == 1
    assert np.array_equal(histogram.reshape(3, 256).sum(axis=1), [4, 4, 4])
    normalized = NormalizeHistograms().fit_transform(histogram.reshape(1, -1))
    assert normalized[0, 0] == 0.5
    assert np.allclose(normalized.reshape(1, 3, 256).sum(axis=2), 1)
    assert MeanRedGreen().fit_transform(normalized)[0, 0] == pytest.approx(65 - 12.5)


def test_crop_excludes_background():
    pixels = np.zeros((80, 80, 3), dtype=np.uint8)
    pixels[0:40, 0:40] = (12, 34, 56)
    histogram = extract_histogram(Image.fromarray(pixels, mode="RGB"), (0, 0, 0.5, 0.5))
    assert histogram[12] == 1600
    assert histogram[256 + 34] == 1600
    assert histogram[512 + 56] == 1600
    assert histogram[0] == 0


def test_dataset_follows_histogram_contract():
    features, labels = load_dataset("res.csv")
    assert features.shape == (50, 768)
    assert set(labels) == {"limpo", "sujo"}
    assert np.all(np.equal(features, np.floor(features)))
    normalized = NormalizeHistograms().fit_transform(features)
    assert np.allclose(normalized.reshape(-1, 3, 256).sum(axis=2), 1)


def test_reject_empty_crop():
    with pytest.raises(ValueError):
        extract_histogram(Image.new("RGB", (100, 100)), (0.1, 0.1, 0.1, 0.8))
