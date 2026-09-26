import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

import app as webapp
from app import create_app
from water_classifier.features import extract_histogram


def picture_bytes(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (80, 80), (110, 125, 145)).save(buffer, format=format)
    return buffer.getvalue()


def test_home_and_valid_upload():
    client = create_app({"TESTING": True}).test_client()
    home = client.get("/")
    assert home.status_code == 200
    assert b'name="crop_left"' in home.data
    entries = json.loads(Path("water_regions.json").read_text(encoding="utf-8"))
    for entry in (entries[3], entries[4]):
        crop = entry["crop"]
        response = client.post("/", data={
            "photo": (io.BytesIO(Path(entry["path"]).read_bytes()), "copo.jpg"),
            "crop_left": str(crop[0]), "crop_top": str(crop[1]),
            "crop_right": str(crop[2]), "crop_bottom": str(crop[3]),
        }, content_type="multipart/form-data")
        assert response.status_code == 200
        assert "Região analisada".encode("utf-8") in response.data
        expected = b"Limpo" if entry["label"] == "limpo" else b"Sujo"
        assert b"<strong>" + expected + b"</strong>" in response.data


def test_csv_model_receives_full_rgb_histogram(monkeypatch):
    class Recorder:
        def predict(self, X):
            self.seen = X.copy()
            return np.array(["limpo"])

    recorder = Recorder()
    monkeypatch.setattr(webapp, "load_model", lambda _path: {
        "pipeline": recorder, "training_records": 50, "region": "full_image",
    })
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, :32] = (10, 20, 30)
    pixels[:, 32:] = (200, 150, 100)
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    client = create_app({"TESTING": True, "MODEL_PATH": "models/water_model.joblib"}).test_client()
    response = client.post("/", data={"photo": (io.BytesIO(buffer.getvalue()), "copo.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 200
    np.testing.assert_array_equal(recorder.seen[0], extract_histogram(image))


def test_unfamiliar_histogram_still_has_binary_result():
    client = create_app({"TESTING": True, "MODEL_PATH": "models/water_model.joblib"}).test_client()
    response = client.post("/", data={
        "photo": (io.BytesIO(picture_bytes()), "copo.png"),
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"<strong>Limpo</strong>" in response.data or b"<strong>Sujo</strong>" in response.data
    assert b"fora do padr\xc3\xa3o" not in response.data


def test_photo_model_requires_water_region():
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/", data={
        "photo": (io.BytesIO(Path("ImagemCopos/limpo_copo.jpg").read_bytes()), "copo.jpg"),
    }, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Selecione a região da água".encode("utf-8") in response.data


def test_muddy_water_with_broad_selection_is_dirty():
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/", data={
        "photo": (io.BytesIO(Path("ImagemCopos/sujo12.jpg").read_bytes()), "sujo12.jpg"),
        "crop_left": "0", "crop_top": "0", "crop_right": "1", "crop_bottom": "1",
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"<strong>Sujo</strong>" in response.data


def test_reject_invalid_image_and_missing_upload():
    client = create_app({"TESTING": True, "MODEL_PATH": "models/water_model.joblib"}).test_client()
    bad = client.post("/", data={
        "photo": (io.BytesIO(b"not-an-image"), "fake.png"),
    }, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert b"corrompida" in bad.data
    no_photo = client.post("/", data={}, content_type="multipart/form-data")
    assert no_photo.status_code == 400
    assert b"Escolha uma foto" in no_photo.data


def test_reject_oversize_file():
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/", data={"photo": (io.BytesIO(b"x" * (8 * 1024 * 1024 + 1)), "large.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 413
