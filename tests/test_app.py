import io

from PIL import Image

from app import create_app
from water_classifier.features import select_region


def picture_bytes(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (80, 80), (110, 125, 145)).save(buffer, format=format)
    return buffer.getvalue()


def example_water_bytes(path, crop):
    with Image.open(path) as image:
        region = select_region(image, crop)
    buffer = io.BytesIO()
    region.save(buffer, format="PNG")
    return buffer.getvalue()


def test_home_and_valid_upload():
    client = create_app({"TESTING": True}).test_client()
    assert client.get("/").status_code == 200
    for path, crop, expected in [
        ("ImagemCopos/limpo14.jpg", (0.43, 0.25, 0.71, 0.53), b"Limpo"),
        ("ImagemCopos/limpo_copo.jpg", (0.35, 0.34, 0.65, 0.68), b"Limpo"),
        ("ImagemCopos/sujo12.jpg", (0.39, 0.32, 0.65, 0.72), b"Sujo"),
    ]:
        response = client.post("/", data={
            "photo": (io.BytesIO(example_water_bytes(path, crop)), "copo.png"),
            "crop_left": "0", "crop_top": "0", "crop_right": "1", "crop_bottom": "1",
        }, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"Regi\xc3\xa3o analisada" in response.data
        assert b"<strong>" + expected + b"</strong>" in response.data


def test_unfamiliar_histogram_still_has_binary_result():
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/", data={
        "photo": (io.BytesIO(picture_bytes()), "copo.png"),
        "crop_left": "0", "crop_top": "0", "crop_right": "1", "crop_bottom": "1",
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"<strong>Limpo</strong>" in response.data or b"<strong>Sujo</strong>" in response.data
    assert b"fora do padr\xc3\xa3o" not in response.data


def test_reject_invalid_image_and_missing_selection():
    client = create_app({"TESTING": True}).test_client()
    bad = client.post("/", data={
        "photo": (io.BytesIO(b"not-an-image"), "fake.png"),
        "crop_left": "0", "crop_top": "0", "crop_right": "1", "crop_bottom": "1",
    }, content_type="multipart/form-data")
    assert bad.status_code == 400
    assert b"corrompida" in bad.data
    no_crop = client.post("/", data={"photo": (io.BytesIO(picture_bytes()), "copo.png")},
                          content_type="multipart/form-data")
    assert no_crop.status_code == 400
    assert b"Selecione a regi" in no_crop.data


def test_reject_oversize_file():
    client = create_app({"TESTING": True}).test_client()
    response = client.post("/", data={"photo": (io.BytesIO(b"x" * (8 * 1024 * 1024 + 1)), "large.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 413
