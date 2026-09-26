"""Interface Flask para classificar uma foto pelo modelo persistido."""

import base64
import io
import os

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from water_classifier.features import extract_histogram, focus_center, select_region
from water_classifier.model import load_model


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _parse_crop(form):
    keys = ("crop_left", "crop_top", "crop_right", "crop_bottom")
    try:
        return tuple(float(form[key]) for key in keys)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Selecione a região da água na foto antes de classificar.") from exc


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(MODEL_PATH=os.environ.get("WATER_MODEL_PATH", "models/water_image_interval.joblib"),
                      MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES)
    if test_config:
        app.config.update(test_config)
    bundle = load_model(app.config["MODEL_PATH"])
    requires_crop = bundle["region"] == "water_crop"

    @app.context_processor
    def model_info():
        return {
            "training_records": bundle["training_records"],
            "training_source": "fotos rotuladas" if requires_crop else "histogramas do CSV",
            "requires_crop": requires_crop,
        }

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("index.html", error="Arquivo maior que 8 MB."), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/")
    def predict():
        uploaded = request.files.get("photo")
        if uploaded is None or not uploaded.filename:
            return render_template("index.html", error="Escolha uma foto para enviar."), 400
        raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return render_template("index.html", error="Arquivo maior que 8 MB."), 413
        if not raw:
            return render_template("index.html", error="Arquivo vazio."), 400
        try:
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in ALLOWED_FORMATS:
                    raise ValueError("Use uma imagem JPEG, PNG ou WebP.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("A imagem excede 25 milhões de pixels.")
                source.load()
                crop = _parse_crop(request.form) if requires_crop else None
                region = select_region(source, crop)
                if requires_crop:
                    region = focus_center(region)
                region = region.convert("RGB")
                histogram = extract_histogram(region)
                region.thumbnail((700, 500))
                preview_buffer = io.BytesIO()
                region.save(preview_buffer, format="JPEG", quality=85)
            prediction = bundle["pipeline"].predict(histogram.reshape(1, -1))[0]
            if prediction not in ("limpo", "sujo"):
                raise RuntimeError("Classe prevista inválida.")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return render_template("index.html", error="A imagem está corrompida ou não é válida."), 400
        except ValueError as exc:
            return render_template("index.html", error=str(exc)), 400
        except Exception:
            app.logger.exception("Falha na inferência")
            return render_template("index.html", error="Não foi possível classificar esta foto."), 500
        result = "Limpo" if prediction == "limpo" else "Sujo"
        preview = f"data:image/jpeg;base64,{base64.b64encode(preview_buffer.getvalue()).decode('ascii')}"
        return render_template("index.html", result=result, preview=preview)

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
