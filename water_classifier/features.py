"""Mesmo histograma RGB na avaliação, treino e inferência."""

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class NormalizeHistograms(BaseEstimator, TransformerMixin):
    """Converte contagens por canal em frequências [0, 1]."""

    def fit(self, X, y=None):
        self._validate(X)
        self.n_features_in_ = 768
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = self._validate(X).reshape(-1, 3, 256)
        totals = values.sum(axis=2, keepdims=True)
        return (values / totals).reshape(-1, 768)

    @staticmethod
    def _validate(X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 768:
            raise ValueError("Esperadas 768 contagens RGB por registro.")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Histograma inválido.")
        if (values.reshape(-1, 3, 256).sum(axis=2) <= 0).any():
            raise ValueError("Histograma com canal vazio.")
        return values


class MeanRedGreen(BaseEstimator, TransformerMixin):
    """Diferença entre as médias R e G, em níveis de intensidade (0 a 255)."""

    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = 768
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 768:
            raise ValueError("Esperados 768 valores RGB normalizados.")
        levels = np.arange(256, dtype=np.float64)
        channels = values.reshape(-1, 3, 256)
        return ((channels[:, 0] - channels[:, 1]) @ levels).reshape(-1, 1)


def select_region(image: Image.Image, crop=None) -> Image.Image:
    """Aplica orientação EXIF e retorna exatamente a região selecionada."""
    image = ImageOps.exif_transpose(image)
    if crop is not None:
        if len(crop) != 4 or any(not 0 <= value <= 1 for value in crop):
            raise ValueError("Região selecionada inválida.")
        left, top, right, bottom = crop
        if right <= left or bottom <= top:
            raise ValueError("Região selecionada vazia.")
        box = (round(left * image.width), round(top * image.height),
               round(right * image.width), round(bottom * image.height))
        if box[2] - box[0] < 32 or box[3] - box[1] < 32:
            raise ValueError("Selecione uma área de pelo menos 32 × 32 pixels.")
        image = image.crop(box)
    return image


def focus_center(image: Image.Image) -> Image.Image:
    """Usa a metade central da seleção para reduzir bordas e fundo."""
    width, height = image.size
    return image.crop((round(width * 0.25), round(height * 0.25),
                       round(width * 0.75), round(height * 0.75)))


def extract_histogram(image: Image.Image, crop=None) -> np.ndarray:
    """Contagens inteiras em r0..255, g0..255, b0..255 da região selecionada."""
    image = select_region(image, crop)
    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    return np.concatenate([np.bincount(pixels[:, channel], minlength=256) for channel in range(3)]).astype(np.float64)
