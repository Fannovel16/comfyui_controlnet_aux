import abc
from copy import deepcopy

import cv2
import numpy as np
from sklearn.decomposition import PCA
from typing_extensions import Protocol


class TransformerInterface(Protocol):
    @abc.abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y=None):
        ...

    @abc.abstractmethod
    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        ...


class DomainAdapter:
    def __init__(self,
                 transformer: TransformerInterface,
                 ref_img: np.ndarray,
                 color_conversions=(None, None),
                 ):
        self.color_in, self.color_out = color_conversions
        self.source_transformer = deepcopy(transformer)
        self.target_transformer = transformer
        self.target_transformer.fit(self.flatten(ref_img))

    def to_colorspace(self, img):
        if self.color_in is None:
            return img
        return cv2.cvtColor(img, self.color_in)

    def from_colorspace(self, img):
        if self.color_out is None:
            return img
        return cv2.cvtColor(img.astype('uint8'), self.color_out)

    def flatten(self, img):
        img = self.to_colorspace(img)
        img = img.astype('float32') / 255.
        return img.reshape(-1, 3)

    def reconstruct(self, pixels, h, w):
        pixels = (np.clip(pixels, 0, 1) * 255).astype('uint8')
        return self.from_colorspace(pixels.reshape(h, w, 3))

    @staticmethod
    def _pca_sign(x):
        return np.sign(np.trace(x.components_))

    def __call__(self, image: np.ndarray):
        h, w, _ = image.shape
        pixels = self.flatten(image)
        self.source_transformer.fit(pixels)

        if self.target_transformer.__class__ in (PCA,):
            # dirty hack to make sure colors are not inverted
            if self._pca_sign(self.target_transformer) != self._pca_sign(self.source_transformer):
                self.target_transformer.components_ *= -1

        representation = self.source_transformer.transform(pixels)
        result = self.target_transformer.inverse_transform(representation)
        return self.reconstruct(result, h, w)
