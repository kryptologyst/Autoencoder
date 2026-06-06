import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Autoencoder
from src.data import load_data


class TestAutoencoder:
    def test_fit(self):
        X = load_data("iris")
        ae = Autoencoder(encoding_dim=4)
        ae.fit(X)
        encoded = ae.transform(X)
        assert encoded.shape == (len(X), 4)

    def test_reconstruct(self):
        X = load_data("iris")
        ae = Autoencoder(encoding_dim=4)
        ae.fit(X)
        X_recon = ae.reconstruct(X)
        assert X_recon.shape == X.shape
