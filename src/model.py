import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from loguru import logger


class Autoencoder:
    def __init__(self, encoding_dim: int = 8, hidden_layers: tuple = (32,), random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.encoder = MLPRegressor(
            hidden_layer_sizes=hidden_layers + (encoding_dim,),
            activation="relu", random_state=random_state, max_iter=500,
        )
        self.decoder = MLPRegressor(
            hidden_layer_sizes=hidden_layers[::-1],
            activation="relu", random_state=random_state, max_iter=500,
        )

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        encoded = self.encoder.fit_transform(X_scaled, X_scaled)
        self.decoder.fit(encoded, X_scaled)
        X_recon = self.decoder.predict(encoded)
        mse = mean_squared_error(X_scaled, X_recon)
        logger.info(f"Autoencoder: {X.shape[1]} → {self.encoding_dim} → {X.shape[1]}, MSE={mse:.4f}")
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled)

    def reconstruct(self, X):
        X_scaled = self.scaler.transform(X)
        encoded = self.encoder.predict(X_scaled)
        return self.decoder.predict(encoded)
