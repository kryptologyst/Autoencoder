import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class AEVisualizer:
    @staticmethod
    def plot_reconstruction(original, reconstructed, n_examples=5, save_path=None):
        fig, axes = plt.subplots(2, n_examples, figsize=(12, 5))
        for i in range(n_examples):
            axes[0, i].imshow(original[i].reshape(8, 8), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed[i].reshape(8, 8), cmap="gray")
            axes[1, i].axis("off")
        axes[0, 0].set_ylabel("Original"); axes[1, 0].set_ylabel("Reconstructed")
        plt.suptitle("Autoencoder Reconstruction")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
