import typer
import sys
from loguru import logger

from .config import settings
from .data import load_data
from .model import Autoencoder
from .visualizer import AEVisualizer

app = typer.Typer(help="Autoencoder CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def train(dataset: str = typer.Option("digits", help="Dataset: digits, iris, wine")):
    logger.info(f"Training autoencoder on {dataset}...")
    X = load_data(dataset)
    ae = Autoencoder(encoding_dim=16)
    ae.fit(X)
    X_recon = ae.reconstruct(X[:5])
    AEVisualizer.plot_reconstruction(X[:5], X_recon, save_path=settings.plots_dir / "autoencoder.png")
    logger.success("Done!")


if __name__ == "__main__":
    app()
