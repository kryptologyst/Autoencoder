import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from loguru import logger

DATASETS = {"iris": load_iris, "wine": load_wine, "digits": load_digits}

def load_data(dataset_name="digits"):
    loader = DATASETS.get(dataset_name, load_digits)
    data = loader()
    X = data.data.astype(np.float64)
    logger.info(f"{dataset_name}: {X.shape[0]} samples, {X.shape[1]} dims")
    return X
