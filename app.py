import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_data
from src.model import Autoencoder

st.set_page_config(page_title="Autoencoder", page_icon="🔮", layout="wide")
st.title("🔮 Autoencoder")
st.markdown("MLP-based autoencoder for dimensionality reduction and reconstruction.")

dataset_name = st.selectbox("Dataset", ["digits", "iris", "wine"])
X = load_data(dataset_name)

encoding_dim = st.slider("Encoding Dimension", 2, 32, 16)
if st.button("Train Autoencoder", type="primary"):
    with st.spinner("Training..."):
        ae = Autoencoder(encoding_dim=encoding_dim)
        ae.fit(X)
    X_recon = ae.reconstruct(X[:5])
    st.subheader("Original vs Reconstructed (first 5 samples)")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.caption(f"Sample {i+1}")
            if dataset_name == "digits":
                st.image(np.hstack([X[i].reshape(8,8), X_recon[i].reshape(8,8)]), width=150)
            else:
                st.write(f"Orig: {X[i][:4].round(2)}")
                st.write(f"Recon: {X_recon[i][:4].round(2)}")
