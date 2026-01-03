# Standard library imports
from io import BytesIO

# Third-party imports
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# --- APP CONFIG ---
st.set_page_config(page_title="Ghost Audio", icon="👻", layout="wide")

st.title("Ghost Audio & MFCC Visualizer")
st.markdown(
    "Upload audio to see its MFCC 'fingerprint' and hear what the AI hears.")
