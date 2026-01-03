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
st.set_page_config(page_title="Ghost Audio Lab", page_icon="👻")

st.title("Audio to MFCC Converter")
st.markdown("""
Extract the **MFCC fingerprint** of your audio and reconstruct it using the **Griffin-Lim algorithm**. 
The result is a 'ghostly' version containing only the vocal tract shapes and textures.
""")

# --- CACHED LOGIC ---


@st.cache_data
def generate_ghost_audio(audio_bytes, n_mfcc, n_iter):
    """
    Handles the heavy lifting: Loading, MFCC extraction, and Reconstruction.
    Caching ensures we don't re-run this unless settings change.
    """
    # Load audio from bytes
    y, sr = librosa.load(BytesIO(audio_bytes), sr=None)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Reconstruct audio
    y_ghost = librosa.feature.inverse.mfcc_to_audio(
        mfccs, sr=sr, n_iter=n_iter)

    # Create a buffer for the output file
    out_buffer = BytesIO()
    sf.write(out_buffer, y_ghost, sr, format='WAV')
    out_buffer.seek(0)

    return mfccs, sr, out_buffer


# --- SIDEBAR ---
st.sidebar.header("Algorithm Settings")
n_mfcc = st.sidebar.slider("MFCC Coefficients", 1, 80,
                           13, help="Higher = more detail, Lower = more 'ghostly'")
n_iter = st.sidebar.slider("Griffin-Lim Iterations",
                           16, 128, 32, help="More iterations = cleaner sound")

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader(
    "Upload a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file:
    # Read file into memory for the caching function
    file_bytes = uploaded_file.read()

    # Run processing
    mfccs, sr, ghost_buffer = generate_ghost_audio(file_bytes, n_mfcc, n_iter)

    # Display Players
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.audio(file_bytes)

    with col2:
        st.subheader("Ghost Reconstruction")
        st.audio(ghost_buffer)
        # The Download Button
        st.download_button(
            label="Download Ghost Audio",
            data=ghost_buffer,
            file_name="ghost_reconstruction.wav",
            mime="audio/wav"
        )

    # Display Visuals
    st.divider()
    st.subheader("Visual Fingerprint (MFCC)")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    plt.colorbar(img, ax=ax)
    st.pyplot(fig)

else:
    st.info("👋 Upload an audio file in the sidebar or main window to start!")
