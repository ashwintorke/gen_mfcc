import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO

# --- PAGE SETUP ---
st.set_page_config(page_title="Ghost Audio Lab", page_icon="👻")

st.title("👻 Ghost Audio Lab")
st.markdown("""
Extract the **MFCC fingerprint** of your audio and reconstruct it using the **Griffin-Lim algorithm**. 
The result is a 'ghostly' version containing only the vocal tract shapes and textures.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Settings")
n_mfcc = st.sidebar.slider("Number of Coefficients",
                           min_value=1, max_value=40, value=13)
n_iter = st.sidebar.slider("Griffin-Lim Iterations",
                           min_value=16, max_value=100, value=32)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # 1. Load Audio
    # Librosa can read the BytesIO object directly
    y, sr = librosa.load(uploaded_file, sr=None)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Audio")
        st.audio(uploaded_file)

    # 2. Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 3. Reconstruct "Ghost" Audio
    with st.spinner("Generating Ghost Audio..."):
        # We use the inverse function we discussed
        y_ghost = librosa.feature.inverse.mfcc_to_audio(
            mfccs, sr=sr, n_iter=n_iter)

        # Save to a buffer so Streamlit can play/download it
        buffer = BytesIO()
        sf.write(buffer, y_ghost, sr, format='WAV')
        buffer.seek(0)

    with col2:
        st.subheader("Ghost Audio")
        st.audio(buffer, format='audio/wav')
        st.download_button("Download Ghost Audio", data=buffer,
                           file_name="ghost.wav", mime="audio/wav")

    # 4. Plotting
    st.divider()
    st.subheader(f"MFCC Spectrogram ({n_mfcc} Coefficients)")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    st.pyplot(fig)

else:
    st.info("Please upload an audio file to begin.")
