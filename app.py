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
st.set_page_config(page_title="GenMFCC", page_icon="🟠")

st.markdown("""
    <h1 style='font-family: "Lucida Console", Monaco, monospace; font-size: 32px; color: orange;'>
        GenMFCC
    </h1>
    """, unsafe_allow_html=True)

st.markdown("""
This tool allows you to extract the **MFCC fingerprint** of your audio and reconstruct it using the **Griffin-Lim algorithm**. 
It is intended to be a learning tool to explore how MFCCs capture the vocal tract characteristics of sound.
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


st.sidebar.header("Visualization Settings")

# Choose the color palette
cmap_option = st.sidebar.selectbox(
    "Select Color Palette",
    ["magma", "plasma", "cool", "winter", "viridis", "mako"]
)

# Contrast Sliders
# We use a range that typically fits MFCC decibel scales
v_min = st.sidebar.slider(
    "Darkness Cutoff (vmin)",
    min_value=-100,
    max_value=0,
    value=-40,
    help="Higher values make the background darker/cleaner."
)

v_max = st.sidebar.slider(
    "Neon Intensity (vmax)",
    min_value=0,
    max_value=100,
    value=20,
    help="Lower values make the 'glow' more intense."
)

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader(
    "Upload a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file:
    # Read file into memory for the caching function
    file_bytes = uploaded_file.read()

    # Run processing
    mfccs, sr, ghost_buffer = generate_ghost_audio(file_bytes, n_mfcc, n_iter)

    # Display Visuals
    st.divider()
    st.markdown("##### MFCC Spectrogram")

    # --- Plotting the MFCC ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Apply the Cyberpunk background to the plot figure
    fig.patch.set_facecolor('#141414ff')
    ax.set_facecolor('#c6c6c6ff')

    img = librosa.display.specshow(
        mfccs,
        x_axis='time',
        sr=sr,
        ax=ax,
        cmap=cmap_option,  # Linked to Selectbox
        vmin=v_min,       # Linked to Slider 1
        vmax=v_max        # Linked to Slider 2
    )

    # Styling the axes to match the Neon theme
    ax.tick_params(colors='#c6c6c6ff')
    ax.xaxis.label.set_color('#c6c6c6ff')
    ax.yaxis.label.set_color('#c6c6c6ff')

    # 1. Capture the colorbar object
    cbar = plt.colorbar(img, ax=ax)

    # Set the label text and color
    cbar.set_label("Magnitude (dB)", color='#c6c6c6ff', fontsize=10)

    # 2. Change the color of the tick labels (numbers)
    cbar.ax.tick_params(colors='#c6c6c6ff')

    # 3. Optional: Change the color of the colorbar outline/frame
    cbar.outline.set_edgecolor('#c6c6c6ff')
    st.pyplot(fig)

    # --- DOWNLOAD PLOT ---
    # Save the figure to a buffer
    plot_buffer = BytesIO()
    fig.savefig(plot_buffer, format='png',
                facecolor=fig.get_facecolor(), dpi=300)
    plot_buffer.seek(0)

    # Display Players
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Original")
        st.audio(file_bytes)
        # Spectrogram Download Button
        st.download_button(
            label="Download Spectrogram (PNG)",
            data=plot_buffer,
            file_name="genmfcc_spectrogram.png",
            mime="image/png"
        )

    with col2:
        st.markdown("##### Ghost Reconstruction")
        st.audio(ghost_buffer)
        # Audio Download Button
        st.download_button(
            label="Download Ghost Audio",
            data=ghost_buffer,
            file_name="ghost_reconstruction.wav",
            mime="audio/wav"
        )


else:
    st.info("Upload an audio file to start")
