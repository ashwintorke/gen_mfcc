# Welcome to gen_mfcc

An interactive web application built with **Streamlit** and **Librosa** that explores the boundaries of audio feature extraction and reconstruction.

## Live Demo
https://genmfcc.streamlit.app/

## What is "Ghost Audio"?
This project uses **Mel-Frequency Cepstral Coefficients (MFCCs)** to create a spectral "fingerprint" of an audio file. MFCCs represent the broad spectral shape of a sound, effectively capturing the "texture" or "timbre" while discarding exact pitch and phase.

By using the **Griffin-Lim algorithm** to reconstruct the audio from these coefficients, we create a "Ghost" version of the original track. It sounds like a robotic, whispering version of the source because the computer is "guessing" the missing information based only on the spectral shapes.


## Features
- **Real-time Processing:** Upload any `.wav` or `.mp3` file.
- **Dynamic MFCC Control:** Adjust the number of coefficients to see how it affects audio clarity.
- **Iterative Reconstruction:** Control the Griffin-Lim algorithm to improve reconstruction quality.
- **Visual Mapping:** View high-resolution MFCC spectrograms of your audio.
- **Downloadable Results:** Save your "Ghost" reconstructions directly to your device.

## Tech Stack
- **Python** (Core Logic)
- **Streamlit** (Web Interface & Deployment)
- **Librosa** (Digital Signal Processing)
- **Matplotlib** (Data Visualization)
- **NumPy** (Mathematical Operations)

## Local Setup

# 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/ashwintorke/gen_mfcc.git
cd ghost-audio-app

# 2. Create a virtual environment
# For MacOS/Linux:
python3 -m venv venv
source venv/bin/activate

# For Windows:
# python -m venv venv
# .\venv\Scripts\activate

# 3. Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py

Note for Linux Users: If you encounter errors related to sndfile, install the underlying C-library using:

sudo apt-get install libsndfile1