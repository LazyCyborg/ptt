# PTT (Python Transcription Tool)

A small app for transcribing interview text using torch-audio and OpenAI Whisper through the Transformers library

## Create a virtual environment (very recommended)
For more info se: https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

```bash
conda create -n transcribe python=3.10
conda activate transcribe

```
## Clone the repo and install the packages

```bash
git clone https://github.com/LazyCyborg/ptt.git
cd ptt
pip install -r requirements.txt

```
## Activate your virtual environment (if not already activated):
Open the terminal and re-activate your environmnet 
You can se which environmnet you are in, in the terminal by looking at the prefix before your username

Example:
(transcribe) usernamee@xxx ~ % 

```bash
conda activate transcribe

```

## Run the app

```bash
cd ptt
streamlit run app.py

```
The app will open in your wwebbrowser

## Closing the app
To close the app just press **Ctr + C** in the command line in the terminal
Just close the tab in your browser

**Note that the first time one runs the app it will take some time since OpenAI's Whisper needs to be downloaded to your machine**
