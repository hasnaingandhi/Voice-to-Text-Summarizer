# Voice-to-Text-Summarizer
Gen AI based Voice to text summarizer
Steps to run the code

1. Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
2. .\gen_ai\Scripts\activate.ps1

Requirements:
pip install requirements.txt
$Env:HF_TOKEN = "" ## Set your environment key here
streamlit run sample_check.py

