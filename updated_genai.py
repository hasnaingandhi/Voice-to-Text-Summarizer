import os
os.environ.pop("HF_ENDPOINT", None)
del os.environ['HF_TOKEN'] 
os.environ['HF_TOKEN'] = ## Add the HF token
import streamlit as st
from audiorecorder import audiorecorder
import time
import re
from rouge_score import rouge_scorer

import tempfile
import whisper
from pydub import AudioSegment
from io import BytesIO
#from huggingface_hub import InferenceApi
from huggingface_hub import InferenceClient


# --- Streamlit Page Setup ---
st.set_page_config(page_title="GenAI Meeting Summarizer", layout="wide")
st.markdown(
    """
    <style>
      .reportview-container {background-color: #f0f2f6;}
      .header-title {font-size: 2.8rem; font-weight: 700; color: #3E3E3E;}
      .header-sub {font-size: 1.2rem; color: #5A5A5A;}
      .card {background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05); margin-bottom: 1rem;}
      .sidebar .sidebar-content {background-color: #003366; color: white;}
      .sidebar h1, .sidebar label {color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions ---
@st.cache_resource
def load_whisper_model(size: str = "base"):
    return whisper.load_model(size)

@st.cache_resource
def get_inference_client(token: str):
    return InferenceClient(
        model="facebook/bart-large-cnn",
        token=token
    )


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.([A-Za-z])', r'. \1', text)
    text = re.sub(r'\s([,.])', r'\1', text)
    return text.strip()

def ensure_mono_wav(path: str) -> str:
    audio = AudioSegment.from_file(path)
    if audio.channels > 1:
        mono_path = path.replace(".wav", "_mono.wav")
        audio.set_channels(1).export(mono_path, format="wav")
        return mono_path
    return path

@st.cache_resource
def transcribe(path: str, model_size: str) -> str:
    model = load_whisper_model(model_size)
    mono = ensure_mono_wav(path)
    result = model.transcribe(mono, language="en")
    return result.get("text", "")



def summarize_via_hf(transcript: str, hf_client) -> str:
    output = hf_client.summarization(transcript)
    return output.summary_text.strip()

def summarize(text: str, client, max_tokens: int) -> str:
    prompt = f"Summarize the following meeting transcript in bullet points:\n{text}"
    output = client.summarization(prompt, max_new_tokens=max_tokens)
    return output.generated_text.strip()
# --- UI ---
def main():
    # Sidebar
     # --- 1) INIT timers in session state ---
    if 'transcription_time' not in st.session_state:
        st.session_state['transcription_time'] = None
    if 'summary_time' not in st.session_state:
        st.session_state['summary_time'] = None
    with st.sidebar:
        st.markdown("<h1 class='header-title'>Settings</h1>", unsafe_allow_html=True)
        whisper_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=1)
        max_summary_tokens = st.slider("Max Summary Tokens", 50, 500, 150)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("HF_TOKEN not set. Please configure your environment variable.")
            return
        st.markdown("---")
        st.markdown("## About")
        st.markdown("GenAI Audio-to-Text Summarizer using Whisper + LLaMA model 2.7.")

    # Header
    st.markdown(
        "<div class='card'><h1 class='header-title'>GenAI Meeting Summarizer 🤖</h1>"
        "<p class='header-sub'>Upload or record audio, then get a transcript and concise summary.</p></div>",
        unsafe_allow_html=True
    )

    # Audio input section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h2>1️⃣ Record Audio</h2></div>", unsafe_allow_html=True)
        recording = audiorecorder("Start 🚀", "Stop ✋")

    with col2:
        # —— Timer card in place of upload —— 
        st.markdown("<div class='card'><h2>⏱️ Timing</h2></div>", unsafe_allow_html=True)
        timing_slot = st.empty()
        # if st.session_state['transcription_time'] is not None:
        #     tcol1, tcol2 = st.columns(2)
        #     tcol1.metric("Transcription", f"{st.session_state['transcription_time']:.2f}s")
        #     tcol2.metric("Summarization", f"{st.session_state['summary_time']:.2f}s")
        # else:
        #     st.write("▶️ Upload or record audio to see timing here.")

        # —— Now the uploader below —— 
        # st.markdown("<div class='card'><h2>📁 Upload Audio</h2></div>", unsafe_allow_html=True)
        # uploaded = st.file_uploader("Select WAV/MP3/M4A file:", type=["wav","mp3","m4a"])
    
    audio_bytes = None
    # if uploaded:
    #     try:
    #         raw_bytes = uploaded.read()
    #         ext = uploaded.name.split('.')[-1].lower()
    #         audio = AudioSegment.from_file(BytesIO(raw_bytes), format=ext)
    #         buf = BytesIO()
    #         audio.export(buf, format="wav")
    #         audio_bytes = buf.getvalue()
    #     except Exception as e:
    #         st.error(f"Upload error: {e}")
    if recording:
        try:
            buf = BytesIO()
            if hasattr(recording, 'export'):
                recording.export(buf, format="wav")
            else:
                buf.write(recording)
            audio_bytes = buf.getvalue()
        except Exception as e:
            st.error(f"Recording error: {e}")

    if audio_bytes:
        st.markdown("<div class='card'><h2>🎧 Playback & Download</h2></div>", unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/wav")
        st.download_button("Download Audio", audio_bytes, file_name="input.wav")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        with st.spinner("🔊 Transcribing..."):
            t0 = time.time()
            transcript = transcribe(audio_path, whisper_size)
            t1 = time.time()
            st.session_state['transcription_time'] = t1 - t0
            transcription_time = t1-t0

        with st.spinner("📝 Summarizing..."):
            t2 = time.time()
            client = get_inference_client(hf_token)
            summary = summarize_via_hf(clean_text(transcript), client)
            t3 = time.time()
            st.session_state['summary_time'] = t3 - t2
            summary_time = t3-t2

            #client = get_inference_client(hf_token)
            #api = get_inference_api(hf_token)
            #summary = summarize_via_hf(clean_text(transcript), client, max_summary_tokens)
            #summary = summarize_via_hf(clean_text(transcript), api, max_summary_tokens)
            #summary = summarize_via_hf(clean_text(transcript), client)

            #client = get_inference_client(hf_token)
            #summary = summarize(clean_text(transcript), client, max_summary_tokens)
        with timing_slot:
            tcol1, tcol2 = st.columns(2)
            tcol1.metric("Transcription", f"{transcription_time:.2f}s")
            tcol2.metric("Summarization",  f"{summary_time:.2f}s")
        # Display results
        st.markdown("<div class='card'><h2>Results</h2></div>", unsafe_allow_html=True)
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            st.markdown("<div class='card'><h3>📜 Transcript</h3></div>", unsafe_allow_html=True)
            st.text_area("Transcript", transcript, height=300)
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")

            st.markdown("<div class='card'><h3>🔖 Reference Summary</h3></div>", unsafe_allow_html=True)
            reference = st.text_area(
                "Paste your gold‑standard summary here and press Ctrl + Enter to evaluate",
                height=150,
                key="reference_input"
            )

        with right_col:
            st.markdown("<div class='card'><h3>📝 Summary</h3></div>", unsafe_allow_html=True)
            st.text_area("Model Summary", summary, height=300)
            st.download_button("Download Summary", summary, file_name="summary.txt")

            # only when the user has actually pasted something
            if reference and reference.strip():
                # compute ROUGE scores
                scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
                scores = scorer.score(reference, summary)

                # show metrics
                st.markdown("<div class='card'><h3>📊 Evaluation Metrics</h3></div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("ROUGE‑1 F1", f"{scores['rouge1'].fmeasure*100:.2f}%")
                m2.metric("ROUGE‑2 F1", f"{scores['rouge2'].fmeasure*100:.2f}%")
                m3.metric("ROUGE‑L F1", f"{scores['rougeL'].fmeasure*100:.2f}%")
    else:
        st.markdown("<div class='card'><p>Upload or record audio to begin.</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><em>GEN AI Team - Sumit, Ran, Suchita, Hasnain</em></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    print("HF_ENDPOINT =", os.getenv("HF_ENDPOINT"))  # should show None
    print("Token =", os.getenv("HF_TOKEN"))
    main()
