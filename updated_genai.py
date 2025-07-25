import os
os.environ.pop("HF_ENDPOINT", None)
del os.environ['HF_TOKEN'] 
os.environ['HF_TOKEN'] = ##
import streamlit as st
from audiorecorder import audiorecorder
import time
import re
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
#nltk.download('punkt') 
from bert_score import score as bert_score

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
      :root {
        --accent: #003366;      /* your primary brand color */
        --bg: #f0f2f6;          /* light grey background */
        --card: #ffffff;        /* card background */
        --text: #3E3E3E;        /* main text color */
      }
      /* Page background */
      .reportview-container, .stApp {
        background-color: var(--bg);
        color: var(--text);
      }
      /* Sticky header */
      header {
        background-color: var(--accent) !important;
        color: white !important;
      }
      /* Card styling */
      .card {
        background-color: var(--card);
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
      }
      .header-title {
        color: var(--accent);
        margin-bottom: 0.25rem;
      }
      .header-sub {
        color: var(--text);
        margin-top: 0;
        margin-bottom: 0.75rem;
      }
      /* Sidebar adjustments */
      .sidebar .sidebar-content {
        background-color: var(--accent);
        color: white;
      }
      .sidebar h1, .sidebar label {
        color: white;
      }
      /* Consistent button style */
      button, .stDownloadButton>button, .stButton>button {
        background-color: var(--accent);
        color: white;
      }
    </style>
    """,
    unsafe_allow_html=True,
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
    # init timers
    if 'trans_time' not in st.session_state:
        st.session_state['trans_time'] = None
    if 'sum_time' not in st.session_state:
        st.session_state['sum_time'] = None

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings", unsafe_allow_html=True)
        whisper_size = st.selectbox("Whisper Model Size", ["tiny","base","small","medium","large"], index=1)
        max_tokens = st.slider("Max Summary Tokens", 50, 500, 150)
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("HF_TOKEN not set.")
            return
        st.markdown("---")
        st.markdown("#### ℹ️ About")
        st.markdown("Audio‑to‑Text Summarizer using Whisper + LLM.")

    # Header Card
    st.markdown(
        "<div class='card'><h1 class='header-title'>GenAI Meeting Summarizer 🤖</h1>"
        "<p class='header-sub'>Record or upload audio, then transcribe & summarize.</p></div>",
        unsafe_allow_html=True,
    )

    # Audio + Timing + Upload
    c1, c2 = st.columns([1,1], gap="medium")
    with c1:
        st.markdown("<div class='card'><h2>1️⃣ Record Audio</h2></div>", unsafe_allow_html=True)
        recording = audiorecorder("Start 🚀","Stop ✋")

    with c2:
        st.markdown("<div class='card'><h2>⏱️ Timing</h2></div>", unsafe_allow_html=True)
        timer_slot = st.empty()

    # Convert recording to bytes
    audio_bytes = None
    if recording:
        buf = BytesIO()
        recording.export(buf, format="wav") if hasattr(recording, 'export') else buf.write(recording)
        audio_bytes = buf.getvalue()

    # If we have audio, process it
    if audio_bytes:
        # show playback
        st.markdown("<div class='card'><h2>2️⃣ Playback & Download</h2></div>", unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/wav")
        st.download_button("Download Audio", audio_bytes, file_name="input.wav")

        # write to temp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name

        # transcribe
        t0 = time.time()
        transcript = transcribe(path, whisper_size)
        t1 = time.time()
        st.session_state['trans_time'] = t1 - t0

        # summarize
        t2 = time.time()
        client = get_inference_client(hf_token)
        summary = summarize_via_hf(clean_text(transcript), client)
        t3 = time.time()
        st.session_state['sum_time'] = t3 - t2

        # update timers in place
        with timer_slot:
            tc, ts = st.columns(2)
            tc.metric("Transcription", f"{st.session_state['trans_time']:.2f}s")
            ts.metric("Summarization",   f"{st.session_state['sum_time']:.2f}s")

        # Results: Transcript + Reference on left; Summary + Metrics on right
        st.markdown("<div class='card'><h2>3️⃣ Results</h2></div>", unsafe_allow_html=True)
        left, right = st.columns([2,3], gap="large")

        with left:
            st.markdown("<div class='card'><h3>📜 Transcript</h3></div>", unsafe_allow_html=True)
            st.text_area("Transcript", transcript, height=280)
            st.download_button("Save Transcript", transcript, file_name="transcript.txt")
            st.markdown("<div class='card'><h3>🔖 Reference Summary</h3></div>", unsafe_allow_html=True)
            reference = st.text_area("Paste gold summary & press Ctrl+Enter", "", height=150, key="ref")

        with right:
            st.markdown("<div class='card'><h3>📝 Summary</h3></div>", unsafe_allow_html=True)
            st.text_area("Model Summary", summary, height=300)
            st.download_button("Save Summary", summary, file_name="summary.txt")

            if reference.strip():
                rouge_scorer_obj = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
                rouge_scores = rouge_scorer_obj.score(reference, summary)

                # 2) METEOR
                ref_tokens = reference.split()
                hyp_tokens = summary.split()
                meteor_val = meteor_score([ref_tokens], hyp_tokens)

                # BERT‐Score
                P, R, F1 = bert_score(
                    cands=[summary],
                    refs=[reference],
                    lang="en",
                    rescale_with_baseline=True
                )
                bertscore_f1 = F1[0].item()

                # display metrics
                st.markdown("<div class='panel'><h3>Evaluation Metrics</h3></div>", unsafe_allow_html=True)
                rcol, mcol, bcol = st.columns(3, gap="small")
                rcol.metric("ROUGE-1 F1",  f"{rouge_scores['rouge1'].fmeasure*100:.2f}%")
                mcol.metric("METEOR",       f"{meteor_val*100:.2f}%")
                bcol.metric("BERTScore F1", f"{bertscore_f1*100:.2f}%")

    else:
        st.markdown("<div class='card'><p>▶️ Record audio to begin.</p></div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        "<div class='card' style='text-align:center;'>"
        "<em>GEN AI Team – Sumit, Ran, Suchita, Hasnain</em>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    print("HF_ENDPOINT =", os.getenv("HF_ENDPOINT"))  # should show None
    print("Token =", os.getenv("HF_TOKEN"))
    main()
