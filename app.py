import streamlit as st
import os
# __import__('pysqlite3') 
# import sys 
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from yt_dlp import YoutubeDL
from webvtt import WebVTT
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import subprocess
import random
import requests

# Dialog Box on Load
st.markdown("""
    <style>
    .modal {
        display: block;
        position: fixed;
        z-index: 999;
        padding-top: 150px;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0,0,0,0.5);
    }

    .modal-content {
        background-color: #fefefe;
        margin: auto;
        padding: 20px;
        border: 1px solid #888;
        width: 50%;
        text-align: center;
        border-radius: 10px;
    }

    .close {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
    }

    .close:hover,
    .close:focus {
        color: black;
        text-decoration: none;
        cursor: pointer;
    }
    </style>

    <div id="customModal" class="modal">
      <div class="modal-content">
        <span class="close" onclick="document.getElementById('customModal').style.display='none'">&times;</span>
        <h3>Welcome! 👋</h3>
        <p>We’ve reached the monthly limit of the free tier for this month. We kindly invite you to watch our demo video. We’ll be back shortly. Thank you for your patience!</p>
        <a href="https://drive.google.com/file/d/1-875uZg6x6asRabQEuau2_xxaaLLP6j-/view?usp=sharing" target="_blank">Open This Link</a>
      </div>
    </div>
    <script>
  function hideDiv() {
    document.getElementById("customModal").style.display = "none";
  }
    </script>
""", unsafe_allow_html=True)

# ----------------- Model Loading -----------------
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

# tokenizer, model = load_model()

HF_API_TOKEN = os.environ['key']
HF_MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

def ask_model(question, context):
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(HF_MODEL_ENDPOINT, headers=headers, json=payload)
    
    if response.status_code == 200:
        print(response.json(),"response")
        return response.json()["answer"]
    else:
        print(response.text)
        return f"API request failed with status {response.status_code}: {response.text}"


# ----------------- Helper Functions -----------------
# def ask_model(question, context):
#     prompt = f"""You are a helpful assistant. Use the following context to answer the question:

# Context: {context}

# Question: {question}
# Answer:"""
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if "Answer:" in decoded_output:
#         answer = decoded_output.split("Answer:")[-1].strip()
#     else:
#         answer = decoded_output.strip()

#     return answer

def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com') and query.path == '/watch':
        return parse_qs(query.query).get('v', [None])[0]
    return None

def fetch_transcript(video_url):
    try:
        video_id = get_video_id(video_url)
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True,
            'outtmpl': f'{video_id}.%(ext)s',
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        vtt_file = f"{video_id}.en.vtt"
        if os.path.exists(vtt_file):
            transcript = [caption.text.strip() for caption in WebVTT().read(vtt_file)]
            return ' '.join(transcript)
        else:
            st.error("Subtitle file not found.")
            return None
    except Exception as e:
        st.error(f"Failed to fetch transcript: {e}")
        return None

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ----------------- Streamlit App -----------------
st.title("🎬 YouTube Video Q&A App")

# Initialize session state
for key in ["video_fetched", "video_input", "question_input", "transcript_text"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "video_fetched" else ""

# Section 1: Fetch YouTube Transcript
# Add YouTube-themed background via CSS

st.header("1. Fetch YouTube Transcript")
video_url = st.text_input("Enter YouTube Video URL:", value=st.session_state.video_input)
st.session_state.video_input = video_url  # persist the value

if st.button("Fetch Transcript"):
    if video_url:
        with st.spinner("Fetching video details..."):
            transcript = fetch_transcript(video_url)

        if transcript:
            st.success("Video fetched successfully!")
            chunks = chunk_text(transcript)

            # st.info(f"Transcript split into {len(chunks)} chunks.")

            # Setup ChromaDB
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            chroma_client = chromadb.Client()

            try:
                chroma_client.reset()
            except chromadb.errors.AuthorizationError:
                pass  # In hosted environments, reset may be disabled

            collection_name = f"video{random.randint(1, 10000)}"
            collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_fn)

            for i, chunk in enumerate(chunks):
                collection.add(documents=[chunk], ids=[f"chunk-{i}"])

            # Save in session state
            st.session_state.video_fetched = True
            st.session_state.transcript_text = collection
        else:
            st.error("Could not fetch transcript.")
    else:
        st.error("Please enter a valid YouTube video URL.")

# Section 2: Ask a Question
st.header("2. Ask a Question About the Video")

if not st.session_state.video_fetched:
    st.warning("Please fetch a YouTube video transcript first.")
else:
    question = st.text_input("Enter your question:", value=st.session_state.question_input)
    st.session_state.question_input = question

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Thinking..."):
                collection = st.session_state.transcript_text
                results = collection.query(query_texts=[question], n_results=3)
                context = "\n".join(results["documents"][0])
                answer = ask_model(question, context)

            st.markdown("### 💬 Answer:")
            st.success(answer)
        else:
            st.error("Please enter a question.")