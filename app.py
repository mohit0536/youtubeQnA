import os
from yt_dlp import YoutubeDL
from webvtt import WebVTT
import streamlit as st
# __import__('pysqlite3') 
# import sys 
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
# from sentence_transformers import SentenceTransformer
# from chromadb.config import Settings
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import subprocess
from urllib.parse import urlparse, parse_qs
import random
# ----------------- Utility Functions -----------------
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer globally to avoid reloading on each call
@st.cache_resource
def load_model():
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

def ask_model(question, context):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question:

    Context: {context}

    Question: {question}
    Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com') and query.path == '/watch':
        return parse_qs(query.query)['v'][0]
    return None

def check_if_english(transcript_list):
    try:
        for transcript in transcript_list:
            if transcript.language_code.startswith("en"):  # or just == 'en'
                return True
        return False
    except (TranscriptsDisabled, NoTranscriptFound):
        return False


def fetch_transcript(video_url):
    try:
        # Step 1: Set video URL and video ID (for naming)
        video_id = get_video_id(video_url)

        # Step 2: yt-dlp options to fetch auto-generated subtitles only
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True,
            'outtmpl': f'{video_id}.%(ext)s',
        }

        # Step 3: Download subtitles using yt-dlp
        with YoutubeDL(ydl_opts) as ydl:
            print("Downloading subtitles...")
            ydl.download([video_url])

        # Step 4: Parse the .vtt subtitle file
        vtt_file = f"{video_id}.en.vtt"
        if os.path.exists(vtt_file):
            print("Parsing subtitle file...")
            transcript = []
            for caption in WebVTT().read(vtt_file):
                transcript.append(caption.text.strip())
            
            full_transcript = ' '.join(transcript)
            # print(full_transcript)
            return full_transcript
        else:
            st.error(f"Failed to fetch transcript: {e}")
            return None
    except:
        pass
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ask_ollama(question, context):
    prompt = f"""You are a helpful AI assistant. Use the video transcript context to answer.

Context:
{context}

Question: {question}
Answer:"""
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode().strip()

# ----------------- Streamlit UI -----------------

# st.set_page_config(page_title="🎥 YouTube Q&A with LLM", layout="centered")
st.title("🎥 Ask Questions About Any YouTube Video")

video_url = st.text_input("Enter YouTube video URL:")
variable = 1
if video_url and variable:
    with st.spinner("Fetching transcript..."):
        print("Done")
        text = fetch_transcript(video_url)

    if text:
        st.success("Transcript loaded successfully.")
        chunks = chunk_text(text)

        st.info(f"Transcript split into {len(chunks)} chunks.")

        # Initialize Chroma
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        chroma_client = chromadb.Client()
        try:
            chroma_client.reset()
        except chromadb.errors.AuthorizationError:
            pass
            # st.warning("Reset is disabled, proceeding without reset.")

        # collection = chroma_client.create_collection(name="video", embedding_function=embedding_fn)
        num = random.randint(1, 10)
        if "video"+str(num) not in chroma_client.list_collections():
            # chroma_client.delete_collection(name="video")
            try:
                collection = chroma_client.create_collection(name="video"+str(num), embedding_function=embedding_fn)
            except:
                chroma_client.delete_collection(name="video"+str(num))
                collection = chroma_client.create_collection(name="video"+str(num), embedding_function=embedding_fn)
        else:
            collection = chroma_client.get_collection(name="video"+str(num))

        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"chunk-{i}"])

        # Question input
        question = st.text_input("Ask a question based on the video:")
        if question:
            variable = 0
            with st.spinner("Thinking..."):
                results = collection.query(query_texts=[question], n_results=3)
                context = "\n".join(results["documents"][0])
                answer = ask_model(question, context)

            st.markdown("### 💬 Answer:")
            # print(answer)
            st.write(answer.split("Answer:", 1)[1].strip())