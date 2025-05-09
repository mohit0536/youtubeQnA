import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
import torch
import random

# ----------------- Utility Functions -----------------

def get_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com') and query.path == '/watch':
        return parse_qs(query.query)['v'][0]
    return None

def fetch_transcript(video_url):
    try:
        video_id = get_video_id(video_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Failed to fetch transcript: {e}")
        return None

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# ----------------- Hugging Face Model Setup -----------------

# More powerful QA model
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    device=0 if torch.cuda.is_available() else -1,
)

def ask_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="🎥 YouTube Q&A with Hugging Face", layout="centered")
st.title("🎥 Ask Questions About Any YouTube Video")

video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    with st.spinner("Fetching transcript..."):
        transcript = fetch_transcript(video_url)

    if transcript:
        st.success("Transcript loaded successfully.")
        chunks = chunk_text(transcript, chunk_size=200, overlap=50)

        st.info(f"Transcript split into {len(chunks)} overlapping chunks.")

        # ChromaDB setup
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=True,
        )

        chroma_client = chromadb.Client()
        collection_name = f"video-{random.randint(1000, 9999)}"

        try:
            collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_fn)
        except:
            chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_fn)

        # Add transcript chunks to Chroma
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"chunk-{i}"])

        # Question input
        question = st.text_input("Ask a question based on the video:")

        if question:
            with st.spinner("Thinking..."):
                results = collection.query(query_texts=[question], n_results=3)
                top_context = "\n".join(results["documents"][0])
                answer = ask_question(question, top_context)

            st.markdown("### 💬 Answer:")
            st.write(answer)
