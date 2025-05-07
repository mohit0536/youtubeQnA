import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import pipeline
from urllib.parse import urlparse, parse_qs
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

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ----------------- Hugging Face Model Setup -----------------

# Initialize Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ask_hugging_face(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="🎥 YouTube Q&A with Hugging Face", layout="centered")
st.title("🎥 Ask Questions About Any YouTube Video")

video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    with st.spinner("Fetching transcript..."):
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
            with st.spinner("Thinking..."):
                results = collection.query(query_texts=[question], n_results=3)
                context = "\n".join(results["documents"][0])
                answer = ask_hugging_face(question, context)

            st.markdown("### 💬 Answer:")
            st.write(answer)
