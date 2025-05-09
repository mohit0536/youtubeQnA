FROM ubuntu:22.04

# 1. Install dependencies
RUN apt-get update && apt-get install -y \
    curl sudo unzip build-essential python3 python3-pip

# 2. Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Update PATH and use correct shell for ENV
ENV PATH="/root/.ollama/bin:${PATH}"
SHELL ["/bin/bash", "-c"]

# 4. Pull the model (this now works!)
# RUN ollama pull mistral

# 5. Copy app files
WORKDIR /app
COPY app.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 6. Install Python packages
RUN pip install streamlit transformers==4.51.3 torch==2.7.0 youtube-transcript-api==1.0.3 chromadb==1.0.8 sentence-transformers==4.1.0

# 7. Expose ports
EXPOSE 8501
EXPOSE 11434

# 8. Start Ollama + Streamlit
CMD ["./entrypoint.sh"]
