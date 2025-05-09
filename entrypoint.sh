#!/bin/bash
export OLLAMA_NO_CUDA=1
# 1. Start Ollama in the background
ollama start &

# 2. Give Ollama a moment to initialize
sleep 5

# 3. Preload the model so it's ready
ollama run mistral &

# 4. Wait again for model to load
sleep 10

# 5. Start the Streamlit app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0