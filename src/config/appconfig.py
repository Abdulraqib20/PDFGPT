import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Retrieve the environment variables (Local)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Retrieve the environment variables (Streamlit Cloud)
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# Define the model parameters
MODEL_NAME = "llama-3.1-70b-versatile"
TEMPERATURE = 0.2