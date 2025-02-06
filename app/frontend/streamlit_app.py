import logging

import requests
import streamlit as st

from app.utils.rag_pipeline import RAGPipeline
from app.utils.retrieval import Retriever
from app.utils.utils import set_logger

set_logger(console=True)
logger = logging.getLogger(__name__)

# Initialize the retriever and RAG pipeline
retriever = Retriever()
rag_pipeline = RAGPipeline()

st.title("RAG-based Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    data = {"question": prompt}

    # Generate response using the RAG pipeline
    response = requests.post("http://localhost:8000/ask", json=data)
    prediction = response.text
    response = rag_pipeline.generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
