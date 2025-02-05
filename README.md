# RAG-Based Chatbot

This repository contains the code for a RAG based chatbot which I've built using the following tech stack:<br>
Base Language - Python <br>
Framework - Langchain <br>
VectorDB - FAISS & Chroma <br>
Backend - FastAPI <br>
Frontend - Streamlit <br>

## Data
The data used for this project consists of Markdown files, containing information about the Ubuntu or Linux based OS. It is stored in the ```ubuntu-docs``` directory

## Pre-requisites and Setup
Clone the Repository from the Github, <br>
Create a conda environment by executing the command : ```conda create --name 'your_env_name'``` <br>
Activate the environment by executing the command : ```conda activate your_env_name``` <br>
Install the dependencies : ```pip install -r requirements.txt``` <br>
Process the data and store the embeddings : ```python app/scripts/preproces_data.py``` <br>
Start the FastAPI server : ```python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000``` <br>
Start the Streamlit App : ```streamlit run app/frontend/streamlit_app.py``` <br>


## Components and Working of the Chatbot 

### Chunking
For chunking the document, various strategies can be applied, such as fixed size chunking, recurisive chunking, paragraph based chunking, topic based chunking and so on. While chunking, we need to play around a little with the data to find the technique that works best for our usecase. We need to ensure, that the chunks are of similar length, as large variations in the length of the chunks affects the performance of the Retriever. <br>
I have used Markdown based splitting and further Recursive Splitting to generate the Chunks, and have implemented it using Langchain's Splitters

### Embedding Generation
For generating the embeddings, I have used the ```'all-MiniLM-L6-v2'``` model from the HuggingFace. Other embedding models present in the Huggingface such as sentence-transformers/all-mpnet-base-v2, T5-small, T5-large, or OpenAI embedding models like ada-002, v3-small, v3-large can be explored considering cost, dimensionality, and performance.

### Storing the Embeddings
For storing the embeddings, there are multiple Vector Databases available, such as Chroma, FAISS, Pinecone, Milvus, etc. each having it's own benefits. <br>
I used a combination of Chroma and FAISS, where I used FAISS to store and retrieve the indices of the embeddings as it provides custom controls over the algorithms used for indexing(Flat, IVF, HNSW, etc.) and searching(L2, Cosine, BM25, etc.) {which Chroma does not provides} <br>
I used Chroma to store the metadata of the Chunks and linked it to FAISS for the getting the indices, as Chroma provides the option of storing the metadata, and can be easily integrated with langchain {A feature where FAISS lags}.

### Indexing and Searching Vector DB
There are various algorithms designed for indexing such as Flat (Exact distance based), IVF(Clustering based), HNSW (Graph based), etc.. Since the dataset was quite small (less than 1000 chunks), I went with the FlatL2 indexing method, focusing on the performance only.

Distance metrics while retrieving the top K similar chunks can be L2 (Euclidean Distance), Cosine Similarity, or Dot Product between the vectors. I used L2 as the distance metric used while retrieval.


### Retrieval
To ensure the quality of the chunks retrieved we can leverage a combination of multiple techniques, like Multi Query Formation, Reranking using other embedding models. I used the Cross-Encoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2' for reranking after retrieving the top chunks.

### LLM 
I have provided the option of using either Google's Gemini or HuggingFace GPT 2. Other models from Huggingface like flan-t5 or OpenAI's models can be used to generate the final response as well.
For using the Google's gemini, create a dev.env in the root directory, and place your credentials there.

## Future Scope of Work
This application is built currently at a POC level, and lags scaling capabilities, and optimized algorithms for indexing and searching of embeddings. <br>
The vector db can be switched to Milvus, as it is Production ready, and is the most efficient in retrieval out of the options available. <br>
Algorithms used for indexing and searching can be changed as the data scales up. <br>
Redis can be used to store the conversation memory, for a better user interaction with the chatbot.
