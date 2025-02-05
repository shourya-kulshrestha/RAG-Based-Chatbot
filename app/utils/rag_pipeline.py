import os
import sys

sys.path.append('/Users/shouryakulshrestha/portfolio/AB_InBev_assignment/chatbot')

import logging
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from app.config.base_config import EMBEDDING_MODEL, LLM_MODEL
from app.utils.retrieval import Retriever
from app.utils.utils import load_gemini, load_huggingfacellm

logger = logging.getLogger(__name__)

class RAGPipeline:

    def __init__(self):
        """
        Initializes the RAG pipeline
        """
        self.retriever = Retriever()
        self.model_type = LLM_MODEL

        if self.model_type.lower() == 'gemini':
            self.model = load_gemini()
        elif self.model_type.lower() == 'huggingface':
            self.model = load_huggingfacellm()
        

        else:
            raise ValueError(f"Unsupported Model Type: {self.model_type}, Choose either 'gemini', or 'huggingface'")
        
        logger.info(f"Initialized RAG pipeline with {self.model_type} model")

    
    def generate_response(self, query: str) -> str:
        """
        Generates the response using the RAG Pipeline
        Args:
            query (str): User's query
        Returns:
            str : Generated response
        """
        try:
            logger.info(f"Generating response for the query: {query}")

            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            query_embedding = embedding_model.encode(query, convert_to_numpy=True)

            retrieved_chunks = self.retriever.basic_retrieval(query_embedding)
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks retrieved")
            
            context = "\n\n".join(chunk['content'] for chunk in retrieved_chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

            if self.model_type == 'gemini':
                response = self.model.generate_content(prompt)
                generated_text = response.text
            
            elif self.model_type == 'huggingface':
                generated_text = self.model(prompt, max_length=500, num_return_sequences=1)[0]["generated_text"]
            
            logger.info("Response generated successfully.")
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating response : {e}")
            return "An error occured while generating the response."
        

if __name__ == "__main__":
    rag = RAGPipeline()
    user_query = input("What do you want to ask?")
    response = rag.generate_response(user_query)
    print(response)