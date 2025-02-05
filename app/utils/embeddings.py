import sys
sys.path.append('/Users/shouryakulshrestha/portfolio/AB_InBev_assignment/chatbot')
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import chromadb
import faiss
import numpy as np
from app.config.base_config import (CHROMA_BATCH_SIZE, CHROMA_DB_NAME,
                                CHROMA_DB_PATH, EMBEDDING_MODEL, FAISS_INDEX)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embeddings:

    def __init__(self, chunks: List[Dict[str, Any]]):
        """
        Initializes the Embeddings class with a Pretrained Embedding Model, FAISS Index and Chroma DB".
        Args:
            chunks (List[Dict[str, Any]]): A list of dictionaries with the content of the chunk and the metadata.
        """
        self.chunks = chunks
        self.model_name = EMBEDDING_MODEL
        self.chroma_db_path = CHROMA_DB_PATH
        self.faiss_index_path = FAISS_INDEX

        self.embedding_model = self._load_embedding_model()

        self.faiss_index = None
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name = CHROMA_DB_NAME)
        logger.info("Initialized Embeddings Class")
    

    def _load_embedding_model(self):
        """
        Loads the Sentence Transformer Model from the given model name.
        Args:
            None
        Returns:
            SentenceTransformer: The Sentence Transformer Model.
        """
        try:
            logger.info(f"Loading Embedding Model : {self.model_name}")
            return SentenceTransformer(self.model_name)
        
        except Exception as e:
            logger.error(f"Error Loading the Embedding Model : {e}")
            raise e
    

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generates the embedding for a given text phrase.
        Args:
            text (str): The text phrase to generate the embedding for.
        Returns:
            np.ndarray: The embedding for the given text phrase.
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        
        except Exception as e:
            logger.error(f"Error Generating Embedding for text : {e}")
            return None
    

    def generate_embeddings(self) -> List[np.ndarray]:
        """
        Generates embeddings for a list of chunks.
        Returns:
            List[np.ndarray]: A list of numpy arrays with the embeddings for each chunk.
        """
        try:
            if not self.chunks:
                logger.warning("No chunks provided for embedding generation.")
                return []

            logger.info("Generating embeddings for given chunks")
            chunk_texts = [chunk["content"] for chunk in self.chunks]
            embeddings = []

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(self._generate_embedding, text): text for text in chunk_texts}
                for future in as_completed(futures):
                    embedding = future.result()
                    if embedding is not None:
                        embeddings.append(embedding)

            logger.info(f"Generated embeddings for {len(embeddings)} chunks")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error Generating Embeddings : {e}")
            raise e
        
    
    def create_faiss_index(self, embeddings: List[np.ndarray]):
        """
        Creates a FAISS index for the embeddings.
        Args:
            embeddings (List[np.ndarray]): A list of numpy arrays with the embeddings for each chunk.
        Returns:
            None
        """
        try:
            if not embeddings:
                logger.warning("No embeddings available to create FAISS index.")
                return
            
            logger.info("Creating FAISS Index for the given embeddings")
            dimension = embeddings[0].shape[0]

            if any(embed.shape[0] != dimension for embed in embeddings):
                logger.error("Mismatch detected in embedding dimensions.")
                return
            
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(embeddings))
            logger.info(f"Created FAISS Index with {len(embeddings)} embeddings")
        
        except Exception as e:
            logger.error(f"Error Creating FAISS Index : {e}")
            raise e
    

    def save_faiss_index(self):
        """
        Saves the FAISS index to the given path.
        """
        try:
            if self.faiss_index is None:
                logger.warning("FAISS index not initialized, Nothing to save.")
                return
            
            logger.info(f"Saving FAISS Index to : {self.faiss_index_path}")
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            logger.info(f"FAISS Index saved successfully to : {self.faiss_index_path}")
        
        except Exception as e:
            logger.error(f"Error Saving FAISS Index : {e}")
            raise e
    

    def load_faiss_index(self):
        """
        Loads the FAISS index from the given path.
        """
        try:
            logger.info(f"Loading FAISS Index from : {self.faiss_index_path}")
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            logger.info(f"FAISS Index loaded successfully from : {self.faiss_index_path}")
        
        except Exception as e:
            logger.error(f"Error Loading FAISS Index : {e}")
            raise e
        
    
    def save_metadata_to_chroma(self, chunks: List[Dict[str, Any]]):
        """
        Saves the metadata and the embeddings to the Chroma DB.
        Args:
            chunks (List[Dict[str, Any]]): A list chunks with the 'content' and 'metadata'.
        Returns:
            None
        """
        try:
            if not self.chunks:
                logger.warning("No chunks available to save.")
                return
            
            logger.info("Saving metadata to Chroma DB")
            ids = [str(i) for i in range(len(chunks))]
            metadatas = [chunk['metadata'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]

            for i, metadata in enumerate(metadatas):
                metadata["faiss_index"] = i

            batch_size = CHROMA_BATCH_SIZE
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for i in range(0, len(chunks), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_metadatas = metadatas[i:i+batch_size]
                    batch_documents = documents[i:i+batch_size]
                    futures.append(executor.submit(
                        self.chroma_collection.add,
                        ids=batch_ids,
                        metadatas=batch_metadatas,
                        documents=batch_documents
                    ))
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error saving metadata to Chroma DB : {e}")

            logger.info(f"Saved metadata for {len(chunks)} chunks to Chroma DB")
        
        except Exception as e:
            logger.error(f"Error Saving metadata and embeddings to Chroma DB : {e}")
            raise e
    

    def process_chunks(self):
        """
        Processes the chunks by generating embeddings, creating FAISS Index, and saving metadata to Chroma DB.
        """
        try:
            if not self.chunks:
                logger.warning("No chunks provided")
                return 
            
            logger.info("Processing Chunks")
            embeddings = self.generate_embeddings()
            if embeddings:
                self.create_faiss_index(embeddings)
                self.save_faiss_index()
            self.save_metadata_to_chroma(self.chunks)
        
        except Exception as e:
            logger.error(f"Error Processing Chunks : {e}")
            raise e
            
        