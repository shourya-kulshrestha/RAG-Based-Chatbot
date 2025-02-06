# import sys
# sys.path.append("<Path_to_chatbot_dir>")

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import faiss
import numpy as np
from chromadb import PersistentClient
from sentence_transformers import CrossEncoder

from app.config.base_config import (
    CHROMA_DB_NAME,
    CHROMA_DB_PATH,
    CROSS_ENCODER_MODEL,
    FAISS_INDEX,
    TOP_K,
)

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self):
        """
        Initializes the Retriever class with the FAISS Index and Chroma DB.
        """

        self.chroma_db_path = CHROMA_DB_PATH
        self.faiss_index_path = FAISS_INDEX

        self.faiss_index = self._load_faiss_index()
        self.chroma_client = PersistentClient(path=self.chroma_db_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection(name=CHROMA_DB_NAME)

        # Debugging: Check stored metadata
        logger.info("Checking stored documents in ChromaDB:")
        try:
            print(self.chroma_collection.peek(5))  # Show 5 stored documents with metadata
        except Exception as e:
            logger.error(f"Error fetching stored metadata from ChromaDB: {e}")

        self.cross_encoder = None
        if CROSS_ENCODER_MODEL:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info(f"Initialized Cross Encoder Model : {CROSS_ENCODER_MODEL}")

        logger.info("Initialized Retriever Class")

    def _load_faiss_index(self):
        """
        Loads the FAISS Index from the given path.
        Args:
            None
        Returns:
            faiss.Index: Loaded FAISS Index.
        """
        try:
            if not os.path.exists(self.faiss_index_path):
                logger.warning("FAISS Index file not found. Initializing new FAISS index.")
                return faiss.IndexFlatL2(768)

            logger.info(f"Loading FAISS Index from : {self.faiss_index_path}")
            return faiss.read_index(self.faiss_index_path)

        except Exception as e:
            logger.error(f"Error Loading the FAISS Index : {e}")
            raise e

    def _retrieve_metadata(self, indices: np.ndarray, distances: np.ndarray):
        """
        Retrieves the metadata from Chroma for the given indices.
        Args:
            indices (np.ndarray): Array of indices to retrieve the metadata for.
            distances (np.ndarray): Array of distances corresponding to the given indices.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the metadata and the distance.
        """
        try:
            logger.info("Retrieving metadata for the given indices.")
            top_k_chunks = []

            with ThreadPoolExecutor() as executor:
                futures = []
                for pos, index in enumerate(indices):
                    if index != -1:
                        # Retrieve metadata using stored FAISS index
                        future = executor.submit(
                            self.chroma_collection.get, where={"faiss_index": {"$eq": int(index)}}
                        )
                        futures.append((future, pos))

                for future, pos in futures:
                    try:
                        result = future.result()
                        if result and "metadatas" in result and "documents" in result:
                            metadata = result["metadatas"][0]
                            document = result["documents"][0]
                            top_k_chunks.append(
                                {
                                    "content": document,
                                    "metadata": metadata,
                                    "distance": float(distances[pos]),
                                }
                            )

                    except Exception as e:
                        logger.error(f"Error retrieving metadata for index - {index} : {e}")

            logger.info(f"Retrieved the metadata for the given indices.")
            return top_k_chunks

        except Exception as e:
            logger.error(f"Error Retrieving Metadata : {e}")
            raise e

    def basic_retrieval(self, query_embedding: np.ndarray, k: int = TOP_K):
        """
        Performs the basic retrieval using FAISS index and Chroma.
        Args:
            query_embedding (np.ndarray): The embedding of the query.
            k (int): The number of top results to retrieve. Set to 5 by default.
        Returns:
            List[Dict[str, Any]]: A list of top-k retrieved chunks with metadata.
        """
        try:
            logger.info(f"Performing the base retrieval for the given query embedding.")

            ### Reshaping for FAISS
            query_embedding = np.array([query_embedding])

            ### Searching the FAISS Index
            distances, indices = self.faiss_index.search(query_embedding, k)

            ### Retrieving the metadata from Chroma for the top-k indices
            top_k_chunks = self._retrieve_metadata(indices[0], distances[0])

            logger.info(f"Retrieved the top-{k} chunks.")
            return top_k_chunks

        except Exception as e:
            logger.error(f"Error Performing Basic Retrieval : {e}")
            raise e

    def multi_query_retrieval(
        self, query_text: str, query_embeddings: List[np.ndarray], k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Performs the retireval using multiple query embeddings, (e.g. for query expansion)
        Args:
            query_text (str): The text of the query.
            query_embeddings (List[np.ndarray]): The list of query embeddings.
            k (int): The number of top results to retrieve. Set to 5 by default.
        Returns:
            List[Dict[str, Any]]: A list of top-k retrieved chunks with metadata.
        """
        try:
            logger.info(f"Performing the multi-query retrieval for the given query embeddings.")
            all_results = []

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.basic_retrieval, query_embedding, k): query_embedding
                    for query_embedding in query_embeddings
                }

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)

                    except Exception as e:
                        logger.error(f"Error performing multi-query retrieval : {e}")

            unique_results = self._deduplicate_and_rerank(query_text, all_results, k)

            logger.info(f"Retrieved the top-{k} chunks from the multi-query retrieval.")
            return unique_results

        except Exception as e:
            logger.error(f"Error Performing Multi-Query Retrieval : {e}")
            raise e

    def _deduplicate_and_rerank(
        self, query_text: str, results: List[Dict[str, Any]], k: int
    ) -> List[Dict[str, Any]]:
        """
        Deduplicates and Reranks the retrieved results.
        Args:
            query_text (str): The text of the query.
            results (List[Dict[str, Any]]): The list of retrieved chunks with metadata.
            k (int): The number of top results to retrieve. Set to 5 by default.
        Returns:
            List[Dict[str, Any]]: A list of top-k deduplicated and reranked chunks with metadata.
        """
        try:
            logger.info("Deduplicating and Reranking the results.")

            unique_results = {result["content"]: result for result in results}.values()

            if self.cross_encoder:
                unique_results = self._cross_encoder_reranking(query_text, unique_results)

            reranked_results = sorted(unique_results, key=lambda x: x["distance"])[:k]
            return reranked_results

        except Exception as e:
            logger.error(f"Error Deduplicating and Reranking the results : {e}")
            raise e

    def _cross_encoder_reranking(
        self, query_text: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reranks the candidates using the Cross Encoder Model.
        Args:
            query_text (str): The text of the query.
            query_embedding (np.ndarray): The embedding of the query.
            candidates (List[Dict[str, Any]]): The list of candidate chunks with metadata.
        Returns:
            List[Dict[str, Any]]: The reranked list of chunks with metadata.
        """
        try:
            logger.info("Reranking candidates using the Cross Encoder Model.")

            if not self.cross_encoder:
                logger.warning("Cross-Encoder Model Not loaded, Skipping Reranking.")
                return candidates

            candidate_texts = [candidate["content"] for candidate in candidates]
            query_candidates_pairs = [(query_text, candidate) for candidate in candidate_texts]
            scores = self.cross_encoder.predict(query_candidates_pairs)

            for candidate, score in zip(candidates, scores):
                candidate["score"] = score

            reranked_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

            logger.info("Reranked the Chunks using the Cross Encoder Model.")
            return reranked_candidates

        except Exception as e:
            logger.error(f"Error Performing Cross Encoder Reranking : {e}")
            raise e
