# import sys
# sys.path.append("<Path_to_chatbot_dir>")

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config.base_config import CHUNK_SIZE, OVERLAP_SIZE

logger = logging.getLogger(__name__)


class Chunking:

    def __init__(self, data_dir: str):
        """
        Initializes the Chunking Class with the data directory to read the files from.
        Args:
            data_dir (str): The directory to read the markdown files from.
        Returns:
            None
        """
        self.data_dir = data_dir
        self.split_headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.split_headers)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE
        )
        logger.info(f"Initialized Chunking with the given data directory - {self.data_dir}")

    def read_file(self, filepath: str) -> Dict[str, Any]:
        """
        Reads the file from the given path, and returns the metadata and the content of the file.
        Args:
            filepath (str): The path of the file to read.
        Returns:
            Dict[str, Any]: A dictionary with the content and metadata of the file.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"Skipping empty file - {filepath}")
                return {"filepath": filepath, "content": "", "error": "File is empty"}

            logger.info(f"Successfully read file - {filepath}")
            return {"filepath": filepath, "content": content}

        except FileNotFoundError as e:
            logger.error(f"File not found - {filepath}: {e}")
            return {"filepath": filepath, "content": "", "error": str(e)}
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in file - {filepath}: {e}")
            return {"filepath": filepath, "content": "", "error": str(e)}
        except Exception as e:
            logger.error(f"Error reading file - {filepath}: {e}")
            return {"filepath": filepath, "content": "", "error": str(e)}

    def chunk_file(self, file: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Creates the chunks of text from the file while preserving the metadata.
        Returns a list of dictionaries with the content of the chunk and the metadata.
        Args:
            file (Dict[str, Any]): A dictionary with the file path and the content of the file.
        Returns:
            chunks (List[Dict[str, Any]]): A list of dictionaries with the content of the chunk and the metadata.
        """
        filepath = file["filepath"]
        content = file["content"]
        chunks = []

        if "error" in file:
            logger.warning(
                f"Skipping chunking for the file - {filepath} due to error - {file['error']}"
            )
            return chunks

        try:
            header_chunks = self.markdown_splitter.split_text(content)
            if not header_chunks:
                header_chunks = [content]

            for chunk in header_chunks:
                split_chunks = self.recursive_splitter.split_text(chunk.page_content)
                for split_chunk in split_chunks:
                    chunks.append(
                        {"content": split_chunk, "metadata": {"source": filepath, **chunk.metadata}}
                    )

            logger.info(f"Chunks created for file {filepath}: {len(chunks)}")
            return chunks

        except Exception as e:
            logger.error(f"Error processing file - {filepath}: {e}")
            return chunks

    def process_directory(self) -> List[Dict[str, Any]]:
        """
        Reads and chunks all the files recursively from the given directory and its subdirectories.
        Returns a list of dictionaries with the file path and the content of the file.
        Args:
            None
        Returns:
            markdown_files (List[Dict[str, Any]]): A list of dictionaries with the file path and the content of the file.
        """
        chunks = []

        try:
            files = [str(filepath) for filepath in Path(self.data_dir).rglob("*.md")]
            logger.info(f"Found {len(files)} markdown files in the directory - {self.data_dir}")

        except Exception as e:
            logger.error(f"Error reading files from the directory - {self.data_dir}: {e}")
            return chunks

        with ThreadPoolExecutor(max_workers=2) as executor:
            try:
                files_content = list(executor.map(self.read_file, files))

                chunks = list(executor.map(self.chunk_file, files_content))
                chunks = [chunk for sublist in chunks for chunk in sublist]  # Flatten the list
            except Exception as e:
                logger.error(
                    f"Error processing the files from the directory - {self.data_dir}: {e}"
                )

        logger.info(f"Total chunks created from the files - {len(chunks)}")
        return chunks
