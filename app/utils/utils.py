# import sys
# sys.path.append("<Path_to_chatbot_dir>")

import logging
import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv
from transformers import pipeline

from app.config.base_config import HUGGINGFACE_LLM, LOG_DIR, LOG_FILE

load_dotenv("dev.env")


def set_logger(console: bool = True):
    """
    Sets up the logging configuration for the application.
    Args:
        console (bool): Whether to log to console or not.
    Returns:
        None
    """

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")

    file_handler = logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Logger Configured Successfully")


def load_gemini():
    """
    Loads the Gemini Model
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(gemini_model_name)

    return model


def load_huggingfacellm():
    """
    Loads the HuggingFace LLM
    """
    model = pipeline("text-generation", model=HUGGINGFACE_LLM)
    return model
