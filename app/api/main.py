import os
import sys

sys.path.append('/Users/shouryakulshrestha/portfolio/AB_InBev_assignment/chatbot')

import asyncio
import uvicorn
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.utils.rag_pipeline import RAGPipeline
from app.utils.retrieval import Retriever
from app.utils.utils import set_logger

set_logger(console=True)
logger = logging.getLogger(__name__)

app = FastAPI()

retriever = Retriever()
rag_pipeline = RAGPipeline()

class Query(BaseModel):
    question: str


@app.get('/')
@app.get('/home')
def read_home():
    """
     Home endpoint which can be used to test the availability of the application.
     """
    return {'message': 'System is healthy'}

@app.post("/ask")
async def ask(query: Query):
    logger.info(f"Received Query: {query}")
    response = rag_pipeline.generate_response(query.question)
    return {"response": response}
