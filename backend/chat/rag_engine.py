"""
RAG Engine for Chatting with Resumes.
Handles indexing of resume text and semantic search/QA.
Updated for google-genai SDK.
"""

import os
import logging
import json
import requests
from typing import List, Dict, Any, Generator, Tuple

# Vector Store & Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
# Use new langchain_huggingface if available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Modern LangChain imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

# New Google GenAI SDK
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Manages vector storage and retrieval for Resume RAG.
    """
    def __init__(self, collection_name: str = "resume_chat"):
        self.collection_name = collection_name
        self.embedding_model = None
        self.vector_store = None
        self.vector_store = None
        self.client = None
        self.all_chunks = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize embeddings and Qdrant in-memory."""
        try:
            logger.info("Initializing Embeddings (all-MiniLM-L6-v2)...")
            # Use local embeddings (fast)
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            logger.info("Initializing QdrantClient (In-Memory)...")
            # Initialize explicit client for in-memory storage
            self.client = QdrantClient(":memory:")
            
            # Ensure collection exists
            if not self.client.collection_exists(self.collection_name):
                 self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )

            logger.info("Initializing QdrantVectorStore...")
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name, 
                embedding=self.embedding_model,
            )
            logger.info("RAG Engine Initialized.")
        except Exception as e:
            logger.error(f"Failed to init RAG Engine: {e}")
            raise

    def ingest_text(self, text: str, metadata: Dict[str, Any] = None):
        """
        Chunk and index text. Clears previous collection first for simplicity (1 resume mode).
        """
        if not text: 
            return
            
        chunks = self.text_splitter.split_text(text)
        docs = [
            Document(page_content=chunk, metadata=metadata or {}) 
            for chunk in chunks
        ]
        
        self.vector_store.add_documents(docs)
        self.all_chunks = chunks  # Store for debug
        logger.info(f"Indexed {len(docs)} chunks.")

    def query(self, question: str, provider: str, model: str) -> Generator[Tuple[str, str, str], None, None]:
        """
        RAG Query: Retrieve context -> Stream Answer from LLM.
        Returns generator yielding (Answer Chunk, Context, Prompt).
        """
        # 1. Retrieve
        docs = self.vector_store.similarity_search(question, k=4)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""
        You are a helpful assistant answering questions about a candidate's resume.
        Use the available context to answer the question.
        If you don't know the answer relative to the context, say "I don't see that information in the resume."
        
        RESUME CONTEXT:
        {context_text}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        
        # 2. Generate (Stream)
        if provider == "gemini":
            stream = self._stream_gemini(prompt, model)
        else:
            stream = self._stream_ollama(prompt, model)
            
        for _, chunk in stream:
            yield chunk, context_text, prompt

    def _stream_ollama(self, prompt: str, model: str) -> Generator[Tuple[str, str], None, None]:
        payload = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': True,
            'options': {'temperature': 0.3}
        }
        try:
            resp = requests.post('http://localhost:11434/api/chat', json=payload, stream=True, timeout=60)
            resp.raise_for_status()
            
            buffer = ""
            for line in resp.iter_lines(decode_unicode=True):
                if not line: continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        buffer += content
                        yield "", buffer 
                except: continue
        except Exception as e:
            yield "", f"Error: {e}"

    def _stream_gemini(self, prompt: str, model: str) -> Generator[Tuple[str, str], None, None]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            yield "", "Error: GEMINI_API_KEY missing."
            return
            
        try:
            client = genai.Client(api_key=api_key)
            # generate_content_stream for streaming
            resp = client.models.generate_content_stream(
                model=model,
                contents=prompt
            )
            
            buffer = ""
            for chunk in resp:
                if chunk.text:
                    buffer += chunk.text
                    yield "", buffer
        except Exception as e:
            yield "", f"Error: {e}"
