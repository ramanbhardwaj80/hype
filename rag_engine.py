import os
import chromadb
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_dir: str = "./vector_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document embeddings"}
        )

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.page_content for doc in documents]
        
        # ChromaDB expects list of lists for embeddings if batching, 
        # but here we pass a list of numpy arrays converted to list
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )

    def query(self, query_embedding: np.ndarray, top_k: int = 5):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

class CivicAssistRAG:
    """Main class to handle the RAG flow"""
    def __init__(self):
        # Initialize components
        self.embedder = EmbeddingManager()
        self.vector_store = VectorStore()
        
        # Initialize LLM
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1
        )

    def get_answer(self, query: str) -> Dict[str, Any]:
        # 1. Embed Query
        query_emb = self.embedder.generate_embeddings([query])[0]
        
        # 2. Retrieve
        results = self.vector_store.query(query_emb, top_k=5)
        
        if not results['documents'] or not results['documents'][0]:
            return {"answer": "I couldn't find any relevant information in the documents.", "sources": []}

        # 3. Format Context
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        context = "\n\n".join(documents)
        
        # 4. Generate Answer
        prompt = f"""You are a helpful government scheme assistant. Use the context below to answer the query accurately.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "answer": response.content,
            "sources": [m.get('source_file', 'unknown') for m in metadatas]

        }
