# --- FIX 1: Force pysqlite3 for ChromaDB on Railway ---
# This must be at the very top, before any other imports!
import sys
import os

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # Fallback to standard sqlite3 if locally (optional)

# --- Standard Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from rag_engine import CivicAssistRAG
import uvicorn

# Global variable to hold our AI engine
rag_engine = None

# 1. Define the Startup/Shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global rag_engine
    print("🚀 Loading Civic Assist RAG Engine...")
    try:
        rag_engine = CivicAssistRAG()
        print("✅ RAG Engine loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading RAG Engine: {e}")
    
    yield  # The application runs here
    
    # Shutdown: Cleanup (if needed)
    print("🛑 Shutting down Civic Assist...")

# 2. Initialize the App with the lifespan
app = FastAPI(title="Civic Assist API", lifespan=lifespan)

# 3. Allow Frontend to talk to Backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "Civic Assist API is running"}

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized")
    
    try:
        response = rag_engine.get_answer(request.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # --- FIX 2: Dynamic Port & Host for Cloud Deployment ---
    port = int(os.environ.get("PORT", 8000))
    # Host must be 0.0.0.0 for Railway, not 127.0.0.1
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
