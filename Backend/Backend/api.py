import uvicorn
import textwrap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any

# Import RAG components from the system files
# NOTE: Ensure data_processor.py and rag_system.py are in the same directory.
from rag import setup_rag_chain, LLM_MODEL, EMBEDDING_MODEL
from data import get_retriever, DATA_PATH, CHROMA_PATH
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration and Initialization ---

# Global cache for chat history (in-memory for simple demonstration)
# Key: session_id (str), Value: chat_history (List[AIMessage | HumanMessage])
CHAT_HISTORY: Dict[str, List[Any]] = {} 

# Global variables for the RAG chain and retriever
rag_chain = None
retriever = None

# Initialize FastAPI app
app = FastAPI(
    title="Gemma Ollama RAG API",
    description="Conversational RAG system using local Gemma models via Ollama and LangChain.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add explicit OPTIONS handler for CORS preflight
@app.options("/chat")
async def chat_options():
    """Handle CORS preflight requests for the chat endpoint."""
    return {"message": "OK"}

# Pydantic models for request and response
class ChatRequest(BaseModel):
    """Defines the structure of the incoming chat request."""
    query: str
    session_id: str
    
class ChatResponse(BaseModel):
    """Defines the structure of the outgoing chat response."""
    answer: str
    session_id: str

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RAG API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check with system status."""
    return {
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "retriever_initialized": retriever is not None,
        "active_sessions": len(CHAT_HISTORY)
    }

@app.on_event("startup")
async def startup_event():
    """Initializes the RAG components when the FastAPI application starts."""
    global rag_chain, retriever
    print("--- API Initialization Started ---")
    print(f"LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print(f"Data Path: {DATA_PATH} | DB Path: {CHROMA_PATH}")
    
    # 1. Get the retriever (loads or builds the vector database)
    retriever = get_retriever()
    if not retriever:
        # If retriever fails to initialize, the API cannot function
        print("FATAL ERROR: Could not initialize RAG retriever. Exiting.")
        raise RuntimeError("RAG initialization failed. Check Ollama and data_processor.py logs.")

    # 2. Setup the full conversational RAG chain
    rag_chain = setup_rag_chain(retriever)
    print("--- API Initialization Complete. RAG chain ready. ---")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles conversational chat queries with RAG.
    Maintains history based on the provided session_id.
    """
    # Add request logging
    print(f"Received chat request: {request.query[:50]}... from session: {request.session_id}")
    
    if rag_chain is None:
        print("ERROR: RAG chain not initialized")
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "RAG system not initialized", 
                "message": "Please wait for the system to start up or check server logs"
            }
        )

    session_id = request.session_id
    query = request.query
    
    # Validate input
    if not query or not query.strip():
        raise HTTPException(
            status_code=400, 
            detail={"error": "Empty query", "message": "Query cannot be empty"}
        )
    
    # Get or initialize chat history for the session
    chat_history = CHAT_HISTORY.get(session_id, [])

    print(f"[{session_id}] User Query: {query}")

    try:
        # Invoke the RAG chain
        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })

        answer = result.get('answer', "I apologize, an error occurred while processing your request.")
        
        # Update chat history for the next turn
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        
        # Save updated history back to the global state
        CHAT_HISTORY[session_id] = chat_history

        print(f"[{session_id}] GemmaDoc Response: {textwrap.fill(answer, width=80)[:80]}...") # Log snippet
        
        return ChatResponse(
            answer=answer,
            session_id=session_id
        )

    except Exception as e:
        print(f"[{session_id}] Exception during RAG invocation: {e}")
        # Log and raise an internal server error with more details
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal RAG processing error", 
                "message": str(e),
                "session_id": session_id
            }
        )

# --- Execution Block ---

# To run the API: uvicorn api:app --reload
if __name__ == "__main__":
    print("Starting API server...")
    print("Access API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
