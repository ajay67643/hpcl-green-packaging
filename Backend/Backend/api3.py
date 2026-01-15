import uvicorn
import ollama
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Pydantic model for the request body ---
# This ensures the incoming data has a 'chat' field.
class ChatRequest(BaseModel):
    chat: str

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Data-Centric Chat API",
    description="An API that uses Ollama's deepseekcoder-v2 to generate responses based on a chat string that includes a goal and JSON data.",
    version="1.1.0"
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Context at Startup ---
try:
    with open('Data/context.txt', 'r', encoding='utf-8') as f:
        CONTEXT_DATA = f.read()
except FileNotFoundError:
    print("WARNING: context.txt not found. The application will run without it.")
    CONTEXT_DATA = "No context provided."


# --- API Endpoint ---
@app.post("/chat/", summary="Generate a data-centric response from text")
async def generate_response(request: ChatRequest):
    """
    This endpoint takes a single string containing both a user's goal and the relevant JSON data.
    It then generates a direct, data-centric response from the deepseekcoder-v2 model.
    """
    # The entire input from the user is in request.chat
    user_input = request.chat

    # --- Construct the Prompt for the LLM ---
    system_prompt = (
        "You are a highly intelligent, data-centric AI assistant. "
        "Your primary function is to analyze the user's goal and provide a direct, factual response based "
        "exclusively on the provided 'CONTEXT' and the 'USER INPUT' which contains JSON data. "
        "Do not add any conversational fluff, greetings, apologies, or explanations of your process. "
        "Get straight to the point. Your response must be concise and directly address the user's request "
        "Impoertant give 3 suggestions like using more scrap would reduce thiss this factors by x% "
        "using the provided data points."
    )

    # The user prompt is now simpler, combining the goal and data.
    user_prompt = f"""
--- CONTEXT ---
{CONTEXT_DATA}

--- USER INPUT (contains goal and JSON data) ---
{user_input}

--- RESPONSE ---
"""

    # --- Interact with Ollama ---
    try:
        response = ollama.chat(
            model='gemma3:12b',
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': user_prompt,
                },
            ]
        )
        final_response = response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Ollama: {e}")

    # --- Return the Response ---
    return {"response": final_response}


# --- Run the Application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

