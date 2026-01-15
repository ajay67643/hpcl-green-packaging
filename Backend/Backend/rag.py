import os
import textwrap
from pathlib import Path
from typing import List
import warnings # Added for suppressing warnings

# Suppress all warnings, which is common practice for silencing unnecessary 
# deprecation or library-internal messages when running a script.
warnings.filterwarnings("ignore")

# Import necessary components for data processing and vectorization
from data import get_retriever, DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL

# --- LangChain Imports ---
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration (LLM Specific) ---
LLM_MODEL = "command-r7b:latest"      # The large model for high-quality responses

# ----------------- Step 3: RAG and Conversation Chain Setup -----------------

def setup_rag_chain(retriever):
    """Sets up the history-aware conversational RAG chain."""

    # We reuse the LLM instance
    llm = ChatOllama(model=LLM_MODEL)

    # 1. Contextualize Question Prompt (Rephrases query based on history)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, generate a standalone "
        "question that can be used to retrieve relevant documents from the vector store. "
        "If the question is already standalone, just return that question. Do not answer the question."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create the history-aware retriever to incorporate chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2. Answer Question Chain Setup (The part that uses context to answer)
    qa_system_prompt = (
        # NOTE: Removed the instruction to cite sources.
        "You are an expert RAG system named 'LCA-AssistantBot'. Use the following retrieved context "
        "to answer the user's question. Be concise and professional. "
        "If the context does not contain the answer, answer on your own just based on your knowledge."
        "Answer in brief"
        "Don't write like from the context . dont talk about context in response"
        "\n\nContext: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Build the QA chain using create_stuff_documents_chain (combines context and prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3. Create the final RAG chain, combining retrieval and QA
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# ----------------- Step 4: Main Chat Loop -----------------

def main():
    """Main function to run the RAG chat system."""
    print("--- Ollama RAG System Initializing ---")
    print(f"LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    print(f"Data Path: {DATA_PATH} | DB Path: {CHROMA_PATH}")
    print("--------------------------------------")

    # Get the retriever (This handles building the database if it doesn't exist)
    retriever = get_retriever()
    if not retriever:
        print("Cannot run the system without a valid database. Exiting.")
        return

    # Now call setup_rag_chain with the correct implementation
    rag_chain = setup_rag_chain(retriever)
    chat_history: List[AIMessage | HumanMessage] = []

    print("\nSystem Ready. Type 'quit' or 'exit' to end the chat.")

    while True:
        try:
            query = input("\nYou: ")

            if query.lower() in ["quit", "exit"]:
                print("Exiting chat. Goodbye!")
                break

            if not query.strip():
                continue

            # Invoke the RAG chain
            print("GemmaDoc is thinking...")
            result = rag_chain.invoke({
                "input": query,
                "chat_history": chat_history
            })

            # Extract generated answer and source documents
            answer = result['answer']
            
            # NOTE: Removed code for extracting unique_sources and displaying references.
            
            # --- Display Response ---
            print("\n" + "="*50)
            print("GemmaDoc:")
            print(textwrap.fill(answer, width=80))
            print("="*50)

            # NOTE: Removed display of sources/referencing logic.
            
            print("="*50)

            # Update chat history for the next turn
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=answer))

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure Ollama is running and models are pulled.")


if __name__ == "__main__":
    # The main execution simply calls main(), letting get_retriever handle the initial database build.
    main()
