import streamlit as st
import sys
import os

# Add the project root to the sys.path to allow imports from assistant/ and query_docs.py
# This assumes app.py is in the project root (i.e., ~/Projects/ai-assistant/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import your existing backend functions
from assistant.memory import load_history, add_exchange
from assistant.llm import query_ollama
from query_docs import query_vectorstore # Make sure this is correctly imported

# Define the maximum turns for conversational memory (same as in main.py)
MAX_SESSION_TURNS = 5

# This function is reused from your main.py to build the prompt for the LLM
def build_prompt(session_history, context_chunks, user_query):
    prompt_parts = []
    # Include a limited number of past turns for conversational memory
    for exchange in session_history[-MAX_SESSION_TURNS:]:
        # Streamlit displays messages, but the LLM prompt needs the User/Assistant labels
        prompt_parts.append(f"User: {exchange['prompt']}\nAssistant: {exchange['response']}")

    # Add RAG context if available
    context = "\n".join(context_chunks)
    if context: # Only add context if there are relevant chunks found
        prompt_parts.append(f"\nContext from documents:\n{context}")

    # Add the current user query, preparing for the assistant's response
    prompt_parts.append(f"\nUser: {user_query}\nAssistant:")
    return "\n\n".join(prompt_parts)

# --- Streamlit UI Setup ---

# Set basic page configuration for the web app
st.set_page_config(page_title="Local AI Assistant", layout="centered")
st.title("Local AI Assistant 🤖") # Main title of your application

# Initialize chat history in Streamlit's session state.
# Streamlit's session_state acts like a dictionary that persists across reruns of the script.
# We load the actual persistent history from disk only ONCE when the app starts or is reloaded.
if "messages" not in st.session_state:
    # Load the global conversation history from your external SSD
    # and format it for Streamlit's chat display (role, content)
    loaded_history = load_history()
    st.session_state.messages = []
    for entry in loaded_history:
        st.session_state.messages.append({"role": "user", "content": entry["prompt"]})
        st.session_state.messages.append({"role": "assistant", "content": entry["response"]})

# Display existing chat messages from the Streamlit session state
# This loop iterates through the history and renders each message
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # Use Streamlit's built-in chat message styling
        st.markdown(message["content"]) # Display the content (supports Markdown for formatting)

# --- Handle User Input ---

# st.chat_input creates an input box at the bottom of the chat interface
# The `:=` (walrus operator) assigns the input value to `user_query` and also evaluates it (True if not empty).
if user_query := st.chat_input("Ask your question..."):
    # Display user message immediately in the chat interface
    with st.chat_message("user"):
        st.markdown(user_query)

    # Add the user's message to Streamlit's internal session history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display a "Thinking..." spinner while the assistant processes
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Load the *full* global memory from disk just before querying the LLM
            # This ensures the LLM always has the latest full context from all sessions.
            global_history = load_history()

            # Get RAG-relevant document chunks based on the user's query
            st.info(f"Searching documents for: '{user_query}'") # Informative message
            context_chunks = query_vectorstore(user_query)
            st.info(f"Found {len(context_chunks)} relevant chunks.") # Informative message

            # Build the complete prompt for the LLM
            full_prompt = build_prompt(global_history, context_chunks, user_query)

            # Query the Ollama LLM
            model_response = query_ollama(full_prompt)

            # Display the LLM's response in the chat interface
            st.markdown(model_response)

            # Add the assistant's response to Streamlit's internal session history
            st.session_state.messages.append({"role": "assistant", "content": model_response})

            # Save the new conversation exchange to your persistent memory on the external SSD
            add_exchange(user_query, model_response)

            # Optional: For debugging, you can uncomment this to see the RAG context used
            st.expander("RAG Context Used:").code("\n---\n".join(context_chunks))
