from assistant.memory import load_history, add_exchange
from assistant.speech import listen_microphone, speak_text
from query_docs import query_vectorstore
from assistant.llm import query_ollama  # You created this earlier

MAX_SESSION_TURNS = 5    # How many past turns to include in LLM prompt


def build_prompt(session_history, context_chunks, user_query):
    # Build history part of the prompt
    prompt_parts = []
    for exchange in session_history[-MAX_SESSION_TURNS:]:
        prompt_parts.append(f"User: {exchange['prompt']}\nAssistant: {exchange['response']}")

    # Add doc-relevant RAG info
    context = "\n".join(context_chunks)
    if context: # Only add context if there are relevant chunks
        prompt_parts.append(f"\nContext from documents:\n{context}")

    # Add the current user query
    prompt_parts.append(f"\nUser: {user_query}\nAssistant:")

    return "\n\n".join(prompt_parts)


def main():
    print("Starting AI Assistant...")

    while True: # ADDED THIS LOOP
        print("\n📝 Enter your question (type 'exit' to quit):")
        user_query = input("You: ")

        if user_query.lower() == 'exit':
            print("Exiting assistant.")
            break # Exits the loop

        # Load global memory from disk
        global_history = load_history()
        print(f"Loaded {len(global_history)} turns from memory.") # ADDED for visibility

        # Get RAG-relevant doc chunks
        context_chunks = query_vectorstore(user_query)

        # Build prompt using session memory + context
        full_prompt = build_prompt(global_history, context_chunks, user_query)
        print("\n--- Prompt sent to LLM ---") # ADDED for visibility
        print(full_prompt) # ADDED for visibility
        print("--------------------------\n") # ADDED for visibility

        # Query Ollama LLM
        print("Querying Ollama LLM...") # ADDED for visibility
        model_response = query_ollama(full_prompt)
        print("Assistant:", model_response) # ADDED to print response

        # Speak it aloud
        # speak_text(model_response) # COMMENTED OUT

        # Save to global memory log
        add_exchange(user_query, model_response)
        print("Saved conversation to memory.") # ADDED for visibility

if __name__ == "__main__":
    main()
