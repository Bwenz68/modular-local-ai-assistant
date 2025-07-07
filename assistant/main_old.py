from assistant.memory import load_history, add_exchange
from assistant.speech import listen_microphone, speak_text
from query_docs import query_vectorstore
from assistant.llm import query_ollama  # You created this earlier

MAX_SESSION_TURNS = 5  # How many past turns to include in LLM prompt


def build_prompt(session_history, context_chunks, user_query):
    # Build history part of the prompt
    prompt_parts = []
    for exchange in session_history[-MAX_SESSION_TURNS:]:
        prompt_parts.append(f"User: {exchange['prompt']}\nAssistant: {exchange['response']}")

    # Add doc-relevant RAG info
    context = "\n".join(context_chunks)
    prompt_parts.append(f"\nContext from documents:\n{context}")

    # Add the current user query
    prompt_parts.append(f"\nUser: {user_query}\nAssistant:")

    return "\n\n".join(prompt_parts)


def main():
   # print("🎙️ Speak your question...")
   # user_query = listen_microphone()
    print("📝 Enter your question (type 'exit' to quit):")
    user_query = input("You: ")
    if user_query.lower() == 'exit':
        print("Exiting assistant.")
        return # Added to allow clean exit for text input

    # Load global memory from disk
    global_history = load_history()

    # Get RAG-relevant doc chunks
    context_chunks = query_vectorstore(user_query)

    # Build prompt using session memory + context
    full_prompt = build_prompt(global_history, context_chunks, user_query)

    # Query Ollama LLM
    model_response = query_ollama(full_prompt)

    # Speak it aloud
    #speak_text(model_response)

    # Save to global memory log
    add_exchange(user_query, model_response)


if __name__ == "__main__":
    main()
