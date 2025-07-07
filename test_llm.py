# test_llm.py
from assistant.llm import query_ollama

print("Welcome to your local AI assistant (LLaMA 3 via Ollama). Type 'exit' to quit.\n")

while True:
    prompt = input("🧠 Prompt> ")
    if prompt.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    response = query_ollama(prompt)
    print("\n💬 Response:")
    print(response)
    print("-" * 50)
