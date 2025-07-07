from assistant.memory import load_history, add_exchange

# Add a new test memory entry
add_exchange("What is the capital of France?", "The capital of France is Paris.")

# Load and display current memory
print("🧠 Current memory contents:\n")
for entry in load_history():
    print(f"User: {entry['prompt']}\nAssistant: {entry['response']}\n")
