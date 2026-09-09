import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def total_tokens(messages, model="gpt-4o-mini"):
    # Rough approximation of token count for a list of messages
    tokens = 0
    for msg in messages:
        tokens += count_tokens(msg["content"], model)
        # Add 4 tokens per message for role/content formatting overhead
        tokens += 4
    # Add 3 tokens for base formatting
    tokens += 3
    return tokens

def trim_history(messages, budget=100):
    """
    Trims the history to stay under the budget, 
    always preserving the first (system) message.
    """
    # Keep trimming while over budget and we have older messages to remove
    # len(messages) > 2 ensures we don't remove the system message or the very last user message
    while total_tokens(messages) > budget and len(messages) > 2:
        print(f"[TRIM] Removing oldest message: {messages[1]['content'][:30]}...")
        messages.pop(1)

def simulate_chat():
    # Artificially small budget for demonstration purposes
    budget = 70 
    history = [
        {"role": "system", "content": "You are a helpful RAG assistant."}
    ]
    
    # Simulate a long user-assistant interaction with retrieved documents
    turns = [
        "Hi, what is your purpose?",
        "I am an assistant to answer your questions based on documents.",
        "Can you explain the new tax policies? [RETRIEVED DOC: The new tax policies state that standard deductions increased by 10%.]",
        "Sure, the new tax policies state that standard deductions increased by 10%.",
        "What about the 401k contribution limits? [RETRIEVED DOC: The 401k limits have been raised to $23,000 for this year.]",
        "The 401k limits have been raised to $23,000 for this year.",
        "Thanks, can you summarize our chat?",
        "We discussed my purpose, new tax policies, and 401k limits."
    ]

    print("=== Starting Chat Simulation ===")
    print(f"Token Budget: {budget}\n")

    for i, msg in enumerate(turns):
        role = "user" if i % 2 == 0 else "assistant"
        
        # Add to history
        history.append({"role": role, "content": msg})
        
        # Check tokens and trim
        current_tokens = total_tokens(history)
        print(f"Turn {i+1} ({role}): Total tokens: {current_tokens}")
        
        if current_tokens > budget:
            print(f"   -> Exceeded budget of {budget}. Trimming history...")
            trim_history(history, budget=budget)
            print(f"   -> Tokens after trimming: {total_tokens(history)}")
        
        print(f"   -> Current history length: {len(history)} messages\n")

if __name__ == "__main__":
    simulate_chat()
