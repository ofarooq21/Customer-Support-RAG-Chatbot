# test_rag_pipeline.py

import json
from generate_response import generate_response

def log_interaction(query, context, response):
    data = {
        "query": query,
        "context": context,
        "response": response,
        "feedback": None  # Placeholder for future feedback
    }
    with open('interaction_logs.jsonl', 'a') as f:
        json.dump(data, f)
        f.write('\n')

def main():
    conversation_history = []
    print("Welcome to the Customer Support Chatbot. Type 'exit' to quit.")
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response, conversation_history, context = generate_response(conversation_history, query)
        print(f"\nAssistant: {response}\n")

        # Log the interaction
        log_interaction(query, context, response)

if __name__ == "__main__":
    main()