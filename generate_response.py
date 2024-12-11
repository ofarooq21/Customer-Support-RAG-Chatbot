# generate_response.py

from hybrid_search import hybrid_search
from openai import OpenAI

client = OpenAI(api_key='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')

# Set your OpenAI API key directly

def generate_response(conversation_history, query):
    # Perform hybrid search to retrieve relevant documents
    instructions, responses = hybrid_search(query, top_k=5)

    # Replace placeholders in the context (ensure this is done)
    context = " \n ".join(responses)

    # Append the new query to the conversation history
    conversation_history.append(f"User: {query}")

    # Limit the conversation history to the last N exchanges
    max_history_length = 5
    if len(conversation_history) > max_history_length * 2:
        conversation_history = conversation_history[-(max_history_length * 2):]

    # Build the conversation history as a single string
    conversation_text = " \n ".join(conversation_history)

    # Create the prompt for the language model
    prompt = f"""You are a helpful customer support assistant.

Context:
{context}

Conversation:
{conversation_text}

Provide a helpful and detailed answer to the user's last question."""

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(model="gpt-4",  
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
        n=1,
        stop=None)

        # Extract the assistant's reply
        response_text = response.choices[0].message.content.strip()

        # Append the assistant's response to the conversation history
        conversation_history.append(f"Assistant: {response_text}")

        return response_text, conversation_history, context

    except Exception as e:
        print(f"Error during response generation: {e}")
        return "I'm sorry, but I'm having trouble generating a response at the moment.", conversation_history, context