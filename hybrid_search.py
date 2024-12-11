# hybrid_search.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import numpy as np

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Load the collection
collection_name = "customer_support"
collection = Collection(name=collection_name)
collection.load()

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def hybrid_search(query, top_k=5):
    # Normalize query embedding
    query_embedding = embedding_model.encode([query])
    query_embedding = normalize_embeddings(query_embedding)
    query_embedding = query_embedding.tolist()

    # Vector search in Milvus
    search_params = {
        "metric_type": "IP",
        "params": {"ef": 64}
    }

    try:
        vector_results = collection.search(
            data=query_embedding,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["instruction", "response"]
        )
    except Exception as e:
        print(f"Error during vector search: {e}")
        vector_results = []

    # Extract vector search results
    vector_instructions = []
    vector_responses = []
    if vector_results:
        for result in vector_results[0]:
            vector_instructions.append(result.entity.get('instruction'))
            vector_responses.append(result.entity.get('response'))

    # Keyword search using Pandas
    mask = df['clean_instruction'].str.contains(query, case=False, na=False)
    keyword_results = df[mask].head(top_k)

    # Combine results
    combined_instructions = vector_instructions.copy()
    combined_responses = vector_responses.copy()

    for idx, row in keyword_results.iterrows():
        instruction = row['clean_instruction']
        response = row['clean_response']
        if instruction not in combined_instructions:
            combined_instructions.append(instruction)
            combined_responses.append(response)

    # Limit to top_k results
    combined_instructions = combined_instructions[:top_k]
    combined_responses = combined_responses[:top_k]

    # Replace placeholders in responses
    placeholder_mappings = {
        "{{Order Number}}": "",
        "{{customer support hours}}": "9 AM to 5 PM",
        "{{customer support phone number}}": "1-800-123-4567",
        "{{website url}}": "www.example.com",
        "{{online company portal info}}": "your account dashboard",
        "{{online order interaction}}": "Order History"
    }

    responses = []
    for response in combined_responses:
        for placeholder, replacement in placeholder_mappings.items():
            response = response.replace(placeholder, replacement)
        responses.append(response)

    return combined_instructions, responses