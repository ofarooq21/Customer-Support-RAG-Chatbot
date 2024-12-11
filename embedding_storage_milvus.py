# embedding_storage_milvus.py

import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
import numpy as np

# Connect to Milvus
print("Connecting to Milvus...")
connections.connect("default", host="localhost", port="19530")

# Define the collection schema
print("Defining collection schema...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="instruction", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="response", dtype=DataType.VARCHAR, max_length=2048),
]

collection_schema = CollectionSchema(fields, description="Customer Support Instructions")
collection_name = "customer_support"

# Drop the collection if it exists
if utility.has_collection(collection_name):
    print(f"Dropping existing collection '{collection_name}'...")
    utility.drop_collection(collection_name)

# Create the collection
print(f"Creating collection '{collection_name}'...")
collection = Collection(name=collection_name, schema=collection_schema)

# Load the preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('preprocessed_data.csv')

# Truncate the response field to meet the 2048-character limit
df['clean_response'] = df['clean_response'].apply(lambda x: x[:2048] if isinstance(x, str) else '')

# Initialize the embedding model
print("Initializing embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
instructions = df['clean_instruction'].tolist()
print("Generating embeddings...")
embeddings = embedding_model.encode(instructions, batch_size=64, show_progress_bar=True)
print("Embeddings generated.")

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_list = embeddings.tolist()

# Prepare data for insertion
print("Preparing data for insertion...")
ids = [i for i in range(len(df))]
responses = df['clean_response'].tolist()

data = [
    ids,
    instructions,
    embeddings_list,
    responses
]

# Insert data into Milvus
try:
    print("Inserting data into Milvus...")
    collection.insert(data)
    print("Data insertion complete.")
except Exception as e:
    print(f"Error during data insertion: {e}")

# Flush to ensure data is persisted
collection.flush()

# Create an index on the embedding field
try:
    print("Creating index on embeddings...")
    index_params = {
        "metric_type": "IP",  # Changed to 'IP'
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created.")
except Exception as e:
    print(f"Error during index creation: {e}")

# Load the collection into memory
collection.load()

# Test the collection with a sample query
print("Performing a test search...")
test_query = "how can i cancel my order"
test_embedding = embedding_model.encode([test_query])
# Normalize the query embedding
test_embedding = test_embedding / np.linalg.norm(test_embedding, axis=1, keepdims=True)
test_embedding = test_embedding.tolist()

search_params = {
    "metric_type": "IP",  # Changed to 'IP'
    "params": {"ef": 64}
}

try:
    results = collection.search(
        data=test_embedding,
        anns_field="embedding",
        param=search_params,
        limit=5,
        expr=None,
        output_fields=["instruction", "response"]
    )

    # Display the results
    print("\nSample Query Results:")
    for result in results[0]:
        print(f"Instruction: {result.entity.get('instruction')}")
        print(f"Response: {result.entity.get('response')}")
        print(f"Distance: {result.distance}\n")
except Exception as e:
    print(f"Error during search: {e}")