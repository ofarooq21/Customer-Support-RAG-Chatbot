# Customer-Support-RAG-Chatbot

This repository hosts the code and configuration for a Customer Support RAG Chatbot. The system leverages a large dataset of customer support instructions, vector embeddings for semantic search, and a large language model (LLM) for generating contextually relevant and helpful responses.

Overview
Key Components:

Data Preprocessing:
Cleans and normalizes customer support instructions and responses.
Embeddings and Vector Storage (Milvus):
Converts instructions into embeddings using Sentence Transformers and stores them in Milvus for efficient similarity search.
Hybrid Search:
Combines semantic vector search with keyword filtering for improved retrieval quality.
RAG Pipeline:
Uses retrieved context and passes it to an LLM (GPT-4) to generate a helpful and context-aware response.
Evaluation (RAGAS):
Evaluates the chatbot's performance on metrics such as response relevance and faithfulness.
Files and Directories

data_preprocessing.py:
Loads the dataset, cleans, preprocesses, and saves a CSV of cleaned instructions and responses.

embedding_storage_milvus.py:
Connects to Milvus, creates a collection schema, generates embeddings using Sentence Transformers, and inserts data into the vector database.

hybrid_search.py:
Implements a hybrid retrieval mechanism using Milvus vector search and keyword search to find relevant context for a given query.

generate_response.py:
Retrieves the top-k relevant documents and uses GPT-4 to generate a final response.

test_rag_pipeline.py:
Provides an interactive shell where you can test the chatbot by typing user queries. Logs interactions for later evaluation.

evaluate_ragas.py:
Uses the RAGAS library to evaluate the chatbot’s performance on logged interactions, measuring metrics like response relevancy and faithfulness.

Prerequisites
Python 3.8+

Key Python Packages:

datasets
pandas
sentence-transformers
pymilvus
ragas
openai
numpy
regex (re)
Milvus Vector Database:
Make sure you have a running Milvus instance.
Milvus Installation Docs

OpenAI API Key:
You will need an OpenAI API key for GPT-4 access. Set it as an environment variable:

