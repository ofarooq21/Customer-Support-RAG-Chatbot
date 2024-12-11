# data_preprocessing.py

from datasets import load_dataset
import pandas as pd
import re

# Step 2.4: Load the Dataset
dataset = load_dataset('Bitext/Bitext-customer-support-llm-chatbot-training-dataset')

# Step 2.5: Convert to DataFrame
df = pd.DataFrame(dataset['train'])

# Step 2.7: Define the Preprocessing Function
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    # Lowercase the text
    text = text.lower()
    # Remove unwanted characters but keep placeholders like {{Order Number}}
    text = re.sub(r'[^a-zA-Z0-9\s\{\}\.,\?!]+', '', text)
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

# Step 2.8: Apply the Preprocessing Function
df['clean_instruction'] = df['instruction'].apply(clean_text)
df['clean_response'] = df['response'].apply(clean_text)

# Step 2.9: Data Validation
# Remove rows where clean_instruction or clean_response is empty
df = df.dropna(subset=['clean_instruction', 'clean_response'])
df = df[df['clean_instruction'].str.strip() != '']
df = df[df['clean_response'].str.strip() != '']

# Remove duplicate clean_instruction entries
df = df.drop_duplicates(subset=['clean_instruction'])

# Step 2.10: Verify the Preprocessed Data
print("Preprocessed Data:")
print(df[['clean_instruction', 'clean_response']].head())

# Step 2.11: Save the Preprocessed Data
df.to_csv('preprocessed_data.csv', index=False)