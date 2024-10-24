# Complete RAG System Implementation Guide

This comprehensive guide walks through building a complete Retrieval-Augmented Generation (RAG) system using LlamaIndex, Jina AI embeddings, Pinecone vector database, and Groq for generation. The system processes documents, creates embeddings, stores them for retrieval, and generates responses to queries.

## Prerequisites

Before starting, ensure you have the following:

1. Python 3.7 or later installed
2. Required API keys:
   - Jina AI API key
   - Pinecone API key
   - Groq API key
3. A PDF document to process

## Installation

Install all required packages:

```bash
pip install python-dotenv llama-parse llama-index pinecone-client requests groq
```

## Project Setup

1. Create your project structure:
```
your_project/
├── .env
├── data/
│   └── report.pdf
├── main.py
└── rag_query.py
```

2. Create a `.env` file with your API keys:
```plaintext
JINA_API_KEY=your_jina_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key_here
```

## Part 1: Document Processing and Indexing (main.py)

### 1. Environment Setup and Imports

```python
from dotenv import load_dotenv
load_dotenv()
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import requests
import os
import json
from pinecone import Pinecone
```

### 2. PDF Processing

```python
parser = LlamaParse(
    result_type='markdown',
)
file_extractor = {".pdf": parser}
output_docs = SimpleDirectoryReader(
    input_files=['./data/report.pdf'], 
    file_extractor=file_extractor
)
docs = output_docs.load_data()

# Convert to markdown and save
md_text = ""
for doc in docs:
    md_text += doc.text
with open('output.md', 'w') as file_handle:
    file_handle.write(md_text)
print("Markdown file created successfully")
```

### 3. Text Chunking

```python
chunk_size = 1000
chunk_overlap = 200

def fixed_size_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = fixed_size_chunks(md_text, chunk_size, chunk_overlap)
print(f"Number of chunks: {len(chunks)}")
```

### 4. Generating Embeddings

```python
jina_api_key = os.getenv('JINA_API_KEY')
headers = {
    'Authorization': f'Bearer {jina_api_key}',
    'Content-Type': 'application/json'
}
url = 'https://api.jina.ai/v1/embeddings'

embedded_chunks = []
for chunk in chunks:
    payload = {
        'input': chunk,
        'model': 'jina-embeddings-v2-base-en'
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        embedded_chunks.append(response.json()['data'][0]['embedding'])
    else:
        print(f"Error embedding chunk: {response.status_code}")

print(f"Number of embedded chunks: {len(embedded_chunks)}")
```

### 5. Saving Embeddings Locally

```python
output_file = 'embedded_chunks.json'
data_to_save = {
    'chunks': chunks,
    'embeddings': embedded_chunks
}
with open(output_file, 'w') as f:
    json.dump(data_to_save, f)
print(f"Embedded chunks saved to {output_file}")
```

### 6. Pinecone Integration

```python
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "pesuio-rag"  # Replace with your preferred index name

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Dimension of Jina embeddings
        metric='cosine'
    )

index = pc.Index(index_name)

# Prepare and upload vectors
vectors_to_upsert = [
    {
        'id': f'chunk_{i}',
        'values': embedding,
        'metadata': {'text': chunk}
    }
    for i, (chunk, embedding) in enumerate(zip(chunks, embedded_chunks))
]

index.upsert(vectors=vectors_to_upsert)
print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone")
```

## Part 2: Retrieval and Generation (rag.py)

### 1. Setup and Imports

```python
import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "pesuio-rag"
index = pc.Index(index_name)

# Jina API setup
jina_api_key = os.getenv('JINA_API_KEY')
headers = {
    'Authorization': f'Bearer {jina_api_key}',
    'Content-Type': 'application/json'
}
url = 'https://api.jina.ai/v1/embeddings'

# Initialize Groq client
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
```

### 2. Retrieval Function

```python
def retrieve_nearest_chunks(query, top_k=5):
    # Embed the query using Jina API
    payload = {
        'input': query,
        'model': 'jina-embeddings-v2-base-en'
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error embedding query: {response.status_code}")
        return []
    
    query_embedding = response.json()['data'][0]['embedding']
    
    # Query Pinecone for nearest vectors
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return query_response['matches']
```

### 3. Generation Function

```python
def generate_response(query, context):
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not in the context, say "I don't have enough information to answer that question."
    
Context:
{context}

User's question: {query}
Answer:"""

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70b-versatile",
    )
    return response.choices[0].message.content
```

### 4. Main Execution

```python
if __name__ == "__main__":
    query = "What is the title of the report?"
    
    # Retrieve relevant chunks
    nearest_chunks = retrieve_nearest_chunks(query)
    
    # Combine chunks into context
    context = "\n".join([chunk['metadata']['text'] for chunk in nearest_chunks])
    
    # Generate response
    rag_response = generate_response(query, context)
    
    # Print results
    print(f"\nQuery: {query}")
    print("\nRAG Response:")
    print(rag_response)
    print("\nRetrieved chunks:")
    for i, chunk in enumerate(nearest_chunks, 1):
        print(f"\n{i}. Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['metadata']['text'][:200]}...")
```

## Running the Complete System

1. **Index your document:**
```bash
python rag_indexer.py
```

2. **Query the system:**
```bash
python rag_query.py
```

## Expected Outputs

### Indexing Phase:
```
Markdown file created successfully
Number of chunks: [X]
Number of embedded chunks: [X]
Embedded chunks saved to embedded_chunks.json
Uploaded [X] vectors to Pinecone
```

### Query Phase:
```
Query: What is the title of the report?

RAG Response:
[Generated answer based on the retrieved context]

Retrieved chunks:
1. Score: 0.8532
Text: [First 200 characters of the chunk]...

2. Score: 0.7845
Text: [First 200 characters of the chunk]...
```

## Customization Options

1. **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in the indexer
2. **Retrieved Chunks**: Adjust `top_k` in `retrieve_nearest_chunks()`
3. **Prompt Template**: Modify the prompt in `generate_response()`
4. **Model Selection**: Change the Groq model in `generate_response()`

## Troubleshooting

Common issues and solutions:

1. **PDF Processing Issues**
   - Ensure PDF is readable and not password-protected
   - Check file path is correct

2. **API Errors**
   - Verify API keys in `.env` file
   - Check API endpoint URLs
   - Confirm API quotas and limits

3. **Memory Issues**
   - Process large documents in batches
   - Reduce chunk size
   - Implement pagination for large result sets

4. **Poor Results**
   - Adjust chunk size and overlap
   - Increase `top_k` for more context
   - Modify prompt template
   - Try different embedding models
