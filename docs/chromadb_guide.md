# ChromaDB — Complete Guide

## What is ChromaDB?

ChromaDB is an open-source, AI-native vector database designed for embedding storage and retrieval. It is optimised for developer simplicity and runs entirely in-process (no separate server required) or as a client-server.

Key features:
- Runs locally with full persistence
- Sub-millisecond query latency on millions of vectors
- Built-in embedding functions (or bring your own)
- Python and JavaScript SDKs
- Docker and cloud deployment support

---

## Installation

```bash
pip install chromadb
```

---

## Quick Start

```python
import chromadb

# In-memory client (no persistence)
client = chromadb.Client()

# Persistent client (recommended for production)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection
collection = client.get_or_create_collection(name="my_docs")
```

---

## Collections

A collection is like a table — it stores documents, embeddings, and metadata together.

### Creating collections

```python
# Default (cosine similarity)
collection = client.create_collection("my_docs")

# Specify distance function
from chromadb.utils import embedding_functions

collection = client.create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},  # cosine | l2 | ip
)

# With a built-in embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.create_collection(name="my_docs", embedding_function=ef)
```

### Listing and getting collections

```python
# List all collections
collections = client.list_collections()

# Get an existing collection
collection = client.get_collection("my_docs")

# Get or create
collection = client.get_or_create_collection("my_docs")

# Delete
client.delete_collection("my_docs")
```

---

## Adding Documents

```python
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "file1.txt"}, {"source": "file2.txt"}],
    ids=["id1", "id2"],  # must be unique strings
)
```

### Batch adding

```python
# ChromaDB handles batching internally, but you can also batch manually
batch_size = 1000
for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    collection.add(
        documents=[d.page_content for d in batch],
        metadatas=[d.metadata for d in batch],
        ids=[f"doc_{i+j}" for j in range(len(batch))],
    )
```

### Upsert (add or update)

```python
collection.upsert(
    documents=["Updated content"],
    metadatas=[{"source": "file1.txt"}],
    ids=["id1"],
)
```

---

## Querying

### Basic similarity search

```python
results = collection.query(
    query_texts=["What is dependency injection?"],
    n_results=5,
)

# results is a dict with keys:
# - ids: list of lists
# - documents: list of lists
# - metadatas: list of lists
# - distances: list of lists
```

### Query with filters (where clause)

```python
results = collection.query(
    query_texts=["FastAPI routing"],
    n_results=5,
    where={"source": "fastapi_guide.md"},  # metadata filter
)

# Multiple conditions
results = collection.query(
    query_texts=["authentication"],
    n_results=5,
    where={
        "$and": [
            {"source": {"$in": ["fastapi_guide.md", "security.md"]}},
            {"chunk_index": {"$lt": 10}},
        ]
    },
)
```

### Get documents by ID

```python
results = collection.get(
    ids=["id1", "id2"],
    include=["documents", "metadatas"],
)
```

### Get all documents

```python
# Get first 100 (use offset for pagination)
results = collection.get(
    limit=100,
    offset=0,
    include=["documents", "metadatas"],
)
```

---

## Updating and Deleting

```python
# Update existing documents
collection.update(
    ids=["id1"],
    documents=["New content for id1"],
)

# Delete specific documents
collection.delete(ids=["id1", "id2"])

# Delete by metadata filter
collection.delete(where={"source": "old_file.txt"})
```

---

## Collection Count

```python
count = collection.count()
print(f"Total documents: {count}")
```

---

## Using with LangChain

ChromaDB integrates natively with LangChain:

```python
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create from documents
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="tech_docs",
)

# Similarity search
docs = vectorstore.similarity_search("How to add middleware in FastAPI?", k=5)

# Search with scores
docs_with_scores = vectorstore.similarity_search_with_score("FastAPI middleware", k=5)
for doc, score in docs_with_scores:
    print(f"Score: {score:.4f} | Source: {doc.metadata.get('source')}")
```

---

## Embedding Functions

ChromaDB supports multiple embedding functions:

```python
from chromadb.utils import embedding_functions

# Sentence Transformers (local, free)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# OpenAI
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small",
)

# Google Gemini
ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key="your-key",
    model_name="models/text-embedding-004",
)
```

---

## Performance Tips

1. **Batch inserts**: Add documents in batches of 100–5000 for best throughput.
2. **HNSW tuning**: Increase `hnsw:ef_construction` for better recall at index time.
3. **Persist strategically**: `PersistentClient` auto-persists; no manual save needed.
4. **Use IDs wisely**: Deterministic IDs enable efficient upsert-based re-ingestion.
5. **Filter early**: Use `where` clauses to narrow the search space before similarity scoring.

---

## Distance Functions

| Metric | `hnsw:space` value | Best for |
|--------|-------------------|----------|
| Cosine similarity | `cosine` | Normalised embeddings (most common) |
| L2 distance | `l2` | When magnitude matters |
| Inner product | `ip` | When vectors are pre-normalised |

ChromaDB returns distances (lower = more similar for l2/cosine, higher for ip).
