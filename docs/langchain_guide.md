# LangChain — Complete Guide

## What is LangChain?

LangChain is an open-source framework for developing applications powered by large language models (LLMs). It provides:

- Composable primitives for working with LLMs
- Integrations with 100+ LLM providers, vector stores, and tools
- The LangChain Expression Language (LCEL) for declarative chain composition
- Built-in abstractions for RAG, agents, memory, and more

---

## Installation

```bash
pip install langchain langchain-core langchain-community
# Plus your LLM provider, e.g.:
pip install langchain-anthropic langchain-openai
```

---

## LangChain Expression Language (LCEL)

LCEL is the recommended way to compose LangChain components. It uses the `|` operator (pipe) to build chains:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

model = ChatAnthropic(model="claude-haiku-4-5-20251001")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of France?"})
# "Paris"
```

### Streaming with LCEL

```python
for chunk in chain.stream({"question": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

### Async with LCEL

```python
result = await chain.ainvoke({"question": "Hello"})
```

---

## Chat Models

### Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=1024,
)
response = llm.invoke("Hello!")
print(response.content)
```

### OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### Message types

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi there!"),
]
response = llm.invoke(messages)
# response.content is the AI reply
```

---

## Prompt Templates

### ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("human", "{question}"),
])

# Format and invoke
chain = template | llm
result = chain.invoke({"domain": "Python", "question": "What are decorators?"})
```

### PromptTemplate (for legacy completions)

```python
from langchain_core.prompts import PromptTemplate

pt = PromptTemplate.from_template("Explain {concept} in simple terms.")
formatted = pt.format(concept="recursion")
```

---

## Output Parsers

### StrOutputParser — plain string output

```python
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()
```

### JsonOutputParser — structured JSON

```python
from langchain_core.output_parsers import JsonOutputParser
chain = prompt | llm | JsonOutputParser()
result = chain.invoke(...)  # returns a dict
```

### PydanticOutputParser — typed output

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=Answer)
chain = prompt | llm | parser
```

---

## Document Loaders

Load content from various sources into Document objects:

```python
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    DirectoryLoader,
)

# Load a text file
loader = TextLoader("path/to/file.txt")
docs = loader.load()  # list[Document]

# Load from URL
loader = WebBaseLoader("https://example.com/page")
docs = loader.load()

# Load all .md files from a directory
loader = DirectoryLoader("./docs", glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
```

Each `Document` has:
- `page_content: str` — the text content
- `metadata: dict` — source, page number, etc.

---

## Text Splitters

Split large documents into chunks for embedding:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(docs)
```

### Code-aware splitter

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)
```

---

## Embeddings

Convert text to vector representations:

```python
# HuggingFace (local, free)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("Hello world")

# OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

---

## Vector Stores

Store and retrieve embeddings:

### ChromaDB

```python
from langchain_community.vectorstores import Chroma

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs",
)

# Connect to existing
vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Search
results = vectorstore.similarity_search("What is FastAPI?", k=5)
```

### FAISS

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

# Load
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## Retrieval

### Basic retriever

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("How do I add middleware?")
```

### MMR retriever (diversity-aware)

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)
```

---

## Chains

### Retrieval-Augmented Generation (RAG) chain

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is dependency injection in FastAPI?")
```

---

## Memory / Chat History

```python
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def chat(question: str) -> str:
    response = chain.invoke({"question": question, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response
```

---

## Tool Use / Function Calling

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # implementation
    return "search results..."

llm_with_tools = llm.bind_tools([search_web])
response = llm_with_tools.invoke("Search for the latest Python release")
```
