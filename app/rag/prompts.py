"""
All LLM prompt templates used by the RAG pipeline.

Keeping prompts in one place makes them easy to review, version, and tune
without touching node logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Query Analysis ─────────────────────────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a search query optimizer for a technical documentation assistant.
Your job is to analyse a user question and return a JSON object with two fields:
  - "rewritten_query": an improved version of the question that adds relevant
    synonyms, expands abbreviations, and is more likely to match documentation text.
  - "query_type": one of ["conceptual", "how-to", "troubleshooting", "api-reference"]

Rules:
- Keep the rewritten_query concise (under 200 chars).
- Do NOT answer the question — only rewrite it.
- If the question already has prior context from chat history, incorporate it.
- Return ONLY valid JSON with the two fields above. No markdown fences.

Examples:
  Input: "how to add middleware?"
  Output: {{"rewritten_query": "how to add and configure middleware in FastAPI ASGI application", "query_type": "how-to"}}

  Input: "what is LCEL?"
  Output: {{"rewritten_query": "LangChain Expression Language LCEL definition concept explanation", "query_type": "conceptual"}}
""",
    ),
    ("human", "Chat history:\n{chat_history}\n\nQuestion: {question}"),
])


# ── Document Grading ───────────────────────────────────────────────────────────

GRADING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a relevance grader for a technical documentation retrieval system.
Given a user question and a retrieved document chunk, decide if the chunk contains
information that is useful for answering the question.

Answer with ONLY the JSON: {{"relevant": true}} or {{"relevant": false}}
Do not explain. Do not add any other text.""",
    ),
    (
        "human",
        "Question: {question}\n\nDocument chunk:\n{document}",
    ),
])


# ── Answer Generation ──────────────────────────────────────────────────────────

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful technical documentation assistant.
Use ONLY the provided context documents to answer the question.
Rules:
- Be accurate, clear, and concise.
- Cite sources inline using [Source: <filename>] notation whenever you use information from a chunk.
- If the context covers the answer only partially, say so and cite what you found.
- Never fabricate information not in the context.
- Format code examples in fenced code blocks with the appropriate language tag.
- If chat history is provided, take it into account for follow-up questions.
""",
    ),
    (
        "human",
        "Chat history:\n{chat_history}\n\nContext documents:\n{context}\n\nQuestion: {question}",
    ),
])


# ── Hallucination Check ────────────────────────────────────────────────────────

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a factual grounding checker.
Given a generated answer and the context documents it was based on, determine
whether every factual claim in the answer is directly supported by the context.

Answer with ONLY the JSON: {{"grounded": true}} or {{"grounded": false}}
Do not explain. Do not add any other text.""",
    ),
    (
        "human",
        "Context documents:\n{context}\n\nGenerated answer:\n{answer}",
    ),
])


# ── Query Rewrite (after failed retrieval) ────────────────────────────────────

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a search query rewriter.
The previous retrieval attempt returned no relevant documents for the question below.
Rewrite the query to try a different angle — use different vocabulary, break it into
sub-concepts, or be more specific/general as appropriate.

Return ONLY the rewritten query string. No JSON. No explanation.""",
    ),
    (
        "human",
        "Original question: {question}\nPrevious query: {previous_query}\nAttempt number: {retry_count}",
    ),
])
