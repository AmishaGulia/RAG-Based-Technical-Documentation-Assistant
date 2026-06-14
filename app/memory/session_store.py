"""
Bonus — Conversation Memory (Session Store)

Maintains per-session chat history in memory (dict). For production, replace
the in-process dict with Redis or a database so history survives restarts.

Each session stores a list of LangChain BaseMessage objects (HumanMessage /
AIMessage) so they can be passed directly to LangGraph state and to the
generation prompt.
"""

import uuid
import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

# Module-level store: {session_id: [BaseMessage, ...]}
_sessions: dict[str, list[BaseMessage]] = {}


def get_or_create_session(session_id: str | None = None) -> str:
    """Return the given session_id, creating a new one if None is passed."""
    if session_id is None or session_id not in _sessions:
        sid = session_id or str(uuid.uuid4())
        _sessions[sid] = []
        logger.debug("New session: %s", sid)
        return sid
    return session_id


def get_history(session_id: str) -> list[BaseMessage]:
    """Return the message history for a session (empty list if unknown)."""
    return _sessions.get(session_id, [])


def append_turn(session_id: str, question: str, answer: str) -> None:
    """Append a completed Q&A turn to the session history."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append(HumanMessage(content=question))
    _sessions[session_id].append(AIMessage(content=answer))
    logger.debug("Session %s now has %d messages", session_id, len(_sessions[session_id]))


def clear_session(session_id: str) -> None:
    """Delete a session's history."""
    _sessions.pop(session_id, None)


def list_sessions() -> list[str]:
    return list(_sessions.keys())
