"""RAG query strings and helpers for self-employed / SEP / solo 401(k) content."""

from __future__ import annotations


def self_employed_401k_rag_query(user_topic: str) -> str:
    """Retrieval query biased toward self-employed 401(k), SEP, and solo plan rules from the brochure."""
    t = user_topic.strip() or "retirement savings"
    return (
        f"{t} self-employed SEP solo 401(k) retirement plan contribution limits "
        "compensation deduction employer employee eligibility elective deferral"
    )
