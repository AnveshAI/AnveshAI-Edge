"""
Knowledge Engine — retrieves relevant information from a local knowledge base.

How it works:
    1. Loads 'knowledge.txt' at startup (one paragraph per blank-line block).
    2. For a given query, scores each paragraph using keyword overlap.
    3. Returns the highest-scoring paragraph + a boolean indicating confidence.
       If confidence is low, the caller (main.py) will escalate to the LLM.

This is intentionally lightweight and fully offline. In the future it can be
swapped for a vector-based retrieval system (FAISS + sentence-transformers)
without changing the rest of the architecture.
"""

import os
import re
from typing import List, Tuple


KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "knowledge.txt")

# A paragraph must score at least this much to be considered a real match.
# Queries below this score are escalated to the LLM fallback.
MIN_RELEVANCE_SCORE = 2

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "and", "but",
    "or", "nor", "so", "yet", "both", "either", "neither", "not", "no",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "their", "tell", "explain", "describe", "give",
    "me", "some", "information", "about",
}


def _load_paragraphs(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    raw = re.split(r"\n\s*\n", content.strip())
    return [p.strip() for p in raw if p.strip()]


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def _score_paragraph(query_tokens: List[str], paragraph: str) -> int:
    para_lower = paragraph.lower()
    score = 0
    for token in query_tokens:
        if re.search(r"\b" + re.escape(token) + r"\b", para_lower):
            score += 2
        elif token in para_lower:
            score += 1
    return score


def _strip_knowledge_prefixes(text: str) -> str:
    prefixes = [
        "what is", "what are", "who is", "who are", "explain", "define",
        "tell me about", "describe", "how does", "why is", "when was",
        "where is", "history of", "meaning of", "knowledge:", "knowledge :",
        "learn about", "facts about", "information about",
    ]
    lowered = text.lower().strip()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


class KnowledgeEngine:
    """Local keyword-scored knowledge retrieval over knowledge.txt."""

    def __init__(self, knowledge_file: str = KNOWLEDGE_FILE):
        self.paragraphs: List[str] = _load_paragraphs(knowledge_file)
        self._loaded = len(self.paragraphs) > 0

    def is_loaded(self) -> bool:
        return self._loaded

    def query(self, user_input: str) -> Tuple[str, bool]:
        """
        Find the most relevant paragraph for the given query.

        Returns:
            (response, found)
            found = True  → a high-confidence match was found in the KB
            found = False → no confident match; caller should try the LLM
        """
        if not self._loaded:
            return (
                "Knowledge base unavailable. Ensure 'knowledge.txt' exists.",
                False,
            )

        clean_query = _strip_knowledge_prefixes(user_input)
        query_tokens = _tokenize(clean_query)

        if not query_tokens:
            return ("Could you rephrase? I couldn't parse the query.", False)

        scored: List[Tuple[int, str]] = [
            (_score_paragraph(query_tokens, para), para)
            for para in self.paragraphs
        ]

        best_score, best_para = max(scored, key=lambda x: x[0])

        if best_score < MIN_RELEVANCE_SCORE:
            # Signal to caller: escalate to LLM
            return ("", False)

        return (best_para, True)
