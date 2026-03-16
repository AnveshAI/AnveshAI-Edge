"""
Conversation Engine — handles general chat using pattern-response pairs.

How it works:
    1. Loads 'conversation.txt' at startup.
       Each non-comment line format: PATTERN|||RESPONSE
    2. Tries to match user input against compiled regex patterns.
    3. Returns (response, matched):
         matched = True  → a pattern fired; use the response directly
         matched = False → no match; caller should escalate to LLM fallback

EXTENSION POINT: The respond() method can be replaced wholesale with an LLM
call without touching any other module. Contract: (str) → (str, bool).
"""

import os
import re
from typing import List, Tuple


CONVERSATION_FILE = os.path.join(os.path.dirname(__file__), "conversation.txt")


def _load_patterns(filepath: str) -> List[Tuple[re.Pattern, str]]:
    patterns: List[Tuple[re.Pattern, str]] = []
    if not os.path.exists(filepath):
        return patterns
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|||" not in line:
                continue
            pattern_part, response_part = line.split("|||", 1)
            pattern_str  = pattern_part.strip()
            response_str = response_part.strip()
            if not pattern_str or not response_str:
                continue
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                patterns.append((compiled, response_str))
            except re.error:
                continue
    return patterns


class ConversationEngine:
    """Rule-based pattern-matching chat engine backed by conversation.txt."""

    def __init__(self, conversation_file: str = CONVERSATION_FILE):
        self.patterns = _load_patterns(conversation_file)

    def respond(self, user_input: str) -> Tuple[str, bool]:
        """
        Match user input against stored patterns.

        Returns:
            (response, matched)
            matched = True  → pattern found, response is ready to use
            matched = False → no pattern matched; escalate to LLM
        """
        text = user_input.strip()
        for pattern, response in self.patterns:
            if pattern.search(text):
                return (response, True)
        # Signal: no rule matched — let the LLM handle it
        return ("", False)
