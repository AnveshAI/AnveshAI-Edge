"""
Memory System — stores and retrieves conversation history using SQLite.

Schema:
    conversations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,  -- ISO 8601 format
        user_input  TEXT    NOT NULL,
        response    TEXT    NOT NULL
    )

Commands:
    /history   → display the last 10 interactions
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Tuple


DB_PATH = os.path.join(os.path.dirname(__file__), "anveshai_memory.db")
HISTORY_LIMIT = 10


def _get_connection() -> sqlite3.Connection:
    """Open and return a SQLite database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    """
    Create the conversations table if it does not already exist.
    Called once at startup.
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                user_input  TEXT    NOT NULL,
                response    TEXT    NOT NULL
            )
        """)
        conn.commit()


def save_interaction(user_input: str, response: str) -> None:
    """
    Persist a single interaction (user input + assistant response) to the DB.
    """
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (timestamp, user_input, response) VALUES (?, ?, ?)",
            (timestamp, user_input, response),
        )
        conn.commit()


def get_recent_history(limit: int = HISTORY_LIMIT) -> List[sqlite3.Row]:
    """
    Retrieve the most recent `limit` interactions, ordered oldest-first.
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, timestamp, user_input, response
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    # Reverse so oldest appears first
    return list(reversed(rows))


def format_history() -> str:
    """
    Return a formatted string of the last HISTORY_LIMIT interactions.
    Returns a notice if there's nothing to show.
    """
    rows = get_recent_history()

    if not rows:
        return "No conversation history yet."

    lines = [f"  Last {len(rows)} interaction(s):\n"]
    for row in rows:
        lines.append(f"  [{row['timestamp']}]")
        lines.append(f"  You      : {row['user_input']}")
        lines.append(f"  AnveshAI : {row['response']}")
        lines.append("")  # blank line between entries

    return "\n".join(lines).rstrip()
