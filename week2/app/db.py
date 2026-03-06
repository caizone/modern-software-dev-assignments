"""
Database layer with proper connection management and domain model returns.

Rationale:
- Context manager pattern ensures connections are properly closed
- Single connection per request context (no redundant connections)
- Returns domain models instead of leaking sqlite3.Row
- Clear separation between data access and business logic
- Configurable database path via Settings
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from .config import get_settings
from .exceptions import DatabaseError, NoteNotFoundError, ActionItemNotFoundError
from .schemas import ActionItemModel, NoteModel


# =============================================================================
# Connection Management
# =============================================================================


class DatabaseManager:
    """
    Manages database connections with proper lifecycle control.
    
    This class provides:
    - Centralized connection creation
    - Context manager for automatic cleanup
    - Initialization of database schema
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self._db_path = db_path or settings.db_path
        self._ensure_data_directory()

    def _ensure_data_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection as a context manager.
        
        Usage:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                ...
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except sqlite3.Error as e:
            conn.rollback()
            raise DatabaseError(str(e), original_error=e) from e
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Initialize database schema (create tables if not exist)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS action_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    note_id INTEGER,
                    text TEXT NOT NULL,
                    done INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (note_id) REFERENCES notes(id)
                );
                """
            )
            conn.commit()


# Module-level instance (lazy initialization)
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def reset_db_manager() -> None:
    """Reset the database manager (useful for testing)."""
    global _db_manager
    _db_manager = None


# =============================================================================
# Repository Functions (Data Access Layer)
# =============================================================================


def init_db() -> None:
    """Initialize the database schema."""
    get_db_manager().init_schema()


def insert_note(content: str) -> NoteModel:
    """
    Insert a new note and return the created model.
    
    Args:
        content: The note content
        
    Returns:
        NoteModel with the created note data
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))
        conn.commit()
        note_id = cursor.lastrowid

        cursor.execute(
            "SELECT id, content, created_at FROM notes WHERE id = ?",
            (note_id,),
        )
        row = cursor.fetchone()
        return NoteModel(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
        )


def get_note(note_id: int) -> NoteModel:
    """
    Get a note by ID.
    
    Args:
        note_id: The note ID
        
    Returns:
        NoteModel with the note data
        
    Raises:
        NoteNotFoundError: If note doesn't exist
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, created_at FROM notes WHERE id = ?",
            (note_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise NoteNotFoundError(note_id)
        return NoteModel(
            id=row["id"],
            content=row["content"],
            created_at=row["created_at"],
        )


def list_notes() -> List[NoteModel]:
    """
    List all notes ordered by ID descending.
    
    Returns:
        List of NoteModel objects
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, content, created_at FROM notes ORDER BY id DESC")
        return [
            NoteModel(
                id=row["id"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]


def insert_action_items(
    items: List[str],
    note_id: Optional[int] = None,
) -> List[ActionItemModel]:
    """
    Insert multiple action items and return the created models.
    
    Args:
        items: List of action item texts
        note_id: Optional associated note ID
        
    Returns:
        List of ActionItemModel objects
    """
    if not items:
        return []

    db = get_db_manager()
    created: List[ActionItemModel] = []

    with db.get_connection() as conn:
        cursor = conn.cursor()
        for item_text in items:
            cursor.execute(
                "INSERT INTO action_items (note_id, text) VALUES (?, ?)",
                (note_id, item_text),
            )
            item_id = cursor.lastrowid

            cursor.execute(
                "SELECT id, note_id, text, done, created_at FROM action_items WHERE id = ?",
                (item_id,),
            )
            row = cursor.fetchone()
            created.append(
                ActionItemModel(
                    id=row["id"],
                    note_id=row["note_id"],
                    text=row["text"],
                    done=bool(row["done"]),
                    created_at=row["created_at"],
                )
            )
        conn.commit()

    return created


def list_action_items(note_id: Optional[int] = None) -> List[ActionItemModel]:
    """
    List action items, optionally filtered by note ID.
    
    Args:
        note_id: Optional filter by note ID
        
    Returns:
        List of ActionItemModel objects
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        if note_id is None:
            cursor.execute(
                "SELECT id, note_id, text, done, created_at "
                "FROM action_items ORDER BY id DESC"
            )
        else:
            cursor.execute(
                "SELECT id, note_id, text, done, created_at "
                "FROM action_items WHERE note_id = ? ORDER BY id DESC",
                (note_id,),
            )
        return [
            ActionItemModel(
                id=row["id"],
                note_id=row["note_id"],
                text=row["text"],
                done=bool(row["done"]),
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]


def get_action_item(action_item_id: int) -> ActionItemModel:
    """
    Get an action item by ID.
    
    Args:
        action_item_id: The action item ID
        
    Returns:
        ActionItemModel with the action item data
        
    Raises:
        ActionItemNotFoundError: If action item doesn't exist
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, note_id, text, done, created_at "
            "FROM action_items WHERE id = ?",
            (action_item_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise ActionItemNotFoundError(action_item_id)
        return ActionItemModel(
            id=row["id"],
            note_id=row["note_id"],
            text=row["text"],
            done=bool(row["done"]),
            created_at=row["created_at"],
        )


def mark_action_item_done(action_item_id: int, done: bool) -> ActionItemModel:
    """
    Mark an action item as done/undone.
    
    Args:
        action_item_id: The action item ID
        done: Whether the item is done
        
    Returns:
        Updated ActionItemModel
        
    Raises:
        ActionItemNotFoundError: If action item doesn't exist
    """
    db = get_db_manager()
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Check existence first
        cursor.execute(
            "SELECT id FROM action_items WHERE id = ?",
            (action_item_id,),
        )
        if cursor.fetchone() is None:
            raise ActionItemNotFoundError(action_item_id)

        # Update
        cursor.execute(
            "UPDATE action_items SET done = ? WHERE id = ?",
            (1 if done else 0, action_item_id),
        )
        conn.commit()

        # Return updated model
        return get_action_item(action_item_id)
