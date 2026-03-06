"""
Notes router with strict Pydantic request/response schemas.

Rationale:
- Pydantic models provide automatic validation
- Type-safe request handling eliminates manual parsing
- Response models ensure consistent API contracts
- Custom exceptions bubble up with proper HTTP status codes
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter

from .. import db
from ..schemas import CreateNoteRequest, NoteResponse, NotesListResponse


router = APIRouter(prefix="/notes", tags=["notes"])


@router.get(
    "",
    response_model=NotesListResponse,
    summary="List all notes",
    description="Returns all stored notes ordered by ID descending",
)
def list_notes() -> NotesListResponse:
    """List all notes."""
    notes = db.list_notes()
    return NotesListResponse(
        notes=[
            NoteResponse(
                id=note.id,
                content=note.content,
                created_at=str(note.created_at) if note.created_at else None,
            )
            for note in notes
        ]
    )


@router.post(
    "",
    response_model=NoteResponse,
    status_code=201,
    summary="Create a new note",
    description="Creates a new note with the provided content",
)
def create_note(request: CreateNoteRequest) -> NoteResponse:
    """Create a new note."""
    note = db.insert_note(request.content)
    return NoteResponse(
        id=note.id,
        content=note.content,
        created_at=str(note.created_at) if note.created_at else None,
    )


@router.get(
    "/{note_id}",
    response_model=NoteResponse,
    summary="Get a note by ID",
    description="Returns a single note by its ID",
)
def get_single_note(note_id: int) -> NoteResponse:
    """Get a single note by ID."""
    note = db.get_note(note_id)
    return NoteResponse(
        id=note.id,
        content=note.content,
        created_at=str(note.created_at) if note.created_at else None,
    )
