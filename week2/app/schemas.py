"""
Pydantic schemas for strict API contracts.

Rationale:
- Explicit request/response validation at API boundaries
- Self-documenting API with auto-generated OpenAPI specs
- Type safety throughout the application layer
- Clear separation between API DTOs and internal domain models
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Domain Models (Internal representations)
# =============================================================================


class ActionItemModel(BaseModel):
    """Internal domain model for action items."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    note_id: Optional[int] = None
    text: str
    done: bool = False
    created_at: Optional[datetime] = None


class NoteModel(BaseModel):
    """Internal domain model for notes."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    content: str
    created_at: Optional[datetime] = None


# =============================================================================
# Request Schemas (API Input)
# =============================================================================


class ExtractActionItemsRequest(BaseModel):
    """Request schema for extracting action items from text."""

    text: str = Field(
        ...,
        min_length=1,
        description="The text to extract action items from",
    )
    save_note: bool = Field(
        default=False,
        description="Whether to save the text as a note",
    )


class CreateNoteRequest(BaseModel):
    """Request schema for creating a new note."""

    content: str = Field(
        ...,
        min_length=1,
        description="The content of the note",
    )


class MarkDoneRequest(BaseModel):
    """Request schema for marking an action item done/undone."""

    done: bool = Field(
        default=True,
        description="Whether the action item is done",
    )


# =============================================================================
# Response Schemas (API Output)
# =============================================================================


class ActionItemResponse(BaseModel):
    """Response schema for a single action item."""

    id: int = Field(..., description="Unique identifier")
    text: str = Field(..., description="The action item text")


class ActionItemDetailResponse(BaseModel):
    """Detailed response schema for an action item."""

    id: int = Field(..., description="Unique identifier")
    note_id: Optional[int] = Field(None, description="Associated note ID")
    text: str = Field(..., description="The action item text")
    done: bool = Field(..., description="Completion status")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class ExtractActionItemsResponse(BaseModel):
    """Response schema for action item extraction."""

    note_id: Optional[int] = Field(
        None,
        description="ID of the saved note (if save_note was true)",
    )
    items: List[ActionItemResponse] = Field(
        default_factory=list,
        description="Extracted action items",
    )


class NoteResponse(BaseModel):
    """Response schema for a note."""

    id: int = Field(..., description="Unique identifier")
    content: str = Field(..., description="Note content")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class MarkDoneResponse(BaseModel):
    """Response schema for marking an action item done."""

    id: int = Field(..., description="Action item ID")
    done: bool = Field(..., description="New completion status")


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code for programmatic handling")


class NotesListResponse(BaseModel):
    """Response schema for listing all notes."""

    notes: List[NoteResponse] = Field(
        default_factory=list,
        description="List of all notes",
    )
