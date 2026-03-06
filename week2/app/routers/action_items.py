"""
Action items router with strict Pydantic request/response schemas.

Rationale:
- Pydantic models provide automatic validation
- Type-safe request handling eliminates manual parsing
- Response models ensure consistent API contracts
- Proper HTTP status codes for different scenarios
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter

from .. import db
from ..schemas import (
    ActionItemDetailResponse,
    ActionItemResponse,
    ExtractActionItemsRequest,
    ExtractActionItemsResponse,
    MarkDoneRequest,
    MarkDoneResponse,
)
from ..services.extract import extract_action_items, extract_action_items_llm


router = APIRouter(prefix="/action-items", tags=["action-items"])


@router.post(
    "/extract",
    response_model=ExtractActionItemsResponse,
    summary="Extract action items from text",
    description="Extract actionable items from the provided text by performing heuristic-based extraction",
)
def extract(request: ExtractActionItemsRequest) -> ExtractActionItemsResponse:
    """
    Extract action items from text using LLM.
    
    Optionally saves the text as a note if save_note is True.
    """
    note_id: Optional[int] = None

    if request.save_note:
        note = db.insert_note(request.text)
        note_id = note.id

    items = extract_action_items(request.text)
    created_items = db.insert_action_items(items, note_id=note_id)

    return ExtractActionItemsResponse(
        note_id=note_id,
        items=[
            ActionItemResponse(id=item.id, text=item.text)
            for item in created_items
        ],
    )


@router.post(
    "/extract/llm",
    response_model=ExtractActionItemsResponse,
    summary="Extract action items using LLM only",
    description="Uses LLM (Ollama) to extract actionable items from text without heuristic fallback",
)
def extract_llm(request: ExtractActionItemsRequest) -> ExtractActionItemsResponse:
    """
    Extract action items from text using LLM only.
    
    Unlike the standard /extract endpoint, this does NOT fall back to 
    heuristic extraction if the LLM fails. Raises an error instead.
    
    Optionally saves the text as a note if save_note is True.
    """
    note_id: Optional[int] = None

    if request.save_note:
        note = db.insert_note(request.text)
        note_id = note.id

    # Use LLM extraction without fallback
    items = extract_action_items_llm(request.text, fallback_on_error=False)
    created_items = db.insert_action_items(items, note_id=note_id)

    return ExtractActionItemsResponse(
        note_id=note_id,
        items=[
            ActionItemResponse(id=item.id, text=item.text)
            for item in created_items
        ],
    )


@router.get(
    "",
    response_model=List[ActionItemDetailResponse],
    summary="List action items",
    description="Returns all action items, optionally filtered by note_id",
)
def list_all(note_id: Optional[int] = None) -> List[ActionItemDetailResponse]:
    """List all action items, optionally filtered by note ID."""
    items = db.list_action_items(note_id=note_id)
    return [
        ActionItemDetailResponse(
            id=item.id,
            note_id=item.note_id,
            text=item.text,
            done=item.done,
            created_at=str(item.created_at) if item.created_at else None,
        )
        for item in items
    ]


@router.post(
    "/{action_item_id}/done",
    response_model=MarkDoneResponse,
    summary="Mark action item done/undone",
    description="Updates the completion status of an action item",
)
def mark_done(action_item_id: int, request: MarkDoneRequest) -> MarkDoneResponse:
    """Mark an action item as done or undone."""
    updated = db.mark_action_item_done(action_item_id, request.done)
    return MarkDoneResponse(id=updated.id, done=updated.done)
