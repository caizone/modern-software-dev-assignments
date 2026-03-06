"""
Hierarchical custom exceptions for structured error handling.

Rationale:
- Specific exception types enable granular error handling
- HTTP status codes encapsulated with business exceptions
- Clear error messages with contextual information
- Proper exception chaining preserves stack traces
"""
from __future__ import annotations

from typing import Optional


class AppException(Exception):
    """
    Base exception for all application errors.
    
    All custom exceptions inherit from this, enabling catch-all
    handling while preserving specific exception semantics.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.status_code = status_code


# =============================================================================
# Validation Errors (400 Bad Request)
# =============================================================================


class ValidationError(AppException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
        )
        self.field = field


class EmptyInputError(ValidationError):
    """Raised when required input is empty or missing."""

    def __init__(self, field: str = "input") -> None:
        super().__init__(
            message=f"{field} is required and cannot be empty",
            field=field,
        )
        self.code = "EMPTY_INPUT"


# =============================================================================
# Not Found Errors (404 Not Found)
# =============================================================================


class NotFoundError(AppException):
    """Base exception for resource not found errors."""

    def __init__(self, resource: str, identifier: Optional[str] = None) -> None:
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} with id '{identifier}' not found"
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=404,
        )
        self.resource = resource
        self.identifier = identifier


class NoteNotFoundError(NotFoundError):
    """Raised when a note is not found."""

    def __init__(self, note_id: int) -> None:
        super().__init__(resource="Note", identifier=str(note_id))
        self.code = "NOTE_NOT_FOUND"


class ActionItemNotFoundError(NotFoundError):
    """Raised when an action item is not found."""

    def __init__(self, action_item_id: int) -> None:
        super().__init__(resource="Action item", identifier=str(action_item_id))
        self.code = "ACTION_ITEM_NOT_FOUND"


# =============================================================================
# External Service Errors (502/503)
# =============================================================================


class ExternalServiceError(AppException):
    """Base exception for external service failures."""

    def __init__(
        self,
        service: str,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        full_message = f"{service} error: {message}"
        super().__init__(
            message=full_message,
            code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
        )
        self.service = service
        self.original_error = original_error


class LLMServiceError(ExternalServiceError):
    """Raised when LLM service fails."""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            service="LLM",
            message=message,
            original_error=original_error,
        )
        self.code = "LLM_SERVICE_ERROR"


class LLMConnectionError(LLMServiceError):
    """Raised when connection to LLM service fails."""

    def __init__(self, original_error: Optional[Exception] = None) -> None:
        super().__init__(
            message="Failed to connect to LLM service (is Ollama running?)",
            original_error=original_error,
        )
        self.code = "LLM_CONNECTION_ERROR"


class LLMResponseParseError(LLMServiceError):
    """Raised when LLM response cannot be parsed."""

    def __init__(self, response_text: str) -> None:
        super().__init__(
            message=f"Failed to parse LLM response: {response_text[:100]}..."
        )
        self.code = "LLM_PARSE_ERROR"
        self.response_text = response_text


# =============================================================================
# Database Errors (500)
# =============================================================================


class DatabaseError(AppException):
    """Base exception for database errors."""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Database error: {message}",
            code="DATABASE_ERROR",
            status_code=500,
        )
        self.original_error = original_error
