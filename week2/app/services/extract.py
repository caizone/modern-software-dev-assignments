"""
Action item extraction service with heuristic and LLM-based methods.

Rationale:
- Proper logging instead of print statements
- Specific exception handling with clear error types
- Configuration via centralized Settings
- Clean separation between extraction strategies
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from ollama import chat, ResponseError

from ..config import get_settings
from ..exceptions import (
    EmptyInputError,
    LLMConnectionError,
    LLMResponseParseError,
    LLMServiceError,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Heuristic Patterns
# =============================================================================

BULLET_PREFIX_PATTERN = re.compile(r"^\s*([-*•]|\d+\.)\s+")
KEYWORD_PREFIXES = (
    "todo:",
    "action:",
    "next:",
)

IMPERATIVE_STARTERS = frozenset({
    "add",
    "create",
    "implement",
    "fix",
    "update",
    "write",
    "check",
    "verify",
    "refactor",
    "document",
    "design",
    "investigate",
})


# =============================================================================
# Heuristic Extraction
# =============================================================================


def _is_action_line(line: str) -> bool:
    """Check if a line looks like an action item."""
    stripped = line.strip().lower()
    if not stripped:
        return False
    if BULLET_PREFIX_PATTERN.match(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in KEYWORD_PREFIXES):
        return True
    if "[ ]" in stripped or "[todo]" in stripped:
        return True
    return False


def _looks_imperative(sentence: str) -> bool:
    """Check if a sentence starts with an imperative verb."""
    words = re.findall(r"[A-Za-z']+", sentence)
    if not words:
        return False
    return words[0].lower() in IMPERATIVE_STARTERS


def _deduplicate_items(items: List[str]) -> List[str]:
    """Deduplicate items while preserving order."""
    seen: set[str] = set()
    unique: List[str] = []
    for item in items:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(item)
    return unique


def extract_action_items(text: str) -> List[str]:
    """
    Extract action items using heuristic pattern matching.
    
    This method identifies action items by:
    1. Bullet points and numbered lists
    2. TODO/Action/Next prefixes
    3. Checkbox markers ([ ], [todo])
    4. Imperative sentence structures (as fallback)
    
    Args:
        text: The input text to extract action items from
        
    Returns:
        List of extracted action items (deduplicated)
    """
    lines = text.splitlines()
    extracted: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_action_line(line):
            cleaned = BULLET_PREFIX_PATTERN.sub("", line)
            cleaned = cleaned.strip()
            cleaned = cleaned.removeprefix("[ ]").strip()
            cleaned = cleaned.removeprefix("[todo]").strip()
            extracted.append(cleaned)

    # Fallback: if nothing matched, try imperative sentences
    if not extracted:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            s = sentence.strip()
            if s and _looks_imperative(s):
                extracted.append(s)

    return _deduplicate_items(extracted)


# =============================================================================
# LLM-powered Extraction
# =============================================================================

SYSTEM_PROMPT_ACTION_ITEMS = """
You are an expert task extraction assistant. Your job is to analyze text and extract actionable items (todos/tasks).

Rules for extraction:
1. Only extract items that represent concrete actions to be done
2. Ignore general statements, descriptions, or completed items
3. Look for patterns like:
   - Bullet points or numbered lists with tasks
   - Sentences starting with action verbs (add, create, fix, implement, update, write, check, verify, refactor, document, design, investigate, etc.)
   - Items marked with [ ], [TODO], or similar markers
   - Phrases starting with "todo:", "action:", "next:"
4. Clean up the extracted items by removing markers like "- ", "* ", "[ ]", "TODO:", etc.
5. Each action item should be a clear, standalone task

Output format:
- Return ONLY a JSON array of strings
- Each string is one action item
- If no action items found, return an empty array: []
- Do not include any explanation or extra text

Example 1:
Input: "Meeting notes: - Fix login bug\n- [ ] Update documentation\nThe weather is nice today."
Output: ["Fix login bug", "Update documentation"]

Example 2:
Input: "TODO: Review the PR\nAction: Send weekly report\nThe project is on track."
Output: ["Review the PR", "Send weekly report"]

Example 3:
Input: "It was a productive day. We discussed the roadmap."
Output: []
"""


def _build_user_prompt(text: str) -> str:
    """Build the user prompt for action item extraction."""
    return f"""Extract all action items from the following text. Return ONLY a JSON array of strings.

Text:
\"\"\"
{text}
\"\"\"

Action items (JSON array):"""


def _parse_llm_response(response_text: str) -> List[str]:
    """
    Parse the LLM response to extract action items.
    
    Handles various response formats:
    1. Clean JSON array: ["item1", "item2"]
    2. JSON array with surrounding text
    3. Fallback: line-by-line parsing
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        List of extracted action items
        
    Raises:
        LLMResponseParseError: If response cannot be parsed
    """
    response_text = response_text.strip()

    # Try direct JSON parsing first
    try:
        result = json.loads(response_text)
        if isinstance(result, list):
            return [str(item).strip() for item in result if item]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from response using regex
    json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, list):
                return [str(item).strip() for item in result if item]
        except json.JSONDecodeError:
            pass

    # Fallback: parse line by line (for malformed responses)
    lines = response_text.splitlines()
    extracted = []
    for line in lines:
        line = line.strip()
        if not line or line in ['[', ']', ',']:
            continue
        cleaned = re.sub(r'^[\s"\']+|[\s"\',]+$', '', line)
        if cleaned:
            extracted.append(cleaned)

    if not extracted:
        logger.warning(f"Could not parse LLM response: {response_text[:200]}")

    return extracted


def extract_action_items_llm(
    text: str,
    model: str | None = None,
    temperature: float | None = None,
    fallback_on_error: bool = True,
) -> List[str]:
    """
    Extract action items from text using an LLM (Ollama).
    
    This is an LLM-powered alternative to the heuristic-based
    extract_action_items() function, providing better semantic
    understanding of action items.
    
    Args:
        text: The input text to extract action items from
        model: The Ollama model to use (default from settings)
        temperature: Generation temperature (default from settings)
        fallback_on_error: If True, fall back to heuristic method on LLM failure
        
    Returns:
        List of extracted action items (deduplicated)
        
    Raises:
        EmptyInputError: If text is empty
        LLMServiceError: If LLM call fails and fallback_on_error is False
    """
    # Validate input
    if not text or not text.strip():
        if fallback_on_error:
            return []
        raise EmptyInputError("text")

    # Get settings with defaults
    settings = get_settings()
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature

    try:
        logger.debug(f"Calling LLM model={model} temperature={temperature}")

        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ACTION_ITEMS},
                {"role": "user", "content": _build_user_prompt(text)},
            ],
            options={"temperature": temperature},
        )

        response_text = response.message.content.strip()
        logger.debug(f"LLM response: {response_text[:200]}...")

        extracted = _parse_llm_response(response_text)
        logger.debug(f"Extracted {len(extracted)} items")

        return _deduplicate_items(extracted)

    except ConnectionError as e:
        logger.error(f"LLM connection failed: {e}")
        if fallback_on_error:
            logger.info("Falling back to heuristic extraction")
            return extract_action_items(text)
        raise LLMConnectionError(original_error=e) from e

    except ResponseError as e:
        logger.error(f"LLM response error: {e}")
        if fallback_on_error:
            logger.info("Falling back to heuristic extraction")
            return extract_action_items(text)
        raise LLMServiceError(
            message=str(e),
            original_error=e,
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error during LLM extraction: {e}")
        if fallback_on_error:
            logger.info("Falling back to heuristic extraction")
            return extract_action_items(text)
        raise LLMServiceError(
            message=f"Unexpected error: {e}",
            original_error=e,
        ) from e
