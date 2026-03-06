# Action Item Extractor

A FastAPI-powered application that intelligently extracts actionable items from free-form notes using **dual extraction methods**: rule-based heuristics and LLM-powered semantic analysis.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Running the Application](#running-the-application)
- [Usage & API Reference](#usage--api-reference)
  - [Endpoints Overview](#endpoints-overview)
  - [Action Items Endpoints](#action-items-endpoints)
  - [Notes Endpoints](#notes-endpoints)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Architecture Decisions](#architecture-decisions)

---

## Project Overview

The **Action Item Extractor** transforms unstructured meeting notes, project updates, and general text into organized, actionable task lists. It addresses the common challenge of manually sifting through notes to identify tasks.

### Dual-Method Approach

The application offers two extraction strategies:

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Heuristic-Based** | `/action-items/extract` | Fast, rule-based pattern matching using regex and keyword detection. Works offline without external dependencies. |
| **LLM-Powered** | `/action-items/extract/llm` | Semantic understanding via Ollama models (e.g., `codellama:7b`). Better at understanding context and implicit tasks. |

#### Heuristic Extraction Patterns

The heuristic method identifies action items by:
- **Bullet points and numbered lists**: `- task`, `* task`, `1. task`
- **Checkbox markers**: `[ ]`, `[TODO]`
- **Keyword prefixes**: `TODO:`, `ACTION:`, `NEXT:`
- **Imperative verbs**: Sentences starting with `add`, `create`, `fix`, `implement`, `update`, `refactor`, etc.

#### LLM Extraction Capabilities

The LLM method provides:
- Context-aware task identification
- Multi-language support
- Understanding of implicit action items
- Automatic filtering of completed items vs. pending tasks

---

## Features

- ✅ **Dual extraction methods** — Choose between fast heuristics or intelligent LLM analysis
- ✅ **Note persistence** — Optionally save source text as searchable notes
- ✅ **Task management** — Mark action items as done/undone
- ✅ **RESTful API** — Clean JSON API with Pydantic validation
- ✅ **Interactive frontend** — Minimal HTML interface for quick testing
- ✅ **Auto-generated docs** — OpenAPI/Swagger documentation at `/docs`
- ✅ **Configurable** — Environment-based configuration for all settings
- ✅ **Graceful fallback** — LLM errors can optionally fall back to heuristics

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Poetry** (dependency management) or **pip**
- **Ollama** (for LLM-powered extraction) — [Install Ollama](https://ollama.com/download)

### Installation

#### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository and navigate to project root
cd modern-software-dev-assignments

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

#### Option 2: Using pip

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pydantic pydantic-settings ollama pytest
```

### Environment Configuration

Create a `.env` file in the `week2/` directory to customize settings:

```env
# Database Configuration
DB_PATH=./data/app.db

# LLM Configuration (Ollama)
LLM_MODEL=codellama:7b
LLM_TEMPERATURE=0.3
LLM_TIMEOUT_SECONDS=30.0

# Application Settings
APP_NAME=Action Item Extractor
DEBUG=false
LOG_LEVEL=INFO
```

#### Setting Up Ollama

1. Install Ollama from [ollama.com](https://ollama.com/download)
2. Pull the desired model:

```bash
ollama pull codellama:7b
```

3. Ensure Ollama is running:

```bash
ollama serve
```

> **Note**: The application will work without Ollama for heuristic extraction. LLM endpoints will return errors if Ollama is unavailable.

### Running the Application

Start the development server:

```bash
# From the project root directory
poetry run uvicorn week2.app.main:app --reload
```

Or if using pip:

```bash
uvicorn week2.app.main:app --reload
```

Access the application:
- **Frontend UI**: http://127.0.0.1:8000/
- **API Documentation (Swagger)**: http://127.0.0.1:8000/docs
- **API Documentation (ReDoc)**: http://127.0.0.1:8000/redoc

---

## Usage & API Reference

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/action-items/extract` | Extract action items using heuristics |
| `POST` | `/action-items/extract/llm` | Extract action items using LLM |
| `GET` | `/action-items` | List all action items |
| `POST` | `/action-items/{id}/done` | Mark action item as done/undone |
| `GET` | `/notes` | List all saved notes |
| `POST` | `/notes` | Create a new note |
| `GET` | `/notes/{id}` | Get a specific note |

---

### Action Items Endpoints

#### Extract Action Items (Heuristic)

Extracts action items using rule-based pattern matching.

**Endpoint**: `POST /action-items/extract`

**Request Body**:

```json
{
  "text": "Meeting notes:\n- [ ] Fix login bug\n- Implement API endpoint\nTODO: Write documentation",
  "save_note": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | — | The text to extract action items from |
| `save_note` | boolean | No | `false` | Whether to save the text as a note |

**Response** (201 Created):

```json
{
  "note_id": 1,
  "items": [
    { "id": 1, "text": "Fix login bug" },
    { "id": 2, "text": "Implement API endpoint" },
    { "id": 3, "text": "Write documentation" }
  ]
}
```

---

#### Extract Action Items (LLM)

Extracts action items using Ollama LLM for semantic understanding.

**Endpoint**: `POST /action-items/extract/llm`

**Request Body**: Same as heuristic extraction.

```json
{
  "text": "We should probably look into the performance issues. Also, remember to update the README.",
  "save_note": false
}
```

**Response** (200 OK):

```json
{
  "note_id": null,
  "items": [
    { "id": 4, "text": "Look into the performance issues" },
    { "id": 5, "text": "Update the README" }
  ]
}
```

**Error Response** (502 Bad Gateway — LLM unavailable):

```json
{
  "detail": "LLM error: Failed to connect to LLM service (is Ollama running?)",
  "code": "LLM_CONNECTION_ERROR"
}
```

---

#### List All Action Items

Retrieves all action items, optionally filtered by note ID.

**Endpoint**: `GET /action-items`

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `note_id` | integer | No | Filter by associated note ID |

**Response** (200 OK):

```json
[
  {
    "id": 1,
    "note_id": 1,
    "text": "Fix login bug",
    "done": false,
    "created_at": "2025-01-14 10:30:00"
  },
  {
    "id": 2,
    "note_id": 1,
    "text": "Implement API endpoint",
    "done": true,
    "created_at": "2025-01-14 10:30:00"
  }
]
```

---

#### Mark Action Item Done/Undone

Updates the completion status of an action item.

**Endpoint**: `POST /action-items/{action_item_id}/done`

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `action_item_id` | integer | The action item ID |

**Request Body**:

```json
{
  "done": true
}
```

**Response** (200 OK):

```json
{
  "id": 1,
  "done": true
}
```

---

### Notes Endpoints

#### List All Notes

Retrieves all saved notes, ordered by ID descending.

**Endpoint**: `GET /notes`

**Response** (200 OK):

```json
{
  "notes": [
    {
      "id": 2,
      "content": "Sprint planning notes...",
      "created_at": "2025-01-14 11:00:00"
    },
    {
      "id": 1,
      "content": "Meeting notes...",
      "created_at": "2025-01-14 10:30:00"
    }
  ]
}
```

---

#### Create a New Note

Creates a new note without extracting action items.

**Endpoint**: `POST /notes`

**Request Body**:

```json
{
  "content": "Important meeting notes to save for later."
}
```

**Response** (201 Created):

```json
{
  "id": 3,
  "content": "Important meeting notes to save for later.",
  "created_at": "2025-01-14 12:00:00"
}
```

---

#### Get a Specific Note

Retrieves a single note by its ID.

**Endpoint**: `GET /notes/{note_id}`

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `note_id` | integer | The note ID |

**Response** (200 OK):

```json
{
  "id": 1,
  "content": "Meeting notes from Monday...",
  "created_at": "2025-01-14 10:30:00"
}
```

**Error Response** (404 Not Found):

```json
{
  "detail": "Note with id '999' not found",
  "code": "NOTE_NOT_FOUND"
}
```

---

## Testing

The project includes a comprehensive test suite covering both extraction methods.

### Running Tests

Execute all tests from the project root:

```bash
# Using Poetry
poetry run pytest week2/tests/ -v

# Using pip
pytest week2/tests/ -v
```

### Test Coverage

Run tests with coverage report:

```bash
poetry run pytest week2/tests/ -v --cov=week2.app --cov-report=term-missing
```

### Test Categories

| Test Class | Description |
|------------|-------------|
| `TestExtractActionItemsHeuristic` | Tests for rule-based extraction patterns |
| `TestExtractActionItemsLLMIntegration` | Integration tests requiring Ollama |

### Sample Test Output

```
week2/tests/test_extract.py::TestExtractActionItemsHeuristic::test_extract_bullets_and_checkboxes PASSED
week2/tests/test_extract.py::TestExtractActionItemsHeuristic::test_extract_todo_prefixes PASSED
week2/tests/test_extract.py::TestExtractActionItemsHeuristic::test_empty_input_returns_empty_list PASSED
week2/tests/test_extract.py::TestExtractActionItemsHeuristic::test_deduplication PASSED
week2/tests/test_extract.py::TestExtractActionItemsLLMIntegration::test_llm_extracts_bullet_list PASSED
...
```

> **Note**: LLM integration tests require Ollama to be running with the configured model.

---

## Project Structure

```
week2/
├── app/                          # Application source code
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration (Pydantic Settings)
│   ├── db.py                     # Database layer with connection management
│   ├── exceptions.py             # Hierarchical custom exceptions
│   ├── main.py                   # FastAPI app with lifecycle management
│   ├── schemas.py                # Pydantic request/response models
│   ├── routers/                  # API route handlers
│   │   ├── __init__.py
│   │   ├── action_items.py       # Action items CRUD endpoints
│   │   └── notes.py              # Notes CRUD endpoints
│   └── services/                 # Business logic services
│       └── extract.py            # ⭐ Core extraction logic (heuristic + LLM)
├── frontend/                     # Static frontend files
│   └── index.html                # Single-page HTML interface
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_extract.py           # Unit & integration tests
├── data/                         # SQLite database storage (auto-created)
│   └── app.db
├── assignment.md                 # Assignment instructions
├── writeup.md                    # Development documentation
└── README.md                     # This file
```

### Key Files Explained

| File | Description |
|------|-------------|
| `app/services/extract.py` | **Core extraction service** — Contains both `extract_action_items()` (heuristic) and `extract_action_items_llm()` (LLM) functions with pattern matching, LLM prompt engineering, and response parsing. |
| `app/config.py` | **Configuration management** — Uses Pydantic Settings to load environment variables with type validation and sensible defaults. |
| `app/db.py` | **Database layer** — SQLite operations with context-managed connections, returning domain models instead of raw rows. |
| `app/schemas.py` | **API contracts** — Pydantic models for request validation and response serialization, ensuring type-safe API boundaries. |
| `app/exceptions.py` | **Error handling** — Hierarchical exception classes with HTTP status codes for consistent error responses. |
| `app/main.py` | **Application entry point** — FastAPI app setup with lifespan management, exception handlers, and router registration. |

---

## Architecture Decisions

### Why Dual Extraction Methods?

- **Heuristics**: Fast, deterministic, works offline — ideal for structured input with clear markers
- **LLM**: Context-aware, handles natural language — ideal for implicit tasks and varied formats

### Configuration via Environment

All settings are externalized to environment variables:
- Enables different configurations for dev/test/prod
- Secrets (API keys) stay out of source control
- Easy container deployment

### Pydantic for Validation

- Request/response schemas provide automatic validation
- Self-documenting API via OpenAPI generation
- Type safety throughout the application

### Graceful Error Handling

- Hierarchical exceptions map to HTTP status codes
- LLM failures can fall back to heuristics
- Clear error messages for debugging

---

## License

This project is part of the CS146 Modern Software Development course curriculum.
