# Week 2 Write-up
Tip: To preview this markdown file
- On Mac, press `Command (⌘) + Shift + V`
- On Windows/Linux, press `Ctrl + Shift + V`

## INSTRUCTIONS

Fill out all of the `TODO`s in this file.

## SUBMISSION DETAILS

Name: **TODO** \
SUNet ID: **TODO** \
Citations: **TODO**

This assignment took me about **TODO** hours to do. 


## YOUR RESPONSES
For each exercise, please include what prompts you used to generate the answer, in addition to the location of the generated response. Make sure to clearly add comments in your code documenting which parts are generated.

### Exercise 1: Scaffold a New Feature
Prompt: 
```
@week2/app/services/extract.py:34-66 
I have a extrract_action_items function to extract action items into todos using predefined heuristics. I want to impletement an LLM-powered alternative, extracct_action_items_llm，that utilizes Ollama to perform action item extraction via a large language model.

Let me outline the requirements. Please provide a detailed development plan instead of modifying the code directly.
1. You can refer to the code snippet that I provided.  @week1/k_shot_prompting.py:64-71 
2. I use ollama to load large language model in the project.
3. You should write properly USER_PROMPT and system_prompt when call large language model.
``` 

Generated Code Snippets:
```
@week2/app/routers/action_items.py:8 
from ..services.extract import extract_action_items, extract_action_items_llm
@week2/app/routers/action_items.py:24-25 
# items = extract_action_items(text)
items = extract_action_items_llm(text)
@week2/app/services/extract.py:96-246 
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
        # Skip empty lines and JSON syntax
        if not line or line in ['[', ']', ',']:
            continue
        # Remove quotes and commas from JSON-like formatting
        cleaned = re.sub(r'^[\s"\']+|[\s"\',]+$', '', line)
        if cleaned:
            extracted.append(cleaned)
    
    return extracted


def extract_action_items_llm(
    text: str,
    model: str = "codellama:7b",
    temperature: float = 0.3
) -> List[str]:
    """
    Extract action items from text using an LLM (Ollama).
    
    This is an LLM-powered alternative to the heuristic-based
    extract_action_items() function, providing better semantic
    understanding of action items.
    
    Args:
        text: The input text to extract action items from.
        model: The Ollama model to use (default: "mistral-nemo:12b").
        temperature: Generation temperature, lower = more deterministic (default: 0.3).
    
    Returns:
        A list of extracted action items (deduplicated, preserving order).
    """
    # Handle empty input
    if not text or not text.strip():
        return []
    
    try:
        # Call Ollama LLM
        response = chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ACTION_ITEMS},
                {"role": "user", "content": _build_user_prompt(text)},
            ],
            options={"temperature": temperature},
        )
        
        # Extract response content
        response_text = response.message.content.strip()
        print("===> response_text:", response_text)
        
        # Parse LLM response
        extracted = _parse_llm_response(response_text)
        print("===> extracted:", extracted)
        
    except Exception as e:
        # Fallback to heuristic method on error
        print(f"LLM extraction failed, falling back to heuristic method: {e}")
        return extract_action_items(text)
    
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for item in extracted:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(item)
    
    return unique


```

### Exercise 2: Add Unit Tests
Prompt: 
```
You did a perfect job. Now, Please write unit tests for  extract_action_items_llm covering multiple inputs(e.g., bullet lists, keyword-prefixed lines, empty input, etc.) in @week2/tests/test_extract.py 

Why not test the capabilty of the LLM?
``` 

Generated Code Snippets:
```

@week2/tests/test_extract.py:22-146 
class TestExtractActionItemsLLMIntegration:
    """
    Integration tests that call the real LLM.
    
    These tests verify:
    1. The LLM can understand and extract action items correctly
    2. The prompt engineering is effective
    3. The response parsing works with real LLM outputs
    
    Note: These tests require Ollama to be running with the target model loaded.
    """

    def test_llm_extracts_bullet_list(self):
        """Test that LLM correctly extracts items from a bullet list."""
        text = """
        Meeting notes:
        - Fix the login bug
        - Update API documentation
        - Schedule code review
        """
        
        items = extract_action_items_llm(text)
        
        # LLM should extract at least some items
        assert len(items) >= 2, f"Expected at least 2 items, got {items}"
        
        # Check for expected content (flexible matching)
        items_lower = [item.lower() for item in items]
        assert any("login" in item or "bug" in item for item in items_lower), \
            f"Expected 'login bug' related item in {items}"
        assert any("documentation" in item or "doc" in item for item in items_lower), \
            f"Expected 'documentation' related item in {items}"

    def test_llm_extracts_todo_keywords(self):
        """Test that LLM correctly extracts TODO-prefixed items."""
        text = """
        Project update:
        The project is going well overall.
        TODO: Review pull request #42
        ACTION: Send weekly status report
        We had a productive meeting yesterday.
        """
        
        items = extract_action_items_llm(text)
        
        assert len(items) >= 1, f"Expected at least 1 item, got {items}"
        items_lower = [item.lower() for item in items]
        assert any("review" in item or "pull request" in item for item in items_lower), \
            f"Expected 'review' or 'pull request' in {items}"

    def test_llm_returns_empty_for_no_actions(self):
        """Test that LLM returns empty list when no action items exist."""
        text = """
        The weather was beautiful today. 
        We had lunch at the new restaurant downtown.
        Everyone enjoyed the food.
        """
        
        items = extract_action_items_llm(text)
        
        # Should return empty or very few items
        assert len(items) <= 1, f"Expected 0-1 items for non-action text, got {items}"

    def test_llm_handles_checkbox_format(self):
        """Test that LLM extracts unchecked checkbox items."""
        text = """
        Sprint backlog:
        [x] Completed: Set up CI/CD pipeline
        [ ] Pending: Write unit tests for auth module
        [ ] Pending: Implement password reset feature
        [x] Completed: Update dependencies
        """
        
        items = extract_action_items_llm(text)
        
        # Should extract pending items, not completed ones
        assert len(items) >= 1, f"Expected at least 1 item, got {items}"
        items_lower = [item.lower() for item in items]
        # Should include pending tasks
        assert any("unit test" in item or "auth" in item for item in items_lower) or \
               any("password" in item or "reset" in item for item in items_lower), \
            f"Expected pending task items in {items}"

    def test_llm_extracts_imperative_sentences(self):
        """Test that LLM extracts imperative sentences as action items."""
        text = """
        Next steps for the team:
        Refactor the database connection pool.
        Implement rate limiting for the API.
        The current system is performing well.
        Document the new authentication flow.
        """
        
        items = extract_action_items_llm(text)
        
        assert len(items) >= 2, f"Expected at least 2 items, got {items}"
        items_lower = [item.lower() for item in items]
        assert any("refactor" in item or "database" in item for item in items_lower), \
            f"Expected 'refactor' related item in {items}"

    def test_llm_response_is_valid_list(self):
        """Test that LLM response is always a valid Python list."""
        text = "- Task 1\n- Task 2\n- Task 3"
        
        items = extract_action_items_llm(text)
        
        # Should always return a list
        assert isinstance(items, list), f"Expected list, got {type(items)}"
        # All items should be strings
        for item in items:
            assert isinstance(item, str), f"Expected string, got {type(item)}: {item}"

    def test_llm_handles_mixed_languages(self):
        """Test that LLM can handle mixed language content."""
        text = """
        会议记录:
        - 修复登录问题 (Fix login issue)
        - Update the README file
        - 完成单元测试
        """
        
        items = extract_action_items_llm(text)
        
        # Should extract items regardless of language
        assert len(items) >= 2, f"Expected at least 2 items from mixed language text, got {items}"

```

### Exercise 3: Refactor Existing Code for Clarity
Prompt: 
```
Act as a Senior Backend Engineer. Reivew @week2/app/services/extract.py and it's dependencies for architectural debt.

Performa a refactor if current implementation violates clean code principles, specifically focusing on:
1. API Contracts: Ensure schemas are well-defined and strictly typed(e.g., Pydantic models)
2. Database layer: Decouple business logic from DB queries; clean up redunant connections.
3. Lifecycle/Config: Centrailze configuration management and ensure proper startup/shutdown patterns.
4. Error Handling: Implement robust, hierarchical error handing instead of generic try/except blocks.

Provide a brief rationale for each major change.
``` 

Generated/Modified Code Snippets:
```
### new file structure
week2/app/
├── config.py       # NEW: Centralized settings
├── schemas.py      # NEW: Pydantic request/response models
├── exceptions.py   # NEW: Custom exception hierarchy
├── db.py           # REFACTORED: Connection management, domain models
├── main.py         # REFACTORED: Lifespan + exception handlers
├── routers/
│   ├── action_items.py  # REFACTORED: Uses Pydantic schemas
│   └── notes.py         # REFACTORED: Uses Pydantic schemas
└── services/
    └── extract.py       # REFACTORED: Logging, specific errors

schemas.py
# Before (action_items.py)
def extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()

# After
def extract(request: ExtractActionItemsRequest) -> ExtractActionItemsResponse:
    # Automatic validation, type safety, and OpenAPI docs

db.py
# Before
def get_note(note_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as connection:
        ...

# After  
def get_note(note_id: int) -> NoteModel:
    db = get_db_manager()
    with db.get_connection() as conn:
        ...
        return NoteModel(id=row["id"], content=row["content"], ...)

config.py, main.py
# config.py - Single source of truth
class Settings(BaseSettings):
    db_path: Path = ...
    llm_model: str = "codellama:7b"
    llm_temperature: float = 0.3
    log_level: str = "INFO"

# main.py - Proper lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    yield  # App runs
    reset_db_manager()  # Cleanup

exceptions.py
# Exception hierarchy
AppException (base)
├── ValidationError (400)
│   └── EmptyInputError
├── NotFoundError (404)
│   ├── NoteNotFoundError
│   └── ActionItemNotFoundError
└── ExternalServiceError (502)
    └── LLMServiceError
        ├── LLMConnectionError
        └── LLMResponseParseError

# Centralized handlers in main.py
@app.exception_handler(AppException)
async def app_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={...})
```


### Exercise 4: Use Agentic Mode to Automate a Small Task
Prompt: 
```
Modify  @week2/app/services/extract.py and corresponding frontend files to implement the following:
1. LLM Extaction Endpoint: Creat a new POST endpoint(e.g., /extract/llm) that triggers the LLM-powered extraction logic. In the frontend, add an "Extract LLM" button that call this endpoint and updates the UI with the results.
2. List Notes Endpoint: Create a new GET endpoint (e.g., /notes) to retrieve all stored notes from the database. In the frontend, add a "List Notes" button that fetchs all notes and render them in a list view.

Ensure all new endpoint follow the existing error handling and API Schema.

``` 

Generated Code Snippets:
```
week2/app/routers/action_items.py:61-92
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
    ...

week2/app/routers/notes.py:23-41
@router.get(
    "",
    response_model=NotesListResponse,
    summary="List all notes",
    description="Returns all stored notes ordered by ID descending",
)
def list_notes() -> NotesListResponse:
    """List all notes."""
    notes = db.list_notes()
    return NotesListResponse(...)

week2/app/schemas.py 
class NotesListResponse(BaseModel):
    """Response schema for listing all notes."""
    notes: List[NoteResponse] = Field(...)
```


### Exercise 5: Generate a README from the Codebase
Prompt: 
```
Analyze the directory structure and codebase in @week2 . Generate a comprehensive and professional README.md file in the root directory that includes.

1. Project Overview: A clear explanation of the Action Item Extractor's purpose, including the dual-method approach (Heuristics vs. LLM).

2. Getting Started: Step-by-step setup instructions (e.g., virtual environment, dependency installation via pip or poetry, and environment variable configuration like LLM API keys).

3. Usage & API Reference: A structured table or list of available endpoints, including the new /extract/llm and /notes routes. Document the expected request payloads and response formats.

4. Testing: Instructions on how to execute the test suite (e.g., pytest) and how to interpret the results.

5. Project Structure: A tree-style visualization of the folder hierarchy with brief descriptions of key files like extract.py.

Use proper Markdown formatting with code blocks, bold text for emphasis, and a clean layout.
``` 

Generated Code Snippets:
```
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

```


## SUBMISSION INSTRUCTIONS
1. Hit a `Command (⌘) + F` (or `Ctrl + F`) to find any remaining `TODO`s in this file. If no results are found, congratulations – you've completed all required fields. 
2. Make sure you have all changes pushed to your remote repository for grading.
3. Submit via Gradescope. 