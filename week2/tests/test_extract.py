"""
Tests for action item extraction service.

Tests cover both heuristic and LLM-based extraction methods.
"""
import pytest

from ..app.services.extract import extract_action_items, extract_action_items_llm


class TestExtractActionItemsHeuristic:
    """Tests for heuristic-based extraction."""

    def test_extract_bullets_and_checkboxes(self):
        """Test extraction of bullet points and checkbox items."""
        text = """
        Notes from meeting:
        - [ ] Set up database
        * implement API extract endpoint
        1. Write tests
        Some narrative sentence.
        """.strip()

        items = extract_action_items(text)
        assert "Set up database" in items
        assert "implement API extract endpoint" in items
        assert "Write tests" in items

    def test_extract_todo_prefixes(self):
        """Test extraction of TODO/action prefixed items."""
        text = """
        todo: Review the code
        action: Send the report
        next: Schedule meeting
        """

        items = extract_action_items(text)
        assert len(items) >= 3

    def test_empty_input_returns_empty_list(self):
        """Test that empty input returns empty list."""
        assert extract_action_items("") == []
        assert extract_action_items("   ") == []

    def test_deduplication(self):
        """Test that duplicate items are removed."""
        text = """
        - Fix the bug
        - Fix the Bug
        - fix the bug
        """

        items = extract_action_items(text)
        assert len(items) == 1


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

    def test_llm_fallback_on_empty_input(self):
        """Test that empty input returns empty list without raising."""
        items = extract_action_items_llm("")
        assert items == []

        items = extract_action_items_llm("   ")
        assert items == []
