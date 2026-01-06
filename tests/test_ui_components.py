"""Tests for ui/components.py module."""

from unittest.mock import MagicMock, patch

import pytest

from ui.components import CUSTOM_CSS, create_ui_components


def test_custom_css():
    """Test that CUSTOM_CSS is defined."""
    assert isinstance(CUSTOM_CSS, str)
    assert len(CUSTOM_CSS) > 0
    assert ".generating" in CUSTOM_CSS
    assert "h1" in CUSTOM_CSS


@patch("ui.components.gr")
def test_create_ui_components(mock_gr):
    """Test create_ui_components function."""
    # Create mock components
    mock_components = {
        "rag_enabled": MagicMock(),
        "search_type": MagicMock(),
        "doc_filter": MagicMock(),
        "query_rewriting": MagicMock(),
        "reranking": MagicMock(),
        "hybrid_alpha": MagicMock(),
        "chatbot": MagicMock(),
        "context_box": MagicMock(),
        "msg": MagicMock(),
        "submit": MagicMock(),
        "clear": MagicMock(),
        "search_col": MagicMock(),
        "filter_col": MagicMock(),
        "context_section": MagicMock(),
        "advanced_options": MagicMock(),
    }

    # Setup Blocks mock
    mock_blocks = MagicMock()
    mock_blocks.__enter__ = MagicMock(return_value=mock_blocks)
    mock_blocks.__exit__ = MagicMock(return_value=None)
    mock_gr.Blocks.return_value = mock_blocks

    # Setup context manager mocks
    def create_context_manager(name):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=None)
        return cm

    mock_gr.Row.return_value = create_context_manager("Row")
    mock_gr.Column.return_value = create_context_manager("Column")
    mock_gr.Accordion.return_value = create_context_manager("Accordion")

    # Setup component mocks
    mock_gr.Markdown.return_value = MagicMock()
    mock_gr.Checkbox.return_value = mock_components["rag_enabled"]
    mock_gr.Radio.return_value = mock_components["search_type"]
    mock_gr.Dropdown.return_value = mock_components["doc_filter"]
    mock_gr.Slider.return_value = mock_components["hybrid_alpha"]
    mock_gr.Chatbot.return_value = mock_components["chatbot"]
    mock_gr.Textbox.return_value = mock_components["msg"]
    mock_gr.Button.side_effect = [mock_components["submit"], mock_components["clear"]]
    mock_gr.Examples.return_value = MagicMock()

    # Test
    result = create_ui_components(["test1.pdf", "test2.pdf"])

    assert isinstance(result, dict)
    assert "demo" in result
    assert "rag_enabled" in result
    assert "search_type" in result
    assert "doc_filter" in result
    assert "chatbot" in result
    assert "msg" in result
    assert "submit" in result
    assert "clear" in result


def test_create_ui_components_empty_sources():
    """Test create_ui_components with empty sources."""
    with patch("ui.components.gr") as mock_gr:
        mock_blocks = MagicMock()
        mock_blocks.__enter__ = MagicMock(return_value=mock_blocks)
        mock_blocks.__exit__ = MagicMock(return_value=None)
        mock_gr.Blocks.return_value = mock_blocks

        # Mock all components
        for attr in ["Markdown", "Row", "Column", "Accordion", "Checkbox", "Radio", 
                     "Dropdown", "Slider", "Chatbot", "Textbox", "Button", "Examples"]:
            setattr(mock_gr, attr, MagicMock())

        # Mock context managers
        for cm in ["Row", "Column", "Accordion"]:
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=mock_cm)
            mock_cm.__exit__ = MagicMock(return_value=None)
            getattr(mock_gr, cm).return_value = mock_cm

        result = create_ui_components([])

        assert isinstance(result, dict)
        assert "doc_filter" in result

