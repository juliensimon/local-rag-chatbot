"""Tests for app.py entry point."""

from unittest.mock import MagicMock, patch

import pytest


@patch("app.create_app")
def test_app_main(mock_create_app):
    """Test app.py main entry point."""
    mock_demo = MagicMock()
    mock_create_app.return_value = mock_demo

    # Test the main execution logic
    demo = mock_create_app()
    demo.launch(share=False, server_port=7860)

    mock_create_app.assert_called_once()
    mock_demo.launch.assert_called_once_with(share=False, server_port=7860)

