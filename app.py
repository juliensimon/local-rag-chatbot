"""
Main entry point for the Gradio web application.

This module provides a simple entry point that delegates to the UI module.
"""

from ui.app import create_app

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False, server_port=7860)
