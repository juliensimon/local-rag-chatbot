"""
Main entry point for the Gradio web application.

This module provides a simple entry point that delegates to the UI module.
"""

import gradio as gr

from ui.app import create_app
from ui.components import CUSTOM_CSS

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False, server_port=7860, css=CUSTOM_CSS, theme=gr.themes.Soft())
