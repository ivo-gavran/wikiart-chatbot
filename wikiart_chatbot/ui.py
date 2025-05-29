"""Gradio UI implementation for the WikiArt Chatbot."""

import gradio as gr
from typing import List, Dict, Tuple, Optional

from .chatbot import WikiArtChatbot
from .config import Config

def create_ui(config: Optional[Config] = None) -> gr.Blocks:
    """Create and configure the Gradio interface.
    
    Args:
        config: Optional configuration object. If not provided, uses default values.
        
    Returns:
        The configured Gradio interface.
    """
    chatbot = WikiArtChatbot(config)
    
    with gr.Blocks(title="WikiArt Chatbot", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 👨‍🎨 WikiArt Chatbot")
        gr.Markdown("Ask me anything about art! I can help you learn about famous artworks, artists, styles, and more.")
        
        chatbot_interface = gr.Chatbot(
            height=600,
            show_label=False,
            elem_id="chatbot",
            type="messages"
        )
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                container=False
            )
            submit = gr.Button("Send", variant="primary")
        
        def respond(message: str, chat_history: List[Dict]) -> Tuple[str, List[Dict]]:
            return chatbot.process_message(message, chat_history)
        
        submit.click(respond, [msg, chatbot_interface], [msg, chatbot_interface])
        msg.submit(respond, [msg, chatbot_interface], [msg, chatbot_interface])
    
    return interface 