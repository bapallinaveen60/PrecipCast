import os
import sys
import gradio as gr

# Insert server directory into Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from app import app as flask_app

# Mount Flask REST API inside Gradio on 100% Free Gradio SDK
demo = gr.Blocks(title="PrecipCast API")
with demo:
    gr.Markdown("# 🌧️ PRECIPCAST — INSAT-3R Deep Learning Precipitation API Engine")
    gr.Markdown("Flask API routes (`/api/timestamps`, `/api/forecast`, `/api/overlay`, `/api/query`) are mounted and live!")

app = gr.mount_gradio_app(app=flask_app, blocks=demo, path="/")

if __name__ == "__main__":
    app.launch()
