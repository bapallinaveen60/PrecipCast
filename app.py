import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# Safe ZeroGPU decorator fallback
if has_spaces:
    @spaces.GPU
    def init_gpu():
        pass
    try:
        init_gpu()
    except Exception:
        pass

# Insert server directory into Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from app import app as flask_app

# Use a2wsgi or Starlette WSGIMiddleware for mounting Flask in FastAPI
try:
    from a2wsgi import WSGIMiddleware
except ImportError:
    from starlette.middleware.wsgi import WSGIMiddleware

# Create FastAPI wrapper for Flask WSGI
fastapi_app = FastAPI(title="PrecipCast API Engine")
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Flask app using ASGI WSGI Middleware
fastapi_app.mount("/api", WSGIMiddleware(flask_app))
fastapi_app.mount("/static", WSGIMiddleware(flask_app))

demo = gr.Blocks(title="PrecipCast API Engine")
with demo:
    gr.Markdown("# 🌧️ PRECIPCAST — INSAT-3R Deep Learning Precipitation Platform API")
    gr.Markdown("Flask API routes (`/api/timestamps`, `/api/forecast`, `/api/overlay`, `/api/query`) are mounted and live!")

# Mount Gradio and export app for Hugging Face Spaces process runner
app = gr.mount_gradio_app(app=fastapi_app, blocks=demo, path="/gradio")
