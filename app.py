import os
import sys
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# Insert server directory into Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from app import app as flask_app

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

app = gr.mount_gradio_app(app=fastapi_app, blocks=demo, path="/")

if __name__ == "__main__":
    app.launch()
