import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from a2wsgi import WSGIMiddleware
except ImportError:
    from starlette.middleware.wsgi import WSGIMiddleware

# Insert server directory into Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(PROJECT_ROOT, "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from app import app as flask_app

# Create clean FastAPI application wrapper
app = FastAPI(title="PrecipCast API Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Flask WSGI application directly on root "/"
app.mount("/", WSGIMiddleware(flask_app))
