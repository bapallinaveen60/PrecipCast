import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from a2wsgi import WSGIMiddleware

try:
    import spaces
    @spaces.GPU
    def zero_gpu_heartbeat():
        return "GPU Active"
    try:
        zero_gpu_heartbeat()
    except Exception:
        pass
except ImportError:
    pass

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
