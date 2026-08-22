import os
import sys
import webbrowser
import time

SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from app import app
from wsgiref.simple_server import make_server

if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print("=" * 60)
    print("  PRECIPCAST PLATFORM — STARTING SERVER")
    print(f"  URL: {url}")
    print("=" * 60, flush=True)

    # Open browser automatically after a short delay
    try:
        webbrowser.open(url)
    except Exception:
        pass

    server = make_server("127.0.0.1", 5000, app)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPrecipCast Server Stopped.")
