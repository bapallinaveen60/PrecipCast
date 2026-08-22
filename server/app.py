import os
import sys
import time
import traceback
from wsgiref.simple_server import make_server

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SERVER_DIR, ".."))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np

from pipeline import INSATDataReader, PrecipModelPipeline, MapOverlayGenerator

app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")
app.config['PROPAGATE_EXCEPTIONS'] = True
CORS(app)

DATA_DIR = os.path.join(PROJECT_ROOT, "insat_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "4_inputs_BT_only")

reader = INSATDataReader(data_dir=DATA_DIR)
_pipeline = None
PREDICTION_CACHE = {}

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Initializing PrecipCast ML Pipeline on first request...", flush=True)
        _pipeline = PrecipModelPipeline(models_dir=MODELS_DIR)
    return _pipeline

def get_prediction_for_file(filepath):
    if filepath not in PREDICTION_CACHE:
        print(f"Running inference for satellite file: {filepath}...", flush=True)
        t0 = time.time()
        pipeline_inst = get_pipeline()
        bt_dict = reader.read_bt_channels(filepath)
        results = pipeline_inst.predict_full_grid(bt_dict, target_size=(256, 256))
        print(f"Inference completed in {time.time()-t0:.2f}s!", flush=True)
        PREDICTION_CACHE[filepath] = results
    return PREDICTION_CACHE[filepath]

@app.errorhandler(Exception)
def handle_exception(e):
    print("API SERVER EXCEPTION OCCURRED:", flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    return jsonify({"status": "error", "message": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/")
def index():
    idx_path = os.path.join(PROJECT_ROOT, "index.html")
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            return f.read()
    return "PrecipCast Platform API Running."


@app.route("/api/timestamps", methods=["GET"])
def get_timestamps():
    timestamps = reader.list_timestamps()
    return jsonify({
        "status": "success",
        "count": len(timestamps),
        "timestamps": timestamps
    })


@app.route("/api/forecast", methods=["GET"])
def get_forecast_summary():
    filename = request.args.get("filename")
    timestamps = reader.list_timestamps()
    
    if not filename and timestamps:
        filename = timestamps[0]["filename"]
        
    file_info = reader.file_map.get(filename)
    if not file_info:
        return jsonify({"status": "error", "message": "File not found"}), 404
        
    preds = get_prediction_for_file(file_info["path"])
    rain_mask = preds["rain_mask"]
    strat_mask = preds["strat_mask"]
    class4_mask = preds["four_class_mask"]
    conf = preds["confidence"]
    
    total_pixels = rain_mask.size
    rain_pixels = int(np.sum(rain_mask == 1))
    rain_pct = round(float(rain_pixels / total_pixels) * 100, 1)
    
    strat_pixels = int(np.sum(strat_mask == 1))
    conv_pixels = int(np.sum(strat_mask == 2))
    
    class1_px = int(np.sum(class4_mask == 1)) # Stratiform
    class2_px = int(np.sum(class4_mask == 2)) # Deep Convective
    class3_px = int(np.sum(class4_mask == 3)) # Shallow Conv (isolated)
    class4_px = int(np.sum(class4_mask == 4)) # Shallow Conv (non-isolated)
    
    avg_conf = round(float(np.mean(conf)) * 100, 1)
    
    return jsonify({
        "status": "success",
        "timestamp": file_info,
        "summary": {
            "rain_coverage_pct": rain_pct,
            "rain_pixels": rain_pixels,
            "total_pixels": total_pixels,
            "stratiform_pixels": strat_pixels,
            "convective_pixels": conv_pixels,
            "deep_convective_pixels": class2_px,
            "shallow_conv_isolated_pixels": class3_px,
            "shallow_conv_non_isolated_pixels": class4_px,
            "average_confidence": avg_conf
        }
    })


@app.route("/api/overlay", methods=["GET"])
def get_overlay():
    filename = request.args.get("filename")
    layer = request.args.get("layer", "rain")
    
    timestamps = reader.list_timestamps()
    if not filename and timestamps:
        filename = timestamps[0]["filename"]
        
    file_info = reader.file_map.get(filename)
    if not file_info:
        return "File not found", 404
        
    preds = get_prediction_for_file(file_info["path"])
    
    if layer == "stratiform_convective":
        grid = preds["strat_mask"]
        bt_grid = None
    elif layer == "four_class":
        grid = preds["four_class_mask"]
        bt_grid = None
    elif layer == "tir1":
        grid = preds["rain_mask"]
        bt_grid = preds["tir1_bt"]
    else:
        grid = preds["rain_mask"]
        bt_grid = None
        
    png_path = MapOverlayGenerator.create_png_overlay(grid, layer_type=layer, bt_grid=bt_grid)
    return send_file(png_path, mimetype="image/png")


@app.route("/api/query", methods=["GET"])
def query_point():
    try:
        lat = float(request.args.get("lat", 20.5937))
        lon = float(request.args.get("lon", 78.9629))
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid lat/lon"}), 400
        
    filename = request.args.get("filename")
    timestamps = reader.list_timestamps()
    if not filename and timestamps:
        filename = timestamps[0]["filename"]
        
    file_info = reader.file_map.get(filename)
    if not file_info:
        return jsonify({"status": "error", "message": "File not found"}), 404
        
    preds = get_prediction_for_file(file_info["path"])
    
    lat_min, lat_max = 7.0, 37.0
    lon_min, lon_max = 68.0, 97.0
    
    clamped_lat = max(lat_min, min(lat_max, lat))
    clamped_lon = max(lon_min, min(lon_max, lon))
    
    grid_y = int((lat_max - clamped_lat) / (lat_max - lat_min) * 255)
    grid_x = int((clamped_lon - lon_min) / (lon_max - lon_min) * 255)
    
    grid_y = max(0, min(255, grid_y))
    grid_x = max(0, min(255, grid_x))
    
    rain_val = int(preds["rain_mask"][grid_y, grid_x])
    rain_prob = round(float(preds["rain_prob"][grid_y, grid_x]) * 100, 1)
    strat_val = int(preds["strat_mask"][grid_y, grid_x])
    class4_val = int(preds["four_class_mask"][grid_y, grid_x])
    conf_val = round(float(preds["confidence"][grid_y, grid_x]) * 100, 1)
    tir1_val = round(float(preds["tir1_bt"][grid_y, grid_x]), 1)
    
    rain_label = "Rain" if rain_val == 1 else "No Rain"
    
    strat_labels = {0: "No Rain", 1: "Stratiform Rain", 2: "Convective Rain"}
    strat_label = strat_labels.get(strat_val, "No Rain")
    
    class4_labels = {
        0: "No Rain",
        1: "Stratiform Rain",
        2: "Deep Convective Rain",
        3: "Shallow Convective (isolated)",
        4: "Shallow Convective (non-isolated)"
    }
    class4_label = class4_labels.get(class4_val, "No Rain")
    
    return jsonify({
        "status": "success",
        "location": {"lat": lat, "lon": lon},
        "timestamp": file_info,
        "prediction": {
            "rain": rain_label,
            "rain_val": rain_val,
            "rain_probability": rain_prob,
            "stratiform_convective": strat_label,
            "strat_val": strat_val,
            "four_class": class4_label,
            "four_class_val": class4_val,
            "confidence": conf_val,
            "tir1_bt": tir1_val
        }
    })


if __name__ == "__main__":
    print("PrecipCast WSGI Server listening on http://127.0.0.1:5000 ...", flush=True)
    server = make_server("127.0.0.1", 5000, app)
    server.serve_forever()
