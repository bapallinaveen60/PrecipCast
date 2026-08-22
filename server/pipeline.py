import os
import glob
import re
import time
import zipfile
import json
import io
import h5py
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class INSATDataReader:
    """Reads INSAT-3R HDF5 satellite observation files."""
    def __init__(self, data_dir="insat_data"):
        self.data_dir = data_dir
        self.file_map = self._scan_files()

    def _scan_files(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
        file_map = {}
        for f in files:
            fname = os.path.basename(f)
            match = re.search(r'(\d{2}[A-Z]{3}\d{4})_(\d{4})', fname)
            if match:
                date_str, time_str = match.group(1), match.group(2)
                day = date_str[:2]
                mon = date_str[2:5].capitalize()
                yr = date_str[5:]
                hh = time_str[:2]
                mm = time_str[2:]
                fmt_time = f"{day} {mon} {yr} {hh}:{mm} IST"
                iso_time = f"{yr}-{self._month_to_num(mon)}-{day}T{hh}:{mm}:00"
                file_map[fname] = {
                    "path": f,
                    "filename": fname,
                    "date": f"{day} {mon} {yr}",
                    "time": f"{hh}:{mm} IST",
                    "display": fmt_time,
                    "iso": iso_time
                }
        return file_map

    def _month_to_num(self, mon):
        months = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
                  "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
        return months.get(mon, "01")

    def list_timestamps(self):
        return [v for k, v in self.file_map.items()]

    def read_bt_channels(self, filepath):
        with h5py.File(filepath, 'r') as f:
            t1_lut = f['IMG_TIR1_TEMP'][:]
            t2_lut = f['IMG_TIR2_TEMP'][:]
            mir_lut = f['IMG_MIR_TEMP'][:]
            wv_lut = f['IMG_WV_TEMP'][:]
            
            c_t1 = np.clip(f['IMG_TIR1'][0], 0, 1023)
            c_t2 = np.clip(f['IMG_TIR2'][0], 0, 1023)
            c_mir = np.clip(f['IMG_MIR'][0], 0, 1023)
            c_wv = np.clip(f['IMG_WV'][0], 0, 1023)
            
            return {
                "TIR1": t1_lut[c_t1],
                "TIR2": t2_lut[c_t2],
                "WV": wv_lut[c_wv],
                "MIR": mir_lut[c_mir]
            }


class KerasToPyTorchModel(nn.Module):
    """Dynamically converts Keras Functional models to PyTorch execution graph."""
    def __init__(self, keras_path):
        super().__init__()
        self.keras_path = keras_path
        
        with zipfile.ZipFile(keras_path, 'r') as z:
            self.cfg = json.loads(z.read('config.json').decode('utf-8'))['config']
            self.w_bytes = z.read('model.weights.h5')
            
        self.layers_dict = nn.ModuleDict()
        self.weights_map = {}
        
        with h5py.File(io.BytesIO(self.w_bytes), 'r') as f:
            if 'layers' in f:
                for k in f['layers'].keys():
                    grp = f['layers'][k]
                    if 'vars' in grp:
                        self.weights_map[k] = [grp['vars'][vk][:] for vk in sorted(grp['vars'].keys(), key=lambda x: int(x))]

        self._build_layers()

    def _build_layers(self):
        counters = {'Conv2D': 0, 'Conv2DTranspose': 0, 'SeparableConv2D': 0, 'LayerNormalization': 0}
        type_prefix = {'Conv2D': 'conv2d', 'Conv2DTranspose': 'conv2d_transpose', 'SeparableConv2D': 'separable_conv2d', 'LayerNormalization': 'layer_normalization'}
        
        for l in self.cfg['layers']:
            lname = l['name']
            cname = l['class_name']
            lcfg = l.get('config', {})
            
            if cname in counters:
                idx = counters[cname]
                counters[cname] += 1
                prefix = type_prefix[cname]
                hdf5_key = prefix if idx == 0 else f"{prefix}_{idx}"
                
                if hdf5_key in self.weights_map:
                    w_list = self.weights_map[hdf5_key]
                    
                    if cname == 'Conv2D':
                        kernel = torch.from_numpy(w_list[0]).permute(3, 2, 0, 1)
                        bias = torch.from_numpy(w_list[1]) if len(w_list) > 1 else None
                        outC, inC, kH, kW = kernel.shape
                        padding = lcfg.get('padding', 'same')
                        pad = (kH // 2, kW // 2) if padding == 'same' else 0
                        conv = nn.Conv2d(inC, outC, kernel_size=(kH, kW), padding=pad, bias=(bias is not None))
                        conv.weight.data = kernel
                        if bias is not None: conv.bias.data = bias
                        self.layers_dict[lname] = conv
                        
                    elif cname == 'Conv2DTranspose':
                        kernel = torch.from_numpy(w_list[0]).permute(3, 2, 0, 1)
                        bias = torch.from_numpy(w_list[1]) if len(w_list) > 1 else None
                        inC, outC, kH, kW = kernel.shape
                        stride = lcfg.get('strides', [2, 2])
                        out_pad = (stride[0] - 1, stride[1] - 1) if (kH % 2 == 1) else (0, 0)
                        conv_t = nn.ConvTranspose2d(inC, outC, kernel_size=(kH, kW), stride=tuple(stride), padding=(kH//2, kW//2), output_padding=out_pad, bias=(bias is not None))
                        conv_t.weight.data = kernel
                        if bias is not None: conv_t.bias.data = bias
                        self.layers_dict[lname] = conv_t

                    elif cname == 'SeparableConv2D':
                        dw_k = torch.from_numpy(w_list[0]).permute(2, 3, 0, 1)
                        pw_k = torch.from_numpy(w_list[1]).permute(3, 2, 0, 1)
                        bias = torch.from_numpy(w_list[2]) if len(w_list) > 2 else None
                        
                        inC = dw_k.shape[0]
                        outC = pw_k.shape[0]
                        kH, kW = dw_k.shape[2], dw_k.shape[3]
                        
                        class SepConv(nn.Module):
                            def __init__(self, dw, pw, b):
                                super().__init__()
                                self.dw = nn.Conv2d(inC, inC, kernel_size=(kH, kW), padding=(kH//2, kW//2), groups=inC, bias=False)
                                self.pw = nn.Conv2d(inC, outC, kernel_size=1, bias=(b is not None))
                                self.dw.weight.data = dw
                                self.pw.weight.data = pw
                                if b is not None: self.pw.bias.data = b
                            def forward(self, x): return self.pw(self.dw(x))
                                
                        self.layers_dict[lname] = SepConv(dw_k, pw_k, bias)
                        
                    elif cname == 'LayerNormalization':
                        gamma = torch.from_numpy(w_list[0])
                        beta = torch.from_numpy(w_list[1])
                        
                        class CustomLN(nn.Module):
                            def __init__(self, g, b, eps):
                                super().__init__()
                                self.gamma = nn.Parameter(g)
                                self.beta = nn.Parameter(b)
                                self.eps = eps
                            def forward(self, x):
                                x_perm = x.permute(0, 2, 3, 1)
                                out = F.layer_norm(x_perm, (x_perm.shape[-1],), self.gamma, self.beta, self.eps)
                                return out.permute(0, 3, 1, 2)
                                
                        self.layers_dict[lname] = CustomLN(gamma, beta, lcfg.get('epsilon', 1e-3))

    def forward(self, x):
        tensor_map = {}
        in_layer_name = self.cfg['input_layers'][0][0]
        tensor_map[in_layer_name] = x
        
        for l in self.cfg['layers']:
            lname = l['name']
            cname = l['class_name']
            lcfg = l.get('config', {})
            in_nodes = l.get('inbound_nodes', [])
            
            if cname == 'InputLayer': continue
                
            inputs = []
            if in_nodes and len(in_nodes) > 0:
                for node in in_nodes[0]:
                    src_name = node[0]
                    if src_name in tensor_map:
                        inputs.append(tensor_map[src_name])
                        
            if not inputs: continue
            inp = inputs[0]
            
            if lname in self.layers_dict:
                out = self.layers_dict[lname](inp)
            elif cname == 'MaxPooling2D':
                pool_size = lcfg.get('pool_size', [2, 2])
                strides = lcfg.get('strides', pool_size)
                out = F.max_pool2d(inp, kernel_size=tuple(pool_size), stride=tuple(strides))
            elif cname == 'UpSampling2D':
                size = lcfg.get('size', [2, 2])
                out = F.interpolate(inp, scale_factor=tuple(size), mode='nearest')
            elif cname == 'Activation':
                act = lcfg.get('activation', 'relu')
                if act == 'relu': out = F.relu(inp)
                elif act == 'sigmoid': out = torch.sigmoid(inp)
                elif act == 'softmax': out = F.softmax(inp, dim=1)
                else: out = inp
            elif cname == 'Concatenate':
                target_h, target_w = inputs[0].shape[2], inputs[0].shape[3]
                aligned_inputs = []
                for t in inputs:
                    if t.shape[2] != target_h or t.shape[3] != target_w:
                        t = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    aligned_inputs.append(t)
                out = torch.cat(aligned_inputs, dim=1)
            elif cname == 'Add':
                target_shape = inputs[0].shape
                aligned_inputs = []
                for t in inputs:
                    if t.shape != target_shape:
                        t = F.interpolate(t, size=(target_shape[2], target_shape[3]), mode='bilinear', align_corners=False)
                    aligned_inputs.append(t)
                out = aligned_inputs[0]
                for i in range(1, len(aligned_inputs)): out = out + aligned_inputs[i]
            elif cname == 'Multiply':
                t1, t2 = inputs[0], inputs[1]
                if t2.shape[2:] != t1.shape[2:]:
                    t2 = F.interpolate(t2, size=(t1.shape[2], t1.shape[3]), mode='bilinear', align_corners=False)
                out = t1 * t2
            else:
                out = inp
                
            tensor_map[lname] = out
            
        out_layer_name = self.cfg['output_layers'][0][0]
        return tensor_map[out_layer_name]


class PrecipModelPipeline:
    """Manages ML model inference for Rain vs No-Rain and secondary classification."""
    def __init__(self, models_dir="4_inputs_BT_only"):
        self.models_dir = models_dir
        self.model_rain = None
        self.model_strat = None
        self.model_4class = None
        self._load_models()

    def _load_models(self):
        print("Initializing PrecipCast PyTorch Deep Learning Engine...")
        t0 = time.time()
        self.model_rain = KerasToPyTorchModel(os.path.join(self.models_dir, "rain_norain_BT_New.keras"))
        self.model_strat = KerasToPyTorchModel(os.path.join(self.models_dir, "stratiforn_convective_BT_only.keras"))
        self.model_4class = KerasToPyTorchModel(os.path.join(self.models_dir, "fourclasses_layernorm_BT_only.keras"))
        
        self.model_rain.eval()
        self.model_strat.eval()
        self.model_4class.eval()
        print(f"All 3 Deep Learning models initialized successfully in {time.time()-t0:.2f}s!")

    def predict_patch(self, patch_4ch_np):
        p_norm = np.clip(patch_4ch_np, 180.0, 330.0)
        p_norm = (p_norm - 180.0) / 150.0
        
        x_tensor = torch.from_numpy(p_norm).permute(0, 3, 1, 2).float()
        
        with torch.no_grad():
            out_rain = F.softmax(self.model_rain(x_tensor), dim=1)
            out_strat = F.softmax(self.model_strat(x_tensor), dim=1)
            out_4class = F.softmax(self.model_4class(x_tensor), dim=1)
            
            if out_rain.shape[2:] != (128, 128):
                out_rain = F.interpolate(out_rain, size=(128, 128), mode='bilinear', align_corners=False)
                out_strat = F.interpolate(out_strat, size=(128, 128), mode='bilinear', align_corners=False)
                out_4class = F.interpolate(out_4class, size=(128, 128), mode='bilinear', align_corners=False)
                
            rain_prob_np = out_rain[:, 1, :, :].cpu().numpy()
            rain_mask_np = (rain_prob_np > 0.50).astype(np.uint8)
            
            strat_probs_np = out_strat.cpu().numpy()
            class4_probs_np = out_4class.cpu().numpy()
            
            strat_mask_np = torch.argmax(out_strat, dim=1).cpu().numpy()
            class4_mask_np = torch.argmax(out_4class, dim=1).cpu().numpy()
            conf_np = torch.max(out_rain, dim=1).values.cpu().numpy()
            
        final_strat = np.where(rain_mask_np == 1, strat_mask_np + 1, 0)
        final_4class = np.where(rain_mask_np == 1, class4_mask_np + 1, 0)
        
        return {
            "rain_mask": rain_mask_np,
            "rain_prob": rain_prob_np,
            "strat_mask": final_strat,
            "strat_probs": strat_probs_np,
            "four_class_mask": final_4class,
            "four_class_probs": class4_probs_np,
            "confidence": conf_np
        }

    def predict_full_grid(self, bt_dict, target_size=(256, 256)):
        y_min, y_max = 686, 1411
        x_min, x_max = 1138, 1923
        
        t1 = bt_dict["TIR1"][y_min:y_max, x_min:x_max]
        t2 = bt_dict["TIR2"][y_min:y_max, x_min:x_max]
        wv = bt_dict["WV"][y_min:y_max, x_min:x_max]
        mir = bt_dict["MIR"][y_min:y_max, x_min:x_max]
        
        def resize_arr(arr):
            img = Image.fromarray(arr)
            return np.array(img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR))
        
        r_t1 = resize_arr(t1)
        r_t2 = resize_arr(t2)
        r_wv = resize_arr(wv)
        r_mir = resize_arr(mir)
        
        full_rain = np.zeros(target_size, dtype=np.uint8)
        full_rain_prob = np.zeros(target_size, dtype=np.float32)
        full_strat = np.zeros(target_size, dtype=np.uint8)
        full_strat_probs = np.zeros((2,) + target_size, dtype=np.float32)
        full_4class = np.zeros(target_size, dtype=np.uint8)
        full_4class_probs = np.zeros((4,) + target_size, dtype=np.float32)
        full_conf = np.zeros(target_size, dtype=np.float32)
        
        patch_size = 128
        h, w = target_size
        
        patches = []
        coords = []
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                
                p_t1 = np.zeros((128, 128), dtype=np.float32)
                p_t2 = np.zeros((128, 128), dtype=np.float32)
                p_wv = np.zeros((128, 128), dtype=np.float32)
                p_mir = np.zeros((128, 128), dtype=np.float32)
                
                slice_h = y_end - y
                slice_w = x_end - x
                
                p_t1[:slice_h, :slice_w] = r_t1[y:y_end, x:x_end]
                p_t2[:slice_h, :slice_w] = r_t2[y:y_end, x:x_end]
                p_wv[:slice_h, :slice_w] = r_wv[y:y_end, x:x_end]
                p_mir[:slice_h, :slice_w] = r_mir[y:y_end, x:x_end]
                
                patch_4ch = np.stack([p_t1, p_t2, p_wv, p_mir], axis=-1)
                patches.append(patch_4ch)
                coords.append((y, x, slice_h, slice_w))
                
        if patches:
            batch = np.array(patches, dtype=np.float32)
            results = self.predict_patch(batch)
            
            for idx, (y, x, sh, sw) in enumerate(coords):
                full_rain[y:y+sh, x:x+sw] = results["rain_mask"][idx, :sh, :sw]
                full_rain_prob[y:y+sh, x:x+sw] = results["rain_prob"][idx, :sh, :sw]
                full_strat[y:y+sh, x:x+sw] = results["strat_mask"][idx, :sh, :sw]
                full_strat_probs[:, y:y+sh, x:x+sw] = results["strat_probs"][idx, :, :sh, :sw]
                full_4class[y:y+sh, x:x+sw] = results["four_class_mask"][idx, :sh, :sw]
                full_4class_probs[:, y:y+sh, x:x+sw] = results["four_class_probs"][idx, :, :sh, :sw]
                full_conf[y:y+sh, x:x+sw] = results["confidence"][idx, :sh, :sw]

        return {
            "rain_mask": full_rain,
            "rain_prob": full_rain_prob,
            "strat_mask": full_strat,
            "strat_probs": full_strat_probs,
            "four_class_mask": full_4class,
            "four_class_probs": full_4class_probs,
            "confidence": full_conf,
            "tir1_bt": r_t1
        }


class MapOverlayGenerator:
    """Generates transparent PNG map overlays for Leaflet visualization matching prediction_rain.ipynb."""
    PALETTES = {
        "rain": {
            0: (0, 0, 0, 0),         # Transparent No Rain
            1: (33, 150, 243, 210)    # Blue Rain (#2196F3)
        },
        "stratiform_convective": {
            0: (0, 0, 0, 0),         # Transparent No Rain
            1: (58, 134, 255, 220),   # Stratiform Rain (#3A86FF Royal Blue)
            2: (211, 47, 47, 230)     # Convective Rain (#D32F2F Crimson Red)
        },
        "four_class": {
            0: (0, 0, 0, 0),         # Transparent No Rain
            1: (58, 134, 255, 220),   # Stratiform Rain (#3A86FF Royal Blue)
            2: (211, 47, 47, 230),    # Deep Convective Rain (#D32F2F Crimson Red)
            3: (255, 112, 67, 220),   # Shallow Convective (isolated) (#FF7043 Deep Orange)
            4: (255, 193, 7, 220)     # Shallow Convective (non-isolated) (#FFC107 Amber Yellow)
        }
    }

    @staticmethod
    def create_png_overlay(grid, layer_type="rain", bt_grid=None):
        h, w = grid.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        if layer_type == "tir1" and bt_grid is not None:
            norm_bt = np.clip((bt_grid - 180.0) / (320.0 - 180.0), 0.0, 1.0)
            cmap = cm.get_cmap('RdYlBu_r')
            colored = (cmap(norm_bt) * 255).astype(np.uint8)
            colored[..., 3] = 200 # Semi-transparent overlay
            rgba = colored
        else:
            palette = MapOverlayGenerator.PALETTES.get(layer_type, MapOverlayGenerator.PALETTES["rain"])
            for val, color in palette.items():
                mask = (grid == val)
                rgba[mask] = color
            
        img = Image.fromarray(rgba, mode="RGBA")
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(temp_fd)
        img.save(temp_path, format="PNG")
        return temp_path
