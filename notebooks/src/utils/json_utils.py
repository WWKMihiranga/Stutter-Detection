import numpy as np
import torch

def make_json_serializable(obj):
    """Recursively convert numpy / torch types to native Python types"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.cpu().tolist()
    else:
        return obj
