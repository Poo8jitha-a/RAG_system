import os
import hashlib
from typing import List, Dict, Any

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
