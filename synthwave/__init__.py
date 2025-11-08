import os
from importlib.resources import files

PACKAGE_ROOT = files("synthwave")
VESSEL_CACHE_DIR = os.path.join(PACKAGE_ROOT, "vessel_caches")
os.makedirs(VESSEL_CACHE_DIR, exist_ok=True)
