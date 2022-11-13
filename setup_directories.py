"""
Initialize the data directory structure used throughout the rest of the scripts.
Directory names are specified in the config.json configuration file.
"""
from pathlib import Path
import utils

deriv_dir = utils.config["derivatives_directory"]
zips_dir = Path(deriv_dir) / "zips"
data_directories = [deriv_dir, zips_dir]

for directory in data_directories:
    dir_path = Path(directory)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=False)