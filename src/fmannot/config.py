import yaml
import os
from pathlib import Path

root_path = Path(__file__).parent.parent.parent

def load_config(config_path=os.path.join(root_path, "auth.yaml")):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Auth file not found at {config_path}. Please create it.")
    
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    return config