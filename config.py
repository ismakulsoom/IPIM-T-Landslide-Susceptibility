import yaml
import os

class Config:
    """Load configuration from YAML file."""
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
    
    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key):
        return self._config.get(key)

# Create a global config instance
config = Config()