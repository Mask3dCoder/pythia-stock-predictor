"""
Configuration Presets Module

Provides configuration presets for different trading strategies.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, List


class PresetManager:
    """Manages configuration presets."""
    
    def __init__(self, presets_dir: Optional[str] = None):
        if presets_dir is None:
            presets_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
        
        self.presets_dir = Path(presets_dir)
        self._presets = None
    
    def _load_presets(self) -> Dict:
        """Load presets from file."""
        if self._presets is not None:
            return self._presets
        
        presets_file = self.presets_dir / 'presets.yaml'
        
        if presets_file.exists():
            try:
                with open(presets_file, 'r') as f:
                    data = yaml.safe_load(f)
                    self._presets = data.get('presets', {})
                    return self._presets
            except Exception:
                pass
        
        return {}
    
    def list_presets(self) -> List[str]:
        """List available presets."""
        presets = self._load_presets()
        return list(presets.keys())
    
    def get_preset(self, name: str) -> Optional[Dict]:
        """Get a specific preset."""
        presets = self._load_presets()
        return presets.get(name)
    
    def get_preset_info(self, name: str) -> Optional[Dict]:
        """Get preset information."""
        preset = self.get_preset(name)
        if preset:
            return {
                'name': preset.get('name', name),
                'description': preset.get('description', ''),
            }
        return None


def load_preset(name: str) -> Dict:
    """Load a configuration preset."""
    manager = PresetManager()
    preset = manager.get_preset(name)
    
    if preset is None:
        raise ValueError(f"Unknown preset: {name}")
    
    return preset


def list_presets() -> List[str]:
    """List available presets."""
    manager = PresetManager()
    return manager.list_presets()


DEFAULT_PRESETS = {
    'aggressive': {
        'name': 'Aggressive',
        'models': {'default': 'lstm'},
    },
    'conservative': {
        'name': 'Conservative', 
        'models': {'default': 'arima'},
    },
    'balanced': {
        'name': 'Balanced',
        'models': {'default': 'ensemble'},
    },
}
