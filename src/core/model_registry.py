"""
Model Registry Module

Provides model versioning and management:
- Model artifact storage
- Version tracking
- Model metadata
- A/B testing support
- Model rollback
"""

import logging
import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Model metadata."""
    name: str
    version: str
    created_at: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None


class ModelRegistry:
    """Registry for managing model versions."""
    
    def __init__(self, models_dir: str = "models/registry"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.models_dir / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load model index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
        return {'models': {}}
    
    def _save_index(self):
        """Save model index to disk."""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _get_model_path(self, name: str, version: str) -> Path:
        """Get path for model directory."""
        return self.models_dir / name / version
    
    def register_model(
        self,
        name: str,
        version: str,
        model_data: Any,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a new model version.
        
        Args:
            name: Model name
            version: Model version
            model_data: Model data to save
            metrics: Model performance metrics
            parameters: Model parameters
            description: Model description
            tags: Model tags
            
        Returns:
            True if successful
        """
        model_path = self._get_model_path(name, version)
        model_path.mkdir(parents=True, exist_ok=True)
        
        metadata = ModelMetadata(
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            status=ModelStatus.READY.value,
            metrics=metrics or {},
            parameters=parameters or {},
            description=description,
            tags=tags or [],
        )
        
        try:
            metadata_path = model_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            import joblib
            model_file = model_path / "model.joblib"
            joblib.dump(model_data, model_file)
            
            model_hash = hashlib.md5(open(model_file, 'rb').read()).hexdigest()
            
            if name not in self._index['models']:
                self._index['models'][name] = {}
            
            self._index['models'][name][version] = {
                'path': str(model_path),
                'hash': model_hash,
                'created_at': metadata.created_at,
                'status': metadata.status,
            }
            
            self._save_index()
            
            logger.info(f"Registered model {name}:{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            shutil.rmtree(model_path, ignore_errors=True)
            return False
    
    def get_model(self, name: str, version: str) -> Optional[Any]:
        """
        Load a model from registry.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Model data or None
        """
        model_path = self._get_model_path(name, version)
        model_file = model_path / "model.joblib"
        
        if not model_file.exists():
            logger.warning(f"Model {name}:{version} not found")
            return None
        
        try:
            import joblib
            return joblib.load(model_file)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def get_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        model_path = self._get_model_path(name, version)
        metadata_file = model_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return ModelMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def list_models(self, name: Optional[str] = None) -> List[Dict]:
        """
        List all registered models.
        
        Args:
            name: Optional filter by model name
            
        Returns:
            List of model info
        """
        results = []
        
        for model_name, versions in self._index['models'].items():
            if name and model_name != name:
                continue
            
            for version, info in versions.items():
                metadata = self.get_metadata(model_name, version)
                results.append({
                    'name': model_name,
                    'version': version,
                    'status': info.get('status', 'unknown'),
                    'created_at': info.get('created_at', ''),
                    'hash': info.get('hash', ''),
                    'description': metadata.description if metadata else '',
                    'metrics': metadata.metrics if metadata else {},
                })
        
        return sorted(results, key=lambda x: x['created_at'], reverse=True)
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of a model."""
        if name not in self._index['models']:
            return None
        
        versions = list(self._index['models'][name].keys())
        if not versions:
            return None
        
        return sorted(versions, reverse=True)[0]
    
    def set_deployed(self, name: str, version: str) -> bool:
        """Mark a model version as deployed."""
        model_path = self._get_model_path(name, version)
        metadata_file = model_path / "metadata.json"
        
        if not metadata_file.exists():
            return False
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['status'] = ModelStatus.DEPLOYED.value
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._index['models'][name][version]['status'] = ModelStatus.DEPLOYED.value
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to set deployed status: {e}")
            return False
    
    def archive_model(self, name: str, version: str) -> bool:
        """Archive a model version."""
        model_path = self._get_model_path(name, version)
        archive_path = model_path.parent / f"{version}_archived"
        
        try:
            shutil.move(str(model_path), str(archive_path))
            
            if version in self._index['models'][name]:
                del self._index['models'][name][version]
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to archive model: {e}")
            return False
    
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version."""
        model_path = self._get_model_path(name, version)
        
        try:
            shutil.rmtree(model_path, ignore_errors=True)
            
            if name in self._index['models'] and version in self._index['models'][name]:
                del self._index['models'][name][version]
                self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False


registry = ModelRegistry()
