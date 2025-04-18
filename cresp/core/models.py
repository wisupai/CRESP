# cresp/core/models.py

"""Pydantic models for defining the structure of cresp.yaml configuration."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError, validator


class Author(BaseModel):
    """Author information model"""

    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


class Computing(BaseModel):
    """Computing resource requirements model"""

    cpu: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None
    gpu: Optional[Dict[str, Any]] = None
    estimated_runtime: Optional[str] = None
    estimated_storage: Optional[str] = None


class Environment(BaseModel):
    """Environment configuration model"""

    manager: str = "pixi"
    file: str = "pixi.toml"
    python_version: Optional[str] = None


class ValidationRule(BaseModel):
    """Validation rule model"""

    field: Optional[str] = None
    operator: str
    value: Any
    tolerance: Optional[float] = None
    reference: Optional[str] = None
    tolerance_absolute: Optional[float] = None
    tolerance_relative: Optional[float] = None
    similarity_threshold: Optional[float] = None
    method: Optional[str] = None


class ArtifactValidation(BaseModel):
    """Artifact validation model"""

    type: str = "strict"  # strict, weak
    rules: Optional[List[ValidationRule]] = None


class Artifact(BaseModel):
    """Artifact model"""

    path: str
    description: Optional[str] = None
    hash: Optional[str] = None
    hash_method: str = "file"  # file, content, selective
    validation: Optional[ArtifactValidation] = None


class Stage(BaseModel):
    """Experiment stage model"""

    id: str
    description: Optional[str] = None
    dependencies: Optional[List[str]] = None
    outputs: Optional[List[Artifact]] = None
    code_handler: str  # Now required as execution_type is always 'code' implicitly
    parameters: Optional[Dict[str, Any]] = None


class ReproductionConfig(BaseModel):
    """Reproduction configuration model"""

    reproducibility_mode: str = "standard"  # strict, standard, tolerant
    random_seed: Optional[int] = None
    comparison_methods: Optional[List[Dict[str, Any]]] = None


class Metadata(BaseModel):
    """Metadata model"""

    title: str
    authors: List[Author]
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    license: Optional[str] = None
    repository: Optional[str] = None
    created_date: Optional[str] = None


class CrespConfigModel(BaseModel):
    """Complete CRESP configuration model"""

    version: str = "1.0"
    metadata: Metadata
    environment: Environment
    computing: Optional[Computing] = None
    stages: List[Stage]
    reproduction: Optional[ReproductionConfig] = None
