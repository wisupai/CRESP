# cresp/core/models.py

"""Pydantic models for defining the structure of cresp.yaml configuration."""

from typing import Any

from pydantic import BaseModel


class Author(BaseModel):
    """Author information model"""

    name: str
    affiliation: str | None = None
    email: str | None = None
    orcid: str | None = None


class Computing(BaseModel):
    """Computing resource requirements model"""

    cpu: dict[str, Any] | None = None
    memory: dict[str, Any] | None = None
    gpu: dict[str, Any] | None = None
    estimated_runtime: str | None = None
    estimated_storage: str | None = None


class Environment(BaseModel):
    """Environment configuration model"""

    manager: str = "pixi"
    file: str = "pixi.toml"
    python_version: str | None = None


class ValidationRule(BaseModel):
    """Validation rule model"""

    field: str | None = None
    operator: str
    value: Any
    tolerance: float | None = None
    reference: str | None = None
    tolerance_absolute: float | None = None
    tolerance_relative: float | None = None
    similarity_threshold: float | None = None
    method: str | None = None


class ArtifactValidation(BaseModel):
    """Artifact validation model"""

    type: str = "strict"  # strict, weak
    rules: list[ValidationRule] | None = None


class Artifact(BaseModel):
    """Artifact model"""

    path: str
    description: str | None = None
    hash: str | None = None
    hash_method: str = "file"  # file, content, selective
    validation: ArtifactValidation | None = None


class Stage(BaseModel):
    """Experiment stage model"""

    id: str
    description: str | None = None
    dependencies: list[str] | None = None
    outputs: list[Artifact] | None = None
    code_handler: str  # Now required as execution_type is always 'code' implicitly
    parameters: dict[str, Any] | None = None


class ReproductionConfig(BaseModel):
    """Reproduction configuration model"""

    reproducibility_mode: str = "standard"  # strict, standard, tolerant
    random_seed: int | None = None
    comparison_methods: list[dict[str, Any]] | None = None


class Metadata(BaseModel):
    """Metadata model"""

    title: str
    authors: list[Author]
    description: str | None = None
    keywords: list[str] | None = None
    license: str | None = None
    repository: str | None = None
    created_date: str | None = None


class CrespConfigModel(BaseModel):
    """Complete CRESP configuration model"""

    version: str = "1.0"
    metadata: Metadata
    environment: Environment
    computing: Computing | None = None
    stages: list[Stage]
    reproduction: ReproductionConfig | None = None
