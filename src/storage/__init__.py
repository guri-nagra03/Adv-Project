"""
Storage Layer Module

Provides local caching with SQLite and FHIR repository abstraction.

Components:
- SQLiteCache: Local database for caching predictions, stratifications, and metadata
- FHIRRepository: Unified interface for FHIR server + local cache operations
"""

from src.storage.sqlite_cache import SQLiteCache, init_database
from src.storage.fhir_repository import FHIRRepository

__all__ = [
    'SQLiteCache',
    'init_database',
    'FHIRRepository'
]
