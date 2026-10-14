"""Searchable feature catalog: what data exists, and what it costs to use.

The LLM reasons over registry.json (per-column descriptions, temporal
scope, serving class, coverage). search_datasets() is the discovery API.
"""

from .search import load_catalog_dict, load_registry, search_datasets

__all__ = ["load_catalog_dict", "load_registry", "search_datasets"]
