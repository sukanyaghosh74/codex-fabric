"""
Codex Fabric Core Module

This module contains the core functionality for parsing codebases,
building knowledge graphs, and managing the overall system.
"""

from .parser import CodeParser
from .graph_builder import GraphBuilder
from .signal_tracer import SignalTracer
from .embedding_service import EmbeddingService

__version__ = "0.1.0"
__all__ = ["CodeParser", "GraphBuilder", "SignalTracer", "EmbeddingService"] 