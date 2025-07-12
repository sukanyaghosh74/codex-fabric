"""
Embedding Service Module

This module handles the generation of vector embeddings for code elements,
enabling semantic search and similarity analysis in the knowledge graph.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str
    dimension: int
    max_length: int
    batch_size: int
    device: str = "cpu"


@dataclass
class CodeEmbedding:
    """Represents a code element with its embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    model_name: str


class EmbeddingService:
    """
    Service for generating and managing code embeddings.
    
    Supports multiple embedding models including OpenAI, SentenceTransformers,
    and custom models for code-specific embeddings.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 qdrant_url: str = "http://localhost:6333",
                 openai_api_key: Optional[str] = None):
        self.config = EmbeddingConfig(
            model_name=model_name,
            dimension=384,  # Default for all-MiniLM-L6-v2
            max_length=512,
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.model = None
        self.openai_api_key = openai_api_key
        
        self._initialize_model()
        self._initialize_vector_store()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            if self.config.model_name.startswith("sentence-transformers/"):
                self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
                self.config.dimension = self.model.get_sentence_embedding_dimension()
            elif self.config.model_name.startswith("openai/"):
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key required for OpenAI models")
                openai.api_key = self.openai_api_key
                # OpenAI models are handled differently
            else:
                # Try to load as a custom model
                self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
                self.config.dimension = self.model.get_sentence_embedding_dimension()
        
        except Exception as e:
            print(f"Error initializing model {self.config.model_name}: {e}")
            # Fallback to default model
            self.config.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
            self.config.dimension = self.model.get_sentence_embedding_dimension()
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections()
            collection_name = "code_embeddings"
            
            if collection_name not in [col.name for col in collections.collections]:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created vector collection: {collection_name}")
        
        except Exception as e:
            print(f"Error initializing vector store: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if self.config.model_name.startswith("openai/"):
            return self._generate_openai_embedding(text)
        else:
            return self._generate_local_embedding(text)
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return []
    
    def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Truncate text if too long
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]
            
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        
        except Exception as e:
            print(f"Error generating local embedding: {e}")
            return []
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if self.config.model_name.startswith("openai/"):
            return self._generate_openai_batch_embeddings(texts)
        else:
            return self._generate_local_batch_embeddings(texts)
    
    def _generate_openai_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using OpenAI API."""
        embeddings = []
        
        # OpenAI has rate limits, so we process in smaller batches
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating OpenAI batch embeddings: {e}")
                # Add empty embeddings for failed items
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    def _generate_local_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using local model."""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Truncate texts if too long
            truncated_texts = [text[:self.config.max_length] for text in texts]
            
            embeddings = self.model.encode(
                truncated_texts,
                batch_size=self.config.batch_size,
                convert_to_tensor=False
            )
            
            return embeddings.tolist()
        
        except Exception as e:
            print(f"Error generating local batch embeddings: {e}")
            return [[] for _ in texts]
    
    def embed_code_elements(self, code_nodes: List[Any]) -> List[CodeEmbedding]:
        """
        Generate embeddings for code elements.
        
        Args:
            code_nodes: List of code nodes from the parser
            
        Returns:
            List of CodeEmbedding objects
        """
        embeddings = []
        
        for node in code_nodes:
            # Create text representation for embedding
            text = self._create_embedding_text(node)
            
            # Generate embedding
            embedding_vector = self.generate_embedding(text)
            
            if embedding_vector:
                code_embedding = CodeEmbedding(
                    id=node.id,
                    content=text,
                    embedding=embedding_vector,
                    metadata={
                        "name": node.name,
                        "type": node.type,
                        "language": node.language.value,
                        "file_path": node.file_path,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                        "docstring": node.docstring
                    },
                    model_name=self.config.model_name
                )
                embeddings.append(code_embedding)
        
        return embeddings
    
    def _create_embedding_text(self, node: Any) -> str:
        """Create text representation for embedding generation."""
        parts = []
        
        # Add function/class name
        parts.append(f"{node.type}: {node.name}")
        
        # Add docstring if available
        if node.docstring:
            parts.append(f"Documentation: {node.docstring}")
        
        # Add content (truncated)
        content = node.content[:1000]  # Limit content length
        parts.append(f"Code: {content}")
        
        # Add metadata if available
        if hasattr(node, 'metadata') and node.metadata:
            if 'args' in node.metadata:
                parts.append(f"Parameters: {', '.join(node.metadata['args'])}")
            if 'returns' in node.metadata and node.metadata['returns']:
                parts.append(f"Returns: {node.metadata['returns']}")
            if 'bases' in node.metadata:
                parts.append(f"Inherits from: {', '.join(node.metadata['bases'])}")
        
        return " | ".join(parts)
    
    def store_embeddings(self, embeddings: List[CodeEmbedding]):
        """Store embeddings in the vector database."""
        try:
            points = []
            
            for embedding in embeddings:
                point = PointStruct(
                    id=hash(embedding.id) % (2**63),  # Qdrant requires int64 IDs
                    vector=embedding.embedding,
                    payload={
                        "id": embedding.id,
                        "content": embedding.content,
                        "metadata": embedding.metadata,
                        "model_name": embedding.model_name
                    }
                )
                points.append(point)
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name="code_embeddings",
                    points=batch
                )
            
            print(f"Stored {len(embeddings)} embeddings in vector database")
        
        except Exception as e:
            print(f"Error storing embeddings: {e}")
    
    def search_similar(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar code elements.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar code elements with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                return []
            
            # Search in vector database
            results = self.qdrant_client.search(
                collection_name="code_embeddings",
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.payload["id"],
                    "score": result.score,
                    "content": result.payload["content"],
                    "metadata": result.payload["metadata"]
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"Error searching similar code: {e}")
            return []
    
    def find_similar_functions(self, function_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find functions similar to a given function.
        
        Args:
            function_id: ID of the function to find similar ones for
            top_k: Number of similar functions to return
            
        Returns:
            List of similar functions with similarity scores
        """
        try:
            # Get the function's embedding
            results = self.qdrant_client.search(
                collection_name="code_embeddings",
                query_filter={"must": [{"key": "id", "match": {"value": function_id}}]},
                limit=1
            )
            
            if not results:
                return []
            
            function_embedding = results[0].vector
            
            # Find similar functions
            similar_results = self.qdrant_client.search(
                collection_name="code_embeddings",
                query_vector=function_embedding,
                query_filter={"must": [{"key": "metadata.type", "match": {"value": "function"}}]},
                limit=top_k + 1  # +1 to exclude the original function
            )
            
            # Filter out the original function and format results
            formatted_results = []
            for result in similar_results:
                if result.payload["id"] != function_id:
                    formatted_results.append({
                        "id": result.payload["id"],
                        "score": result.score,
                        "name": result.payload["metadata"]["name"],
                        "file_path": result.payload["metadata"]["file_path"],
                        "content": result.payload["content"]
                    })
            
            return formatted_results[:top_k]
        
        except Exception as e:
            print(f"Error finding similar functions: {e}")
            return []
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        try:
            collection_info = self.qdrant_client.get_collection("code_embeddings")
            
            return {
                "total_vectors": collection_info.vectors_count,
                "dimension": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "model_name": self.config.model_name
            }
        
        except Exception as e:
            print(f"Error getting embedding statistics: {e}")
            return {}
    
    def export_embeddings(self, output_file: str, embeddings: List[CodeEmbedding]):
        """Export embeddings to a JSON file."""
        try:
            export_data = []
            
            for embedding in embeddings:
                export_data.append({
                    "id": embedding.id,
                    "content": embedding.content,
                    "embedding": embedding.embedding,
                    "metadata": embedding.metadata,
                    "model_name": embedding.model_name
                })
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Exported {len(embeddings)} embeddings to {output_file}")
        
        except Exception as e:
            print(f"Error exporting embeddings: {e}")
    
    def clear_embeddings(self):
        """Clear all embeddings from the vector database."""
        try:
            self.qdrant_client.delete_collection("code_embeddings")
            self._initialize_vector_store()
            print("Cleared all embeddings from vector database")
        
        except Exception as e:
            print(f"Error clearing embeddings: {e}") 