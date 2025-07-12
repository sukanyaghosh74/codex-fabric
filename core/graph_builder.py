"""
Graph Builder Module

This module handles the construction of knowledge graphs from parsed code,
managing relationships between code elements and storing them in Neo4j.
"""

import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import networkx as nx
from neo4j import GraphDatabase
from .parser import CodeNode, FileInfo, LanguageType


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]


class GraphBuilder:
    """
    Builds and manages knowledge graphs from parsed code.
    
    Converts CodeNode objects into graph structures and stores them
    in Neo4j for querying and analysis.
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.networkx_graph = nx.DiGraph()
        
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def build_graph(self, files: Dict[str, FileInfo]) -> Dict[str, Any]:
        """
        Build a knowledge graph from parsed files.
        
        Args:
            files: Dictionary mapping file paths to FileInfo objects
            
        Returns:
            Dictionary containing graph statistics and metadata
        """
        nodes = []
        relationships = []
        
        # Process each file
        for file_path, file_info in files.items():
            # Add file node
            file_node = self._create_file_node(file_info)
            nodes.append(file_node)
            
            # Process code nodes within the file
            for code_node in file_info.nodes:
                # Add code node
                node = self._create_code_node(code_node)
                nodes.append(node)
                
                # Add relationship between file and code node
                rel = GraphRelationship(
                    source_id=file_node.id,
                    target_id=node.id,
                    type="CONTAINS",
                    properties={}
                )
                relationships.append(rel)
                
                # Add relationships based on code node type
                if code_node.type == "class":
                    self._process_class_relationships(code_node, relationships)
                elif code_node.type == "function":
                    self._process_function_relationships(code_node, relationships)
        
        # Add import relationships
        self._add_import_relationships(files, relationships)
        
        # Add dependency relationships
        self._add_dependency_relationships(files, relationships)
        
        # Store in Neo4j
        self._store_in_neo4j(nodes, relationships)
        
        # Build NetworkX graph for analysis
        self._build_networkx_graph(nodes, relationships)
        
        return {
            "total_files": len(files),
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "languages": list(set(f.language.value for f in files.values())),
            "node_types": self._count_node_types(nodes),
            "relationship_types": self._count_relationship_types(relationships)
        }
    
    def _create_file_node(self, file_info: FileInfo) -> GraphNode:
        """Create a graph node for a file."""
        return GraphNode(
            id=f"file:{file_info.path}",
            labels=["File"],
            properties={
                "path": file_info.path,
                "language": file_info.language.value,
                "size": file_info.size,
                "lines": file_info.lines,
                "imports": file_info.imports,
                "exports": file_info.exports
            }
        )
    
    def _create_code_node(self, code_node: CodeNode) -> GraphNode:
        """Create a graph node for a code element."""
        labels = [code_node.type.capitalize()]
        if code_node.language:
            labels.append(code_node.language.value.capitalize())
        
        properties = {
            "name": code_node.name,
            "type": code_node.type,
            "language": code_node.language.value,
            "file_path": code_node.file_path,
            "line_start": code_node.line_start,
            "line_end": code_node.line_end,
            "column_start": code_node.column_start,
            "column_end": code_node.column_end,
            "content": code_node.content,
            "docstring": code_node.docstring,
            "imports": code_node.imports,
            "dependencies": code_node.dependencies
        }
        
        # Add metadata
        if code_node.metadata:
            properties.update(code_node.metadata)
        
        return GraphNode(
            id=code_node.id,
            labels=labels,
            properties=properties
        )
    
    def _process_class_relationships(self, code_node: CodeNode, relationships: List[GraphRelationship]):
        """Process relationships for class nodes."""
        # Add inheritance relationships
        if "bases" in code_node.metadata:
            for base in code_node.metadata["bases"]:
                rel = GraphRelationship(
                    source_id=code_node.id,
                    target_id=f"class:{base}",
                    type="INHERITS_FROM",
                    properties={}
                )
                relationships.append(rel)
        
        # Add method relationships
        if "methods" in code_node.metadata:
            for method in code_node.metadata["methods"]:
                rel = GraphRelationship(
                    source_id=code_node.id,
                    target_id=f"{code_node.file_path}:{method}",
                    type="HAS_METHOD",
                    properties={}
                )
                relationships.append(rel)
    
    def _process_function_relationships(self, code_node: CodeNode, relationships: List[GraphRelationship]):
        """Process relationships for function nodes."""
        # Add parameter relationships
        if "args" in code_node.metadata:
            for arg in code_node.metadata["args"]:
                rel = GraphRelationship(
                    source_id=code_node.id,
                    target_id=f"parameter:{arg}",
                    type="HAS_PARAMETER",
                    properties={"name": arg}
                )
                relationships.append(rel)
        
        # Add return type relationship
        if "returns" in code_node.metadata and code_node.metadata["returns"]:
            rel = GraphRelationship(
                source_id=code_node.id,
                target_id=f"type:{code_node.metadata['returns']}",
                type="RETURNS",
                properties={"type": code_node.metadata["returns"]}
            )
            relationships.append(rel)
    
    def _add_import_relationships(self, files: Dict[str, FileInfo], relationships: List[GraphRelationship]):
        """Add relationships based on import statements."""
        for file_path, file_info in files.items():
            for import_stmt in file_info.imports:
                # Try to find the imported module/file
                imported_file = self._find_imported_file(import_stmt, files)
                if imported_file:
                    rel = GraphRelationship(
                        source_id=f"file:{file_path}",
                        target_id=f"file:{imported_file}",
                        type="IMPORTS",
                        properties={"import_name": import_stmt}
                    )
                    relationships.append(rel)
    
    def _find_imported_file(self, import_stmt: str, files: Dict[str, FileInfo]) -> Optional[str]:
        """Find the file that corresponds to an import statement."""
        # This is a simplified implementation
        # In production, you'd need more sophisticated module resolution
        
        # Remove common prefixes
        clean_import = import_stmt.replace("from ", "").replace("import ", "")
        
        # Look for files that might match
        for file_path in files.keys():
            file_name = Path(file_path).stem
            if file_name == clean_import or file_name.endswith(f"/{clean_import}"):
                return file_path
        
        return None
    
    def _add_dependency_relationships(self, files: Dict[str, FileInfo], relationships: List[GraphRelationship]):
        """Add dependency relationships between files."""
        # This is a simplified implementation
        # In production, you'd analyze actual dependencies more thoroughly
        
        for file_path, file_info in files.items():
            for code_node in file_info.nodes:
                if code_node.dependencies:
                    for dep in code_node.dependencies:
                        # Try to find the dependency
                        dep_file = self._find_dependency_file(dep, files)
                        if dep_file:
                            rel = GraphRelationship(
                                source_id=f"file:{file_path}",
                                target_id=f"file:{dep_file}",
                                type="DEPENDS_ON",
                                properties={"dependency": dep}
                            )
                            relationships.append(rel)
    
    def _find_dependency_file(self, dependency: str, files: Dict[str, FileInfo]) -> Optional[str]:
        """Find a file that provides a dependency."""
        # Simplified implementation
        for file_path in files.keys():
            if dependency in file_path:
                return file_path
        return None
    
    def _store_in_neo4j(self, nodes: List[GraphNode], relationships: List[GraphRelationship]):
        """Store the graph in Neo4j database."""
        with self.driver.session() as session:
            # Clear existing data (optional)
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create nodes
            for node in nodes:
                labels_str = ":".join(node.labels)
                properties_str = ", ".join([f"{k}: ${k}" for k in node.properties.keys()])
                
                query = f"CREATE (n:{labels_str} {{id: $id, {properties_str}}})"
                session.run(query, id=node.id, **node.properties)
            
            # Create relationships
            for rel in relationships:
                query = """
                MATCH (a {id: $source_id}), (b {id: $target_id})
                CREATE (a)-[r:$rel_type $properties]->(b)
                """
                session.run(query, 
                           source_id=rel.source_id, 
                           target_id=rel.target_id,
                           rel_type=rel.type,
                           properties=rel.properties)
    
    def _build_networkx_graph(self, nodes: List[GraphNode], relationships: List[GraphRelationship]):
        """Build a NetworkX graph for analysis."""
        self.networkx_graph.clear()
        
        # Add nodes
        for node in nodes:
            self.networkx_graph.add_node(node.id, **node.properties)
        
        # Add edges
        for rel in relationships:
            self.networkx_graph.add_edge(rel.source_id, rel.target_id, 
                                       type=rel.type, **rel.properties)
    
    def _count_node_types(self, nodes: List[GraphNode]) -> Dict[str, int]:
        """Count the number of nodes by type."""
        counts = {}
        for node in nodes:
            for label in node.labels:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _count_relationship_types(self, relationships: List[GraphRelationship]) -> Dict[str, int]:
        """Count the number of relationships by type."""
        counts = {}
        for rel in relationships:
            counts[rel.type] = counts.get(rel.type, 0) + 1
        return counts
    
    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query on the Neo4j graph."""
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]
    
    def get_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate centrality metrics for the graph."""
        if not self.networkx_graph.nodes():
            return {}
        
        return {
            "degree_centrality": nx.degree_centrality(self.networkx_graph),
            "betweenness_centrality": nx.betweenness_centrality(self.networkx_graph),
            "closeness_centrality": nx.closeness_centrality(self.networkx_graph),
            "eigenvector_centrality": nx.eigenvector_centrality_numpy(self.networkx_graph)
        }
    
    def find_communities(self) -> Dict[str, List[str]]:
        """Find communities in the graph using Louvain method."""
        if not self.networkx_graph.nodes():
            return {}
        
        try:
            import community
            partition = community.best_partition(self.networkx_graph.to_undirected())
            
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            return communities
        except ImportError:
            # Fallback to connected components
            components = list(nx.connected_components(self.networkx_graph.to_undirected()))
            return {i: list(comp) for i, comp in enumerate(components)}
    
    def export_graph(self, format: str = "json") -> str:
        """Export the graph in various formats."""
        if format == "json":
            return json.dumps({
                "nodes": [asdict(node) for node in self._get_all_nodes()],
                "relationships": [asdict(rel) for rel in self._get_all_relationships()]
            }, indent=2)
        elif format == "gexf":
            nx.write_gexf(self.networkx_graph, "graph.gexf")
            return "Graph exported to graph.gexf"
        elif format == "graphml":
            nx.write_graphml(self.networkx_graph, "graph.graphml")
            return "Graph exported to graph.graphml"
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes from Neo4j."""
        query = "MATCH (n) RETURN n"
        with self.driver.session() as session:
            result = session.run(query)
            nodes = []
            for record in result:
                node_data = record["n"]
                nodes.append(GraphNode(
                    id=node_data["id"],
                    labels=list(node_data.labels),
                    properties=dict(node_data)
                ))
            return nodes
    
    def _get_all_relationships(self) -> List[GraphRelationship]:
        """Get all relationships from Neo4j."""
        query = "MATCH (a)-[r]->(b) RETURN a.id, b.id, type(r), properties(r)"
        with self.driver.session() as session:
            result = session.run(query)
            relationships = []
            for record in result:
                relationships.append(GraphRelationship(
                    source_id=record["a.id"],
                    target_id=record["b.id"],
                    type=record["type(r)"],
                    properties=record["properties(r)"]
                ))
            return relationships 