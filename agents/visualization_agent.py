"""
Visualization Agent Module

This agent specializes in generating visualizations of the knowledge graph,
dependency diagrams, and code architecture visualizations.
"""

from typing import Dict, List, Any, Optional
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class VisualizationAgent:
    """Agent specialized in generating code visualizations and diagrams."""
    
    def __init__(self, graph_builder=None):
        self.graph_builder = graph_builder
    
    def generate_dependency_graph(self, output_path: str = "dependency_graph.png") -> Dict[str, Any]:
        """Generate a dependency graph visualization."""
        if not self.graph_builder:
            return {"error": "Graph builder not available"}
        
        try:
            # Get dependency relationships
            query = """
            MATCH (a)-[:DEPENDS_ON]->(b)
            RETURN a.name as source, b.name as target, a.file_path as source_file, b.file_path as target_file
            """
            results = self.graph_builder.query_graph(query)
            
            if not results:
                return {"error": "No dependencies found"}
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for result in results:
                source = result.get("source", "Unknown")
                target = result.get("target", "Unknown")
                source_file = result.get("source_file", "")
                target_file = result.get("target_file", "")
                
                G.add_node(source, file=source_file)
                G.add_node(target, file=target_file)
                G.add_edge(source, target)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
            
            plt.title("Code Dependency Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "output_path": output_path,
                "nodes": len(G.nodes()),
                "edges": len(G.edges()),
                "dependencies": results
            }
        
        except Exception as e:
            return {"error": f"Failed to generate dependency graph: {str(e)}"}
    
    def generate_architecture_diagram(self, output_path: str = "architecture_diagram.png") -> Dict[str, Any]:
        """Generate an architecture diagram showing layers and components."""
        if not self.graph_builder:
            return {"error": "Graph builder not available"}
        
        try:
            # Get file structure and categorize by type
            query = """
            MATCH (f:File)
            RETURN f.path, f.language, f.size
            """
            results = self.graph_builder.query_graph(query)
            
            if not results:
                return {"error": "No files found"}
            
            # Categorize files by layer/type
            layers = {
                "controllers": [],
                "services": [],
                "models": [],
                "utils": [],
                "tests": [],
                "config": []
            }
            
            for result in results:
                path = result.get("f.path", "")
                if "controller" in path.lower() or "api" in path.lower():
                    layers["controllers"].append(path)
                elif "service" in path.lower() or "business" in path.lower():
                    layers["services"].append(path)
                elif "model" in path.lower() or "entity" in path.lower():
                    layers["models"].append(path)
                elif "test" in path.lower():
                    layers["tests"].append(path)
                elif "config" in path.lower() or "settings" in path.lower():
                    layers["config"].append(path)
                else:
                    layers["utils"].append(path)
            
            # Create layered architecture diagram
            fig, ax = plt.subplots(figsize=(12, 8))
            
            layer_colors = {
                "controllers": "#FF6B6B",
                "services": "#4ECDC4",
                "models": "#45B7D1",
                "utils": "#96CEB4",
                "tests": "#FFEAA7",
                "config": "#DDA0DD"
            }
            
            y_positions = {
                "controllers": 0.9,
                "services": 0.75,
                "models": 0.6,
                "utils": 0.45,
                "tests": 0.3,
                "config": 0.15
            }
            
            for layer_name, files in layers.items():
                if files:
                    y = y_positions[layer_name]
                    color = layer_colors[layer_name]
                    
                    # Draw layer box
                    rect = mpatches.Rectangle((0.1, y-0.05), 0.8, 0.08, 
                                           facecolor=color, alpha=0.7, edgecolor='black')
                    ax.add_patch(rect)
                    
                    # Add layer label
                    ax.text(0.5, y, layer_name.upper(), ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                    
                    # Add file count
                    ax.text(0.95, y, f"{len(files)} files", ha='right', va='center', 
                           fontsize=10)
            
            # Add arrows between layers
            arrow_props = dict(arrowstyle='->', color='gray', lw=2)
            
            # Controllers -> Services
            ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.85), 
                       arrowprops=arrow_props)
            
            # Services -> Models
            ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.7), 
                       arrowprops=arrow_props)
            
            # Models -> Utils
            ax.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.55), 
                       arrowprops=arrow_props)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Code Architecture Layers", fontsize=16, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "output_path": output_path,
                "layers": {k: len(v) for k, v in layers.items()},
                "total_files": sum(len(v) for v in layers.values())
            }
        
        except Exception as e:
            return {"error": f"Failed to generate architecture diagram: {str(e)}"}
    
    def generate_class_hierarchy(self, output_path: str = "class_hierarchy.png") -> Dict[str, Any]:
        """Generate a class hierarchy diagram."""
        if not self.graph_builder:
            return {"error": "Graph builder not available"}
        
        try:
            # Get class inheritance relationships
            query = """
            MATCH (c:Class)-[:INHERITS_FROM]->(parent)
            RETURN c.name as child, parent.name as parent
            """
            results = self.graph_builder.query_graph(query)
            
            if not results:
                return {"error": "No class hierarchy found"}
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for result in results:
                child = result.get("child", "Unknown")
                parent = result.get("parent", "Unknown")
                G.add_node(child)
                G.add_node(parent)
                G.add_edge(parent, child)  # Parent -> Child for hierarchy
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=1500)
            nx.draw_networkx_labels(G, pos, font_size=10, fontweight='bold')
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='darkgreen', arrows=True, 
                                 arrowsize=20, arrowstyle='->')
            
            plt.title("Class Hierarchy Diagram")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "output_path": output_path,
                "classes": len(G.nodes()),
                "relationships": len(G.edges()),
                "hierarchy": results
            }
        
        except Exception as e:
            return {"error": f"Failed to generate class hierarchy: {str(e)}"}
    
    def generate_function_call_graph(self, output_path: str = "function_calls.png") -> Dict[str, Any]:
        """Generate a function call graph visualization."""
        if not self.graph_builder:
            return {"error": "Graph builder not available"}
        
        try:
            # Get function call relationships
            query = """
            MATCH (f1:Function)-[:CALLS]->(f2:Function)
            RETURN f1.name as caller, f2.name as callee, f1.file_path as caller_file, f2.file_path as callee_file
            LIMIT 50
            """
            results = self.graph_builder.query_graph(query)
            
            if not results:
                return {"error": "No function calls found"}
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes and edges
            for result in results:
                caller = result.get("caller", "Unknown")
                callee = result.get("callee", "Unknown")
                caller_file = result.get("caller_file", "")
                callee_file = result.get("callee_file", "")
                
                G.add_node(caller, file=caller_file)
                G.add_node(callee, file=callee_file)
                G.add_edge(caller, callee)
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=1.5, iterations=50)
            
            # Draw nodes with different colors based on file
            node_colors = []
            for node in G.nodes():
                file = G.nodes[node].get('file', '')
                if 'controller' in file.lower():
                    node_colors.append('#FF6B6B')
                elif 'service' in file.lower():
                    node_colors.append('#4ECDC4')
                elif 'model' in file.lower():
                    node_colors.append('#45B7D1')
                else:
                    node_colors.append('#96CEB4')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                 arrowsize=15, alpha=0.7)
            
            plt.title("Function Call Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "output_path": output_path,
                "functions": len(G.nodes()),
                "calls": len(G.edges()),
                "call_graph": results
            }
        
        except Exception as e:
            return {"error": f"Failed to generate function call graph: {str(e)}"}
    
    def generate_complexity_heatmap(self, output_path: str = "complexity_heatmap.png") -> Dict[str, Any]:
        """Generate a complexity heatmap of the codebase."""
        if not self.graph_builder:
            return {"error": "Graph builder not available"}
        
        try:
            # Get function complexity data
            query = """
            MATCH (f:Function)
            RETURN f.name, f.file_path, f.line_end - f.line_start as line_count
            ORDER BY line_count DESC
            LIMIT 100
            """
            results = self.graph_builder.query_graph(query)
            
            if not results:
                return {"error": "No functions found"}
            
            # Group by file
            file_complexity = {}
            for result in results:
                file_path = result.get("f.file_path", "")
                line_count = result.get("line_count", 0)
                
                if file_path not in file_complexity:
                    file_complexity[file_path] = []
                file_complexity[file_path].append(line_count)
            
            # Calculate average complexity per file
            file_avg_complexity = {}
            for file_path, complexities in file_complexity.items():
                file_avg_complexity[file_path] = sum(complexities) / len(complexities)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort files by complexity
            sorted_files = sorted(file_avg_complexity.items(), 
                                key=lambda x: x[1], reverse=True)
            
            files = [f[0] for f in sorted_files[:20]]  # Top 20 files
            complexities = [f[1] for f in sorted_files[:20]]
            
            # Create color mapping
            colors = plt.cm.Reds(plt.Normalize(min(complexities), max(complexities))(complexities))
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(files)), complexities, color=colors)
            
            # Customize appearance
            ax.set_yticks(range(len(files)))
            ax.set_yticklabels([Path(f).name for f in files], fontsize=8)
            ax.set_xlabel('Average Function Complexity (Lines)', fontsize=12)
            ax.set_title('Code Complexity Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                     norm=plt.Normalize(min(complexities), max(complexities)))
            plt.colorbar(sm, ax=ax, label='Complexity')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "output_path": output_path,
                "files_analyzed": len(file_avg_complexity),
                "top_complex_files": sorted_files[:10]
            }
        
        except Exception as e:
            return {"error": f"Failed to generate complexity heatmap: {str(e)}"}
    
    def generate_all_visualizations(self, output_dir: str = "visualizations") -> Dict[str, Any]:
        """Generate all available visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        # Generate each visualization
        visualizations = [
            ("dependency_graph", self.generate_dependency_graph),
            ("architecture_diagram", self.generate_architecture_diagram),
            ("class_hierarchy", self.generate_class_hierarchy),
            ("function_calls", self.generate_function_call_graph),
            ("complexity_heatmap", self.generate_complexity_heatmap)
        ]
        
        for name, func in visualizations:
            output_file = output_path / f"{name}.png"
            result = func(str(output_file))
            results[name] = result
        
        return {
            "success": True,
            "output_directory": str(output_path),
            "visualizations": results
        }
    
    def export_graph_data(self, output_path: str = "graph_data.json") -> Dict[str, Any]:
        """Export graph data for external visualization tools."""
        if not self.graph_builder:
            return {"error": "Graph builder not available"}
        
        try:
            # Export in a format suitable for tools like D3.js
            export_data = self.graph_builder.export_graph("json")
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "success": True,
                "output_path": output_path,
                "message": "Graph data exported for external visualization"
            }
        
        except Exception as e:
            return {"error": f"Failed to export graph data: {str(e)}"} 