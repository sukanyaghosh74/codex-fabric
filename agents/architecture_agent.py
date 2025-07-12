"""
Architecture Agent Module

This agent specializes in analyzing code architecture, identifying patterns,
and providing architectural recommendations.
"""

from typing import Dict, List, Any
import json


class ArchitectureAgent:
    """Agent specialized in architectural analysis and recommendations."""
    
    def __init__(self, graph_builder=None, embedding_service=None):
        self.graph_builder = graph_builder
        self.embedding_service = embedding_service
    
    def analyze_architecture(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze the overall architecture of a codebase."""
        analysis = {
            "layers": self._identify_layers(),
            "patterns": self._identify_patterns(),
            "dependencies": self._analyze_dependencies(),
            "coupling": self._analyze_coupling(),
            "cohesion": self._analyze_cohesion(),
            "recommendations": []
        }
        
        # Generate recommendations based on analysis
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _identify_layers(self) -> List[Dict[str, Any]]:
        """Identify architectural layers in the codebase."""
        layers = []
        
        if self.graph_builder:
            try:
                # Query for different types of files/components
                queries = {
                    "controllers": "MATCH (f:File) WHERE f.path CONTAINS 'controller' OR f.path CONTAINS 'api' RETURN f.path, f.language",
                    "services": "MATCH (f:File) WHERE f.path CONTAINS 'service' OR f.path CONTAINS 'business' RETURN f.path, f.language",
                    "models": "MATCH (f:File) WHERE f.path CONTAINS 'model' OR f.path CONTAINS 'entity' RETURN f.path, f.language",
                    "utils": "MATCH (f:File) WHERE f.path CONTAINS 'util' OR f.path CONTAINS 'helper' RETURN f.path, f.language"
                }
                
                for layer_name, query in queries.items():
                    results = self.graph_builder.query_graph(query)
                    if results:
                        layers.append({
                            "name": layer_name,
                            "files": [r["f.path"] for r in results],
                            "count": len(results)
                        })
            
            except Exception as e:
                print(f"Error identifying layers: {e}")
        
        return layers
    
    def _identify_patterns(self) -> List[Dict[str, Any]]:
        """Identify common design patterns in the codebase."""
        patterns = []
        
        if self.graph_builder:
            try:
                # Look for common patterns
                pattern_queries = {
                    "singleton": "MATCH (c:Class) WHERE c.content CONTAINS 'getInstance' OR c.content CONTAINS 'instance' RETURN c.name, c.file_path",
                    "factory": "MATCH (c:Class) WHERE c.name CONTAINS 'Factory' OR c.content CONTAINS 'create' RETURN c.name, c.file_path",
                    "observer": "MATCH (c:Class) WHERE c.content CONTAINS 'addListener' OR c.content CONTAINS 'notify' RETURN c.name, c.file_path",
                    "decorator": "MATCH (c:Class)-[:INHERITS_FROM]->(parent) WHERE c.content CONTAINS 'decorate' RETURN c.name, parent.name"
                }
                
                for pattern_name, query in pattern_queries.items():
                    results = self.graph_builder.query_graph(query)
                    if results:
                        patterns.append({
                            "name": pattern_name,
                            "instances": [{"class": r.get("c.name", ""), "file": r.get("c.file_path", "")} for r in results],
                            "count": len(results)
                        })
            
            except Exception as e:
                print(f"Error identifying patterns: {e}")
        
        return patterns
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies between components."""
        dependencies = {
            "circular": [],
            "high_coupling": [],
            "external": [],
            "internal": []
        }
        
        if self.graph_builder:
            try:
                # Check for circular dependencies
                circular_query = """
                MATCH path = (a)-[:DEPENDS_ON*]->(a)
                RETURN a.name as component, length(path) as cycle_length
                """
                results = self.graph_builder.query_graph(circular_query)
                dependencies["circular"] = results
                
                # Find highly coupled components
                coupling_query = """
                MATCH (a)-[:DEPENDS_ON]->(b)
                WITH a, count(b) as dependency_count
                WHERE dependency_count > 5
                RETURN a.name, dependency_count
                ORDER BY dependency_count DESC
                """
                results = self.graph_builder.query_graph(coupling_query)
                dependencies["high_coupling"] = results
            
            except Exception as e:
                print(f"Error analyzing dependencies: {e}")
        
        return dependencies
    
    def _analyze_coupling(self) -> Dict[str, Any]:
        """Analyze coupling between modules."""
        coupling_analysis = {
            "tight_coupling": [],
            "loose_coupling": [],
            "metrics": {}
        }
        
        if self.graph_builder:
            try:
                # Calculate coupling metrics
                metrics_query = """
                MATCH (a)-[:DEPENDS_ON]->(b)
                WITH a, count(b) as outbound
                MATCH (c)-[:DEPENDS_ON]->(a)
                WITH a, outbound, count(c) as inbound
                RETURN a.name, outbound, inbound, outbound + inbound as total_coupling
                ORDER BY total_coupling DESC
                """
                results = self.graph_builder.query_graph(metrics_query)
                
                for result in results:
                    total = result.get("total_coupling", 0)
                    if total > 10:
                        coupling_analysis["tight_coupling"].append(result)
                    elif total < 3:
                        coupling_analysis["loose_coupling"].append(result)
                
                coupling_analysis["metrics"] = {
                    "avg_coupling": sum(r.get("total_coupling", 0) for r in results) / len(results) if results else 0,
                    "max_coupling": max(r.get("total_coupling", 0) for r in results) if results else 0,
                    "min_coupling": min(r.get("total_coupling", 0) for r in results) if results else 0
                }
            
            except Exception as e:
                print(f"Error analyzing coupling: {e}")
        
        return coupling_analysis
    
    def _analyze_cohesion(self) -> Dict[str, Any]:
        """Analyze cohesion within modules."""
        cohesion_analysis = {
            "high_cohesion": [],
            "low_cohesion": [],
            "metrics": {}
        }
        
        if self.graph_builder:
            try:
                # Analyze function relationships within files
                cohesion_query = """
                MATCH (f:File)-[:CONTAINS]->(func:Function)
                WITH f, collect(func) as functions
                UNWIND functions as func1
                UNWIND functions as func2
                WHERE func1 <> func2
                MATCH (func1)-[:CALLS]->(func2)
                WITH f, count(*) as internal_calls
                RETURN f.path, size(functions) as function_count, internal_calls
                """
                results = self.graph_builder.query_graph(cohesion_query)
                
                for result in results:
                    function_count = result.get("function_count", 1)
                    internal_calls = result.get("internal_calls", 0)
                    cohesion_ratio = internal_calls / (function_count * (function_count - 1)) if function_count > 1 else 0
                    
                    if cohesion_ratio > 0.3:
                        cohesion_analysis["high_cohesion"].append({
                            "file": result.get("f.path"),
                            "cohesion_ratio": cohesion_ratio
                        })
                    elif cohesion_ratio < 0.1:
                        cohesion_analysis["low_cohesion"].append({
                            "file": result.get("f.path"),
                            "cohesion_ratio": cohesion_ratio
                        })
            
            except Exception as e:
                print(f"Error analyzing cohesion: {e}")
        
        return cohesion_analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate architectural recommendations based on analysis."""
        recommendations = []
        
        # Coupling recommendations
        if len(analysis["dependencies"]["circular"]) > 0:
            recommendations.append("🔴 CRITICAL: Circular dependencies detected. Consider refactoring to break dependency cycles.")
        
        if len(analysis["dependencies"]["high_coupling"]) > 0:
            recommendations.append("🟡 WARNING: High coupling detected. Consider applying dependency inversion principle.")
        
        # Cohesion recommendations
        if len(analysis["cohesion"]["low_cohesion"]) > 0:
            recommendations.append("🟡 WARNING: Low cohesion detected. Consider splitting modules with low internal coupling.")
        
        # Pattern recommendations
        if len(analysis["patterns"]) == 0:
            recommendations.append("💡 SUGGESTION: Consider implementing common design patterns for better code organization.")
        
        # Layer recommendations
        if len(analysis["layers"]) < 3:
            recommendations.append("💡 SUGGESTION: Consider implementing a layered architecture for better separation of concerns.")
        
        return recommendations
    
    def get_architectural_metrics(self) -> Dict[str, Any]:
        """Get comprehensive architectural metrics."""
        metrics = {
            "complexity": self._calculate_complexity_metrics(),
            "maintainability": self._calculate_maintainability_metrics(),
            "testability": self._calculate_testability_metrics(),
            "scalability": self._calculate_scalability_metrics()
        }
        
        return metrics
    
    def _calculate_complexity_metrics(self) -> Dict[str, float]:
        """Calculate complexity metrics."""
        metrics = {
            "cyclomatic_complexity": 0.0,
            "cognitive_complexity": 0.0,
            "nesting_depth": 0.0
        }
        
        if self.graph_builder:
            try:
                # Simplified complexity calculation
                complexity_query = """
                MATCH (f:Function)
                RETURN avg(size(split(f.content, 'if'))) as avg_conditions,
                       avg(size(split(f.content, 'for'))) as avg_loops,
                       avg(size(split(f.content, '}'))) as avg_nesting
                """
                results = self.graph_builder.query_graph(complexity_query)
                
                if results:
                    result = results[0]
                    metrics["cyclomatic_complexity"] = result.get("avg_conditions", 0) + result.get("avg_loops", 0) + 1
                    metrics["cognitive_complexity"] = result.get("avg_conditions", 0) * 2 + result.get("avg_loops", 0)
                    metrics["nesting_depth"] = result.get("avg_nesting", 0)
            
            except Exception as e:
                print(f"Error calculating complexity metrics: {e}")
        
        return metrics
    
    def _calculate_maintainability_metrics(self) -> Dict[str, float]:
        """Calculate maintainability metrics."""
        metrics = {
            "maintainability_index": 0.0,
            "technical_debt_ratio": 0.0,
            "code_duplication": 0.0
        }
        
        # Simplified maintainability calculation
        # In a real implementation, you'd use more sophisticated algorithms
        
        return metrics
    
    def _calculate_testability_metrics(self) -> Dict[str, float]:
        """Calculate testability metrics."""
        metrics = {
            "test_coverage": 0.0,
            "mocking_difficulty": 0.0,
            "dependency_injection": 0.0
        }
        
        return metrics
    
    def _calculate_scalability_metrics(self) -> Dict[str, float]:
        """Calculate scalability metrics."""
        metrics = {
            "horizontal_scalability": 0.0,
            "vertical_scalability": 0.0,
            "bottleneck_risk": 0.0
        }
        
        return metrics 