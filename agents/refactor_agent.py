"""
Refactor Agent Module

This agent specializes in providing refactoring suggestions and code improvements
based on analysis of the knowledge graph and code patterns.
"""

from typing import Dict, List, Any, Optional
import json


class RefactorAgent:
    """Agent specialized in refactoring suggestions and code improvements."""
    
    def __init__(self, graph_builder=None, embedding_service=None):
        self.graph_builder = graph_builder
        self.embedding_service = embedding_service
    
    def analyze_refactoring_opportunities(self, codebase_path: str) -> Dict[str, Any]:
        """Analyze the codebase for refactoring opportunities."""
        opportunities = {
            "code_smells": self._detect_code_smells(),
            "duplicated_code": self._find_duplicated_code(),
            "long_methods": self._find_long_methods(),
            "large_classes": self._find_large_classes(),
            "complex_conditions": self._find_complex_conditions(),
            "recommendations": []
        }
        
        # Generate refactoring recommendations
        opportunities["recommendations"] = self._generate_refactoring_recommendations(opportunities)
        
        return opportunities
    
    def _detect_code_smells(self) -> List[Dict[str, Any]]:
        """Detect common code smells in the codebase."""
        smells = []
        
        if self.graph_builder:
            try:
                # Long parameter list
                long_params_query = """
                MATCH (f:Function)
                WHERE size(f.metadata.args) > 5
                RETURN f.name, f.file_path, size(f.metadata.args) as param_count
                """
                results = self.graph_builder.query_graph(long_params_query)
                for result in results:
                    smells.append({
                        "type": "long_parameter_list",
                        "severity": "medium",
                        "function": result.get("f.name"),
                        "file": result.get("f.file_path"),
                        "details": f"Function has {result.get('param_count')} parameters"
                    })
                
                # Feature envy
                feature_envy_query = """
                MATCH (f:Function)-[:CALLS]->(other:Function)
                WHERE other.file_path <> f.file_path
                WITH f, count(other) as external_calls
                WHERE external_calls > 3
                RETURN f.name, f.file_path, external_calls
                """
                results = self.graph_builder.query_graph(feature_envy_query)
                for result in results:
                    smells.append({
                        "type": "feature_envy",
                        "severity": "medium",
                        "function": result.get("f.name"),
                        "file": result.get("f.file_path"),
                        "details": f"Function makes {result.get('external_calls')} external calls"
                    })
                
                # Data clumps
                data_clumps_query = """
                MATCH (f1:Function)-[:HAS_PARAMETER]->(p1)
                MATCH (f2:Function)-[:HAS_PARAMETER]->(p2)
                WHERE f1 <> f2 AND p1.name = p2.name
                WITH p1.name as param_name, count(DISTINCT f1) as function_count
                WHERE function_count > 2
                RETURN param_name, function_count
                """
                results = self.graph_builder.query_graph(data_clumps_query)
                for result in results:
                    smells.append({
                        "type": "data_clumps",
                        "severity": "low",
                        "parameter": result.get("param_name"),
                        "details": f"Parameter used in {result.get('function_count')} functions"
                    })
            
            except Exception as e:
                print(f"Error detecting code smells: {e}")
        
        return smells
    
    def _find_duplicated_code(self) -> List[Dict[str, Any]]:
        """Find duplicated code patterns."""
        duplicates = []
        
        if self.embedding_service:
            try:
                # Use semantic search to find similar code
                # This is a simplified approach - in production you'd use more sophisticated algorithms
                similar_functions = self.embedding_service.search_similar("function implementation", top_k=20)
                
                # Group similar functions
                similarity_groups = {}
                for result in similar_functions:
                    content_hash = hash(result["content"][:100])  # Simple hash of first 100 chars
                    if content_hash not in similarity_groups:
                        similarity_groups[content_hash] = []
                    similarity_groups[content_hash].append(result)
                
                # Find groups with multiple similar functions
                for group_id, functions in similarity_groups.items():
                    if len(functions) > 1:
                        duplicates.append({
                            "type": "duplicated_code",
                            "severity": "high",
                            "functions": [f["metadata"]["name"] for f in functions],
                            "files": [f["metadata"]["file_path"] for f in functions],
                            "similarity_score": functions[0]["score"]
                        })
            
            except Exception as e:
                print(f"Error finding duplicated code: {e}")
        
        return duplicates
    
    def _find_long_methods(self) -> List[Dict[str, Any]]:
        """Find methods that are too long."""
        long_methods = []
        
        if self.graph_builder:
            try:
                # Find functions with many lines
                long_methods_query = """
                MATCH (f:Function)
                WHERE f.line_end - f.line_start > 20
                RETURN f.name, f.file_path, f.line_end - f.line_start as line_count
                ORDER BY line_count DESC
                """
                results = self.graph_builder.query_graph(long_methods_query)
                
                for result in results:
                    line_count = result.get("line_count", 0)
                    severity = "high" if line_count > 50 else "medium" if line_count > 30 else "low"
                    
                    long_methods.append({
                        "type": "long_method",
                        "severity": severity,
                        "function": result.get("f.name"),
                        "file": result.get("f.file_path"),
                        "line_count": line_count,
                        "details": f"Method has {line_count} lines"
                    })
            
            except Exception as e:
                print(f"Error finding long methods: {e}")
        
        return long_methods
    
    def _find_large_classes(self) -> List[Dict[str, Any]]:
        """Find classes that are too large."""
        large_classes = []
        
        if self.graph_builder:
            try:
                # Find classes with many methods
                large_classes_query = """
                MATCH (c:Class)-[:HAS_METHOD]->(m:Function)
                WITH c, count(m) as method_count
                WHERE method_count > 10
                RETURN c.name, c.file_path, method_count
                ORDER BY method_count DESC
                """
                results = self.graph_builder.query_graph(large_classes_query)
                
                for result in results:
                    method_count = result.get("method_count", 0)
                    severity = "high" if method_count > 20 else "medium" if method_count > 15 else "low"
                    
                    large_classes.append({
                        "type": "large_class",
                        "severity": severity,
                        "class": result.get("c.name"),
                        "file": result.get("c.file_path"),
                        "method_count": method_count,
                        "details": f"Class has {method_count} methods"
                    })
            
            except Exception as e:
                print(f"Error finding large classes: {e}")
        
        return large_classes
    
    def _find_complex_conditions(self) -> List[Dict[str, Any]]:
        """Find complex conditional statements."""
        complex_conditions = []
        
        if self.graph_builder:
            try:
                # Find functions with complex conditions (simplified detection)
                complex_query = """
                MATCH (f:Function)
                WHERE f.content CONTAINS 'if' AND f.content CONTAINS 'else'
                WITH f, size(split(f.content, 'if')) as if_count
                WHERE if_count > 3
                RETURN f.name, f.file_path, if_count
                """
                results = self.graph_builder.query_graph(complex_query)
                
                for result in results:
                    if_count = result.get("if_count", 0)
                    severity = "high" if if_count > 5 else "medium" if if_count > 3 else "low"
                    
                    complex_conditions.append({
                        "type": "complex_conditions",
                        "severity": severity,
                        "function": result.get("f.name"),
                        "file": result.get("f.file_path"),
                        "condition_count": if_count,
                        "details": f"Function has {if_count} conditional statements"
                    })
            
            except Exception as e:
                print(f"Error finding complex conditions: {e}")
        
        return complex_conditions
    
    def _generate_refactoring_recommendations(self, opportunities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific refactoring recommendations."""
        recommendations = []
        
        # Recommendations based on code smells
        for smell in opportunities["code_smells"]:
            if smell["type"] == "long_parameter_list":
                recommendations.append({
                    "type": "extract_parameter_object",
                    "target": smell["function"],
                    "file": smell["file"],
                    "description": f"Extract parameters into a parameter object for {smell['function']}",
                    "priority": "medium",
                    "effort": "medium"
                })
            
            elif smell["type"] == "feature_envy":
                recommendations.append({
                    "type": "move_method",
                    "target": smell["function"],
                    "file": smell["file"],
                    "description": f"Consider moving {smell['function']} to the class it most frequently calls",
                    "priority": "medium",
                    "effort": "high"
                })
        
        # Recommendations based on duplicated code
        for duplicate in opportunities["duplicated_code"]:
            recommendations.append({
                "type": "extract_method",
                "target": duplicate["functions"][0],
                "file": duplicate["files"][0],
                "description": f"Extract common code from {', '.join(duplicate['functions'])} into a shared method",
                "priority": "high",
                "effort": "medium"
            })
        
        # Recommendations based on long methods
        for method in opportunities["long_methods"]:
            if method["severity"] == "high":
                recommendations.append({
                    "type": "extract_method",
                    "target": method["function"],
                    "file": method["file"],
                    "description": f"Break down {method['function']} into smaller, focused methods",
                    "priority": "high",
                    "effort": "medium"
                })
        
        # Recommendations based on large classes
        for class_info in opportunities["large_classes"]:
            if class_info["severity"] == "high":
                recommendations.append({
                    "type": "extract_class",
                    "target": class_info["class"],
                    "file": class_info["file"],
                    "description": f"Split {class_info['class']} into smaller, focused classes",
                    "priority": "high",
                    "effort": "high"
                })
        
        return recommendations
    
    def suggest_refactoring_for_function(self, function_id: str) -> Dict[str, Any]:
        """Suggest specific refactoring for a given function."""
        suggestions = {
            "function_id": function_id,
            "analysis": {},
            "suggestions": []
        }
        
        if self.graph_builder:
            try:
                # Get function details
                function_query = f"""
                MATCH (f:Function {{id: '{function_id}'}})
                RETURN f.name, f.file_path, f.content, f.metadata
                """
                results = self.graph_builder.query_graph(function_query)
                
                if results:
                    function = results[0]
                    suggestions["analysis"] = {
                        "name": function.get("f.name"),
                        "file": function.get("f.file_path"),
                        "line_count": len(function.get("f.content", "").splitlines()),
                        "parameter_count": len(function.get("f.metadata", {}).get("args", [])),
                        "complexity": self._calculate_complexity(function.get("f.content", ""))
                    }
                    
                    # Generate specific suggestions
                    analysis = suggestions["analysis"]
                    
                    if analysis["line_count"] > 20:
                        suggestions["suggestions"].append({
                            "type": "extract_method",
                            "description": "Method is too long. Consider extracting smaller methods.",
                            "priority": "high"
                        })
                    
                    if analysis["parameter_count"] > 5:
                        suggestions["suggestions"].append({
                            "type": "extract_parameter_object",
                            "description": "Too many parameters. Consider using a parameter object.",
                            "priority": "medium"
                        })
                    
                    if analysis["complexity"] > 10:
                        suggestions["suggestions"].append({
                            "type": "simplify_conditions",
                            "description": "High cyclomatic complexity. Simplify conditional logic.",
                            "priority": "high"
                        })
            
            except Exception as e:
                print(f"Error suggesting refactoring for function: {e}")
        
        return suggestions
    
    def _calculate_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = ["if", "elif", "else", "for", "while", "and", "or"]
        for keyword in decision_keywords:
            complexity += content.count(keyword)
        
        return complexity
    
    def get_refactoring_priority_queue(self) -> List[Dict[str, Any]]:
        """Get a prioritized queue of refactoring tasks."""
        opportunities = self.analyze_refactoring_opportunities("")
        
        # Create priority queue based on severity and impact
        priority_queue = []
        
        severity_weights = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        # Add code smells to queue
        for smell in opportunities["code_smells"]:
            priority_queue.append({
                "type": "code_smell",
                "issue": smell,
                "priority_score": severity_weights.get(smell["severity"], 1),
                "effort": "medium"
            })
        
        # Add duplicated code to queue
        for duplicate in opportunities["duplicated_code"]:
            priority_queue.append({
                "type": "duplicated_code",
                "issue": duplicate,
                "priority_score": 3,  # High priority for duplicates
                "effort": "medium"
            })
        
        # Add long methods to queue
        for method in opportunities["long_methods"]:
            priority_queue.append({
                "type": "long_method",
                "issue": method,
                "priority_score": severity_weights.get(method["severity"], 1),
                "effort": "medium"
            })
        
        # Sort by priority score (descending)
        priority_queue.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return priority_queue 