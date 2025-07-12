"""
Language-Agnostic Code Parser

This module provides a unified interface for parsing code in multiple languages
and generating AST representations for the knowledge graph.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import tree_sitter
from tree_sitter import Language, Parser
import json


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVASCRIPT = "javascript"


@dataclass
class CodeNode:
    """Represents a parsed code element."""
    id: str
    name: str
    type: str
    language: LanguageType
    file_path: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    content: str
    docstring: Optional[str] = None
    imports: List[str] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FileInfo:
    """Information about a parsed file."""
    path: str
    language: LanguageType
    size: int
    lines: int
    nodes: List[CodeNode]
    imports: List[str]
    exports: List[str]


class CodeParser:
    """
    Language-agnostic code parser that generates AST representations.
    
    Supports Python, TypeScript, and Go with extensible architecture
    for adding more languages.
    """
    
    def __init__(self):
        self.languages = {}
        self.parsers = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        try:
            # Initialize tree-sitter languages
            Language.build_library(
                'build/my-languages.so',
                [
                    'vendor/tree-sitter-python',
                    'vendor/tree-sitter-typescript',
                    'vendor/tree-sitter-go',
                    'vendor/tree-sitter-javascript'
                ]
            )
            
            # Load languages
            PY_LANGUAGE = Language('build/my-languages.so', 'python')
            TS_LANGUAGE = Language('build/my-languages.so', 'typescript')
            GO_LANGUAGE = Language('build/my-languages.so', 'go')
            JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
            
            # Create parsers
            self.parsers[LanguageType.PYTHON] = self._create_parser(PY_LANGUAGE)
            self.parsers[LanguageType.TYPESCRIPT] = self._create_parser(TS_LANGUAGE)
            self.parsers[LanguageType.GO] = self._create_parser(GO_LANGUAGE)
            self.parsers[LanguageType.JAVASCRIPT] = self._create_parser(JS_LANGUAGE)
            
        except Exception as e:
            # Fallback to built-in parsers
            print(f"Warning: Tree-sitter initialization failed: {e}")
            self._initialize_fallback_parsers()
    
    def _create_parser(self, language: Language) -> Parser:
        """Create a tree-sitter parser for a language."""
        parser = Parser()
        parser.set_language(language)
        return parser
    
    def _initialize_fallback_parsers(self):
        """Initialize fallback parsers using built-in libraries."""
        self.use_fallback = True
    
    def detect_language(self, file_path: str) -> Optional[LanguageType]:
        """Detect the programming language of a file based on extension."""
        ext = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': LanguageType.PYTHON,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.go': LanguageType.GO,
        }
        
        return language_map.get(ext)
    
    def parse_file(self, file_path: str) -> Optional[FileInfo]:
        """Parse a single file and return its AST representation."""
        if not os.path.exists(file_path):
            return None
        
        language = self.detect_language(file_path)
        if not language:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if language == LanguageType.PYTHON:
                return self._parse_python_file(file_path, content)
            elif language == LanguageType.TYPESCRIPT:
                return self._parse_typescript_file(file_path, content)
            elif language == LanguageType.GO:
                return self._parse_go_file(file_path, content)
            elif language == LanguageType.JAVASCRIPT:
                return self._parse_javascript_file(file_path, content)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _parse_python_file(self, file_path: str, content: str) -> FileInfo:
        """Parse a Python file using ast module."""
        try:
            tree = ast.parse(content)
            nodes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    nodes.append(self._extract_function_node(node, file_path, content))
                elif isinstance(node, ast.ClassDef):
                    nodes.append(self._extract_class_node(node, file_path, content))
                elif isinstance(node, ast.Import):
                    imports.extend(self._extract_imports(node))
                elif isinstance(node, ast.ImportFrom):
                    imports.extend(self._extract_import_from(node))
            
            return FileInfo(
                path=file_path,
                language=LanguageType.PYTHON,
                size=len(content),
                lines=len(content.splitlines()),
                nodes=nodes,
                imports=imports,
                exports=[]
            )
            
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return None
    
    def _extract_function_node(self, node: ast.FunctionDef, file_path: str, content: str) -> CodeNode:
        """Extract function information from AST node."""
        lines = content.splitlines()
        docstring = ast.get_docstring(node)
        
        return CodeNode(
            id=f"{file_path}:{node.name}",
            name=node.name,
            type="function",
            language=LanguageType.PYTHON,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            column_end=node.end_col_offset or node.col_offset,
            content="\n".join(lines[node.lineno-1:node.end_lineno]) if node.end_lineno else lines[node.lineno-1],
            docstring=docstring,
            metadata={
                "args": [arg.arg for arg in node.args.args],
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                "returns": self._get_return_annotation(node)
            }
        )
    
    def _extract_class_node(self, node: ast.ClassDef, file_path: str, content: str) -> CodeNode:
        """Extract class information from AST node."""
        lines = content.splitlines()
        docstring = ast.get_docstring(node)
        
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        return CodeNode(
            id=f"{file_path}:{node.name}",
            name=node.name,
            type="class",
            language=LanguageType.PYTHON,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            column_end=node.end_col_offset or node.col_offset,
            content="\n".join(lines[node.lineno-1:node.end_lineno]) if node.end_lineno else lines[node.lineno-1],
            docstring=docstring,
            metadata={
                "bases": [self._get_base_name(base) for base in node.bases],
                "methods": [method.name for method in methods],
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
            }
        )
    
    def _extract_imports(self, node: ast.Import) -> List[str]:
        """Extract import statements."""
        return [alias.name for alias in node.names]
    
    def _extract_import_from(self, node: ast.ImportFrom) -> List[str]:
        """Extract from import statements."""
        module = node.module or ""
        return [f"{module}.{alias.name}" for alias in node.names]
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_attribute_base(decorator.value)}.{decorator.attr}"
        return str(decorator)
    
    def _get_attribute_base(self, node: ast.expr) -> str:
        """Get the base of an attribute access."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_base(node.value)}.{node.attr}"
        return str(node)
    
    def _get_base_name(self, base: ast.expr) -> str:
        """Get base class name."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return f"{self._get_attribute_base(base.value)}.{base.attr}"
        return str(base)
    
    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Get return annotation from function."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return f"{self._get_attribute_base(node.returns.value)}.{node.returns.attr}"
        return None
    
    def _parse_typescript_file(self, file_path: str, content: str) -> FileInfo:
        """Parse a TypeScript file (simplified implementation)."""
        # This is a simplified implementation
        # In production, you'd use tree-sitter or a proper TypeScript parser
        nodes = []
        imports = []
        
        # Extract imports using regex
        import_pattern = r'import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+)?[\'"]([^\'"]+)[\'"]'
        imports = re.findall(import_pattern, content)
        
        # Extract function and class definitions
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)'
        class_pattern = r'(?:export\s+)?class\s+(\w+)'
        
        for match in re.finditer(function_pattern, content):
            name = match.group(1)
            nodes.append(CodeNode(
                id=f"{file_path}:{name}",
                name=name,
                type="function",
                language=LanguageType.TYPESCRIPT,
                file_path=file_path,
                line_start=content[:match.start()].count('\n') + 1,
                line_end=content[:match.end()].count('\n') + 1,
                column_start=match.start(),
                column_end=match.end(),
                content=match.group(0)
            ))
        
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            nodes.append(CodeNode(
                id=f"{file_path}:{name}",
                name=name,
                type="class",
                language=LanguageType.TYPESCRIPT,
                file_path=file_path,
                line_start=content[:match.start()].count('\n') + 1,
                line_end=content[:match.end()].count('\n') + 1,
                column_start=match.start(),
                column_end=match.end(),
                content=match.group(0)
            ))
        
        return FileInfo(
            path=file_path,
            language=LanguageType.TYPESCRIPT,
            size=len(content),
            lines=len(content.splitlines()),
            nodes=nodes,
            imports=imports,
            exports=[]
        )
    
    def _parse_go_file(self, file_path: str, content: str) -> FileInfo:
        """Parse a Go file (simplified implementation)."""
        nodes = []
        imports = []
        
        # Extract imports
        import_pattern = r'import\s+(?:\(([^)]+)\)|"([^"]+)")'
        for match in re.finditer(import_pattern, content):
            if match.group(1):  # Multi-line imports
                import_lines = match.group(1).strip().split('\n')
                for line in import_lines:
                    if '"' in line:
                        imports.append(line.split('"')[1])
            else:  # Single import
                imports.append(match.group(2))
        
        # Extract function definitions
        func_pattern = r'func\s+(?:\(\w+\s+\w+\)\s+)?(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            nodes.append(CodeNode(
                id=f"{file_path}:{name}",
                name=name,
                type="function",
                language=LanguageType.GO,
                file_path=file_path,
                line_start=content[:match.start()].count('\n') + 1,
                line_end=content[:match.end()].count('\n') + 1,
                column_start=match.start(),
                column_end=match.end(),
                content=match.group(0)
            ))
        
        return FileInfo(
            path=file_path,
            language=LanguageType.GO,
            size=len(content),
            lines=len(content.splitlines()),
            nodes=nodes,
            imports=imports,
            exports=[]
        )
    
    def _parse_javascript_file(self, file_path: str, content: str) -> FileInfo:
        """Parse a JavaScript file (simplified implementation)."""
        # Similar to TypeScript but without type annotations
        return self._parse_typescript_file(file_path, content)
    
    def parse_directory(self, directory_path: str, exclude_patterns: List[str] = None) -> Dict[str, FileInfo]:
        """Parse all supported files in a directory recursively."""
        if exclude_patterns is None:
            exclude_patterns = [
                '__pycache__', '.git', 'node_modules', '.venv', 'venv',
                '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.log'
            ]
        
        files = {}
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                # Check if file should be excluded
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue
                
                file_info = self.parse_file(str(file_path))
                if file_info:
                    files[str(file_path)] = file_info
        
        return files
    
    def to_json(self, file_info: FileInfo) -> str:
        """Convert file information to JSON format."""
        return json.dumps({
            "path": file_info.path,
            "language": file_info.language.value,
            "size": file_info.size,
            "lines": file_info.lines,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "language": node.language.value,
                    "file_path": node.file_path,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "column_start": node.column_start,
                    "column_end": node.column_end,
                    "content": node.content,
                    "docstring": node.docstring,
                    "imports": node.imports,
                    "dependencies": node.dependencies,
                    "metadata": node.metadata
                }
                for node in file_info.nodes
            ],
            "imports": file_info.imports,
            "exports": file_info.exports
        }, indent=2) 