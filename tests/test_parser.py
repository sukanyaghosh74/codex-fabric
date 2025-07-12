"""
Test module for the Code Parser

This module contains unit tests for the language-agnostic code parser.
"""

import pytest
import tempfile
import os
from pathlib import Path
from core.parser import CodeParser, LanguageType


class TestCodeParser:
    """Test cases for the CodeParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CodeParser()
    
    def test_detect_language_python(self):
        """Test Python language detection."""
        assert self.parser.detect_language("test.py") == LanguageType.PYTHON
    
    def test_detect_language_typescript(self):
        """Test TypeScript language detection."""
        assert self.parser.detect_language("test.ts") == LanguageType.TYPESCRIPT
        assert self.parser.detect_language("test.tsx") == LanguageType.TYPESCRIPT
    
    def test_detect_language_go(self):
        """Test Go language detection."""
        assert self.parser.detect_language("test.go") == LanguageType.GO
    
    def test_detect_language_javascript(self):
        """Test JavaScript language detection."""
        assert self.parser.detect_language("test.js") == LanguageType.JAVASCRIPT
        assert self.parser.detect_language("test.jsx") == LanguageType.JAVASCRIPT
    
    def test_detect_language_unknown(self):
        """Test unknown language detection."""
        assert self.parser.detect_language("test.txt") is None
    
    def test_parse_python_file(self):
        """Test parsing a Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def test_function():
    \"\"\"Test function docstring.\"\"\"
    return True

class TestClass:
    def __init__(self):
        pass
""")
            temp_file = f.name
        
        try:
            file_info = self.parser.parse_file(temp_file)
            assert file_info is not None
            assert file_info.language == LanguageType.PYTHON
            assert len(file_info.nodes) >= 2  # At least function and class
        finally:
            os.unlink(temp_file)
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        result = self.parser.parse_file("nonexistent.py")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__]) 