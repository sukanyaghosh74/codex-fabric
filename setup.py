"""
Setup script for Codex Fabric

This script installs the Codex Fabric CLI tool and its dependencies.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="codex-fabric",
    version="0.1.0",
    author="Codex Fabric Team",
    author_email="team@codexfabric.com",
    description="Transform codebases into intelligent knowledge graphs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/codex-fabric",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "cfabric=cli.main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="code analysis, knowledge graph, AI, codebase, architecture, refactoring",
    project_urls={
        "Bug Reports": "https://github.com/your-org/codex-fabric/issues",
        "Source": "https://github.com/your-org/codex-fabric",
        "Documentation": "https://github.com/your-org/codex-fabric/docs",
    },
) 