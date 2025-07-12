# Codex Fabric 🧠

> **Transform any codebase into a continuously evolving knowledge graph, enabling GPT agents to reason about code like senior engineers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)

## 🚀 Overview

Codex Fabric is a modular, scalable toolchain that automatically analyzes codebases to create intelligent knowledge graphs. It enables AI agents to understand code architecture, suggest refactors, and provide insights like senior engineers.

### Key Features

- **🔍 Language-Agnostic Parsing**: Supports Python, TypeScript, Go with extensible AST processing
- **📊 Signal Tracing**: Git history analysis, function churn tracking, and runtime signal capture
- **🤖 AI-Powered Insights**: Multi-agent reasoning with LangGraph + LangChain
- **🎯 Developer Experience**: CLI tools + interactive web dashboard
- **⚡ Production Ready**: Dockerized, tested, and scalable architecture

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Tools     │    │   Web Dashboard │    │   API Gateway   │
│   (cfabric)     │    │   (Next.js)     │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Core Engine   │
                    │   (Python)      │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Parser Core   │    │   Agent Layer   │    │   Graph Store   │
│   (Rust)        │    │   (LangGraph)   │    │   (Neo4j)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Rust 1.70+
- Docker & Docker Compose
- Neo4j Database

### Installation

1. **Clone and Setup**
```powershell
git clone https://github.com/sukanyaghosh74/codex-fabric.git
cd codex-fabric
```

2. **Install Dependencies**
```powershell
# Backend
pip install -r requirements.txt

# Frontend
cd ui; npm install

# CLI
pip install -e ./cli
```

3. **Start Services**
```powershell
docker-compose up -d neo4j redis postgres
```

4. **Initialize Codex Fabric**
```powershell
cfabric init --path C:\path\to\your\codebase
```

### Basic Usage

```powershell
# Parse a codebase and build knowledge graph
cfabric init --path .\my-project

# Trace git history and runtime signals
cfabric trace --path .\my-project

# Get AI-powered insights
cfabric suggest --query "Where is authentication handled?"

# Start web dashboard
cfabric serve
```

## 📚 Documentation

- [Architecture Guide](./docs/architecture.md)
- [API Reference](./docs/api.md)
- [CLI Reference](./docs/cli.md)
- [Agent Development](./docs/agents.md)
- [Deployment Guide](./docs/deployment.md)

## 🧪 Development

### Running Tests
```powershell
# Backend tests
pytest tests/

# Frontend tests
cd ui; npm test

# Integration tests
pytest tests/integration/
```

### Local Development
```powershell
# Start all services
docker-compose up -d

# Backend development comment
# comment
uvicorn api.main:app --reload

# Frontend development
cd ui; npm run dev
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

Built with 🩷 by Sukanya Ghosh. 
Special thanks to:
- LangChain & LangGraph communities
- Neo4j for graph database technology
- OpenAI for LLM capabilities

---

**Ready to transform your codebase into an intelligent knowledge graph?** [Get Started →](./docs/quickstart.md) 
