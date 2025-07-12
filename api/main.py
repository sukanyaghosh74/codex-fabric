"""
Codex Fabric FastAPI Application

This module provides the main FastAPI application with endpoints for
interacting with the Codex Fabric knowledge graph and AI agents.
"""

import os
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

# Import core modules
from core.parser import CodeParser
from core.graph_builder import GraphBuilder
from core.signal_tracer import SignalTracer
from core.embedding_service import EmbeddingService
from agents.reasoning_agent import ReasoningAgent

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    codebase_path: str
    exclude_patterns: Optional[List[str]] = None

class TraceRequest(BaseModel):
    codebase_path: str
    days_back: int = 30

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

# Initialize FastAPI app
app = FastAPI(
    title="Codex Fabric API",
    description="API for transforming codebases into intelligent knowledge graphs",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
graph_builder = None
embedding_service = None
reasoning_agent = None

def get_services():
    """Dependency to get service instances."""
    global graph_builder, embedding_service, reasoning_agent
    
    if graph_builder is None:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        graph_builder = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
    
    if embedding_service is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_service = EmbeddingService(qdrant_url=qdrant_url, openai_api_key=openai_api_key)
    
    if reasoning_agent is None and openai_api_key:
        reasoning_agent = ReasoningAgent(
            openai_api_key=openai_api_key,
            graph_builder=graph_builder,
            embedding_service=embedding_service
        )
    
    return graph_builder, embedding_service, reasoning_agent

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Codex Fabric API",
        "version": "0.1.0",
        "description": "Transform codebases into intelligent knowledge graphs",
        "endpoints": [
            "/docs - API documentation",
            "/health - Health check",
            "/analyze - Analyze codebase",
            "/query - Query knowledge graph",
            "/search - Semantic search",
            "/trace - Trace signals",
            "/suggest - AI-powered suggestions"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        graph_builder, embedding_service, _ = get_services()
        
        # Check Neo4j connection
        neo4j_healthy = False
        try:
            graph_builder.query_graph("RETURN 1 as test")
            neo4j_healthy = True
        except:
            pass
        
        # Check Qdrant connection
        qdrant_healthy = False
        try:
            embedding_service.get_embedding_statistics()
            qdrant_healthy = True
        except:
            pass
        
        return {
            "status": "healthy" if neo4j_healthy and qdrant_healthy else "degraded",
            "services": {
                "neo4j": "healthy" if neo4j_healthy else "unhealthy",
                "qdrant": "healthy" if qdrant_healthy else "unhealthy"
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/analyze")
async def analyze_codebase(request: AnalysisRequest):
    """
    Analyze a codebase and build the knowledge graph.
    
    This endpoint will:
    1. Parse the codebase
    2. Build the knowledge graph
    3. Generate embeddings
    4. Return analysis statistics
    """
    try:
        if not os.path.exists(request.codebase_path):
            raise HTTPException(status_code=400, detail="Codebase path does not exist")
        
        # Parse codebase
        parser = CodeParser()
        files = parser.parse_directory(
            request.codebase_path, 
            exclude_patterns=request.exclude_patterns or []
        )
        
        # Build knowledge graph
        graph_builder, embedding_service, _ = get_services()
        graph_stats = graph_builder.build_graph(files)
        
        # Generate embeddings
        all_nodes = []
        for file_info in files.values():
            all_nodes.extend(file_info.nodes)
        
        embeddings = embedding_service.embed_code_elements(all_nodes)
        embedding_service.store_embeddings(embeddings)
        
        return {
            "success": True,
            "files_analyzed": len(files),
            "graph_stats": graph_stats,
            "embeddings_generated": len(embeddings),
            "languages": list(set(f.language.value for f in files.values()))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/query")
async def query_knowledge_graph(request: QueryRequest):
    """
    Execute a Cypher query on the knowledge graph.
    
    This endpoint allows direct querying of the Neo4j graph database
    using Cypher query language.
    """
    try:
        graph_builder, _, _ = get_services()
        results = graph_builder.query_graph(request.query)
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/search")
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search in the codebase.
    
    This endpoint uses vector embeddings to find similar code elements
    based on semantic similarity.
    """
    try:
        _, embedding_service, _ = get_services()
        results = embedding_service.search_similar(request.query, top_k=request.top_k)
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/trace")
async def trace_signals(request: TraceRequest):
    """
    Trace git history and runtime signals for a codebase.
    
    This endpoint analyzes git history, function churn, and runtime
    patterns to provide insights about code evolution.
    """
    try:
        if not os.path.exists(request.codebase_path):
            raise HTTPException(status_code=400, detail="Codebase path does not exist")
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        signal_tracer = SignalTracer(request.codebase_path, redis_url)
        
        # Trace git history
        git_signals = signal_tracer.trace_git_history(days_back=request.days_back)
        
        # Try to analyze function churn if parsed files exist
        churn_signals = []
        try:
            parsed_files_path = "codex-fabric-data/parsed_files.json"
            if os.path.exists(parsed_files_path):
                with open(parsed_files_path, "r") as f:
                    files_data = json.load(f)
                churn_signals = signal_tracer.analyze_function_churn(files_data)
        except:
            pass
        
        # Capture runtime signals
        runtime_signals = signal_tracer.capture_runtime_signals()
        
        # Calculate priority scores
        priority_scores = signal_tracer.calculate_priority_scores(churn_signals, runtime_signals)
        
        # Generate insights
        insights = signal_tracer.generate_insights(git_signals, churn_signals, runtime_signals)
        
        return {
            "success": True,
            "git_signals": len(git_signals),
            "churn_signals": len(churn_signals),
            "runtime_signals": len(runtime_signals),
            "priority_scores": len(priority_scores),
            "insights": insights
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal tracing failed: {str(e)}")

@app.post("/suggest")
async def get_ai_suggestions(request: QueryRequest):
    """
    Get AI-powered suggestions and insights about the codebase.
    
    This endpoint uses the reasoning agent to provide intelligent
    analysis and recommendations.
    """
    try:
        _, _, reasoning_agent = get_services()
        
        if reasoning_agent is None:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key required for AI suggestions"
            )
        
        result = reasoning_agent.answer_question(request.query)
        
        return {
            "success": result["success"],
            "answer": result["answer"],
            "tools_used": result["tools_used"],
            "context": result.get("context", {}),
            "error": result.get("error")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI suggestion failed: {str(e)}")

@app.get("/graph/stats")
async def get_graph_statistics():
    """Get statistics about the knowledge graph."""
    try:
        graph_builder, _, _ = get_services()
        
        # Get basic stats
        stats_query = "MATCH (n) RETURN labels(n) as type, count(n) as count"
        node_stats = graph_builder.query_graph(stats_query)
        
        # Get relationship stats
        rel_query = "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count"
        rel_stats = graph_builder.query_graph(rel_query)
        
        # Get centrality metrics
        centrality = graph_builder.get_centrality_metrics()
        
        # Find communities
        communities = graph_builder.find_communities()
        
        return {
            "success": True,
            "node_statistics": node_stats,
            "relationship_statistics": rel_stats,
            "centrality_metrics": centrality,
            "communities": len(communities)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph statistics: {str(e)}")

@app.get("/embeddings/stats")
async def get_embedding_statistics():
    """Get statistics about the embeddings."""
    try:
        _, embedding_service, _ = get_services()
        stats = embedding_service.get_embedding_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embedding statistics: {str(e)}")

@app.get("/similar/{function_id}")
async def find_similar_functions(function_id: str, top_k: int = 5):
    """Find functions similar to a given function."""
    try:
        _, embedding_service, _ = get_services()
        similar = embedding_service.find_similar_functions(function_id, top_k=top_k)
        
        return {
            "success": True,
            "function_id": function_id,
            "similar_functions": similar
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar functions: {str(e)}")

@app.get("/export/graph")
async def export_graph(format: str = "json"):
    """Export the knowledge graph in various formats."""
    try:
        graph_builder, _, _ = get_services()
        export_data = graph_builder.export_graph(format)
        
        return {
            "success": True,
            "format": format,
            "data": export_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export graph: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 