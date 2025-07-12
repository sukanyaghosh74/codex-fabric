"""
Codex Fabric CLI Main Module

This module provides the main CLI interface for the Codex Fabric system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parser import CodeParser
from core.graph_builder import GraphBuilder
from core.signal_tracer import SignalTracer
from core.embedding_service import EmbeddingService
from agents.reasoning_agent import ReasoningAgent

app = typer.Typer(
    name="cfabric",
    help="Codex Fabric - Transform codebases into intelligent knowledge graphs",
    add_completion=False
)

console = Console()


@app.command()
def init(
    path: str = typer.Argument(..., help="Path to the codebase to analyze"),
    output: str = typer.Option("codex-fabric-data", help="Output directory for analysis data"),
    exclude: List[str] = typer.Option([], help="Patterns to exclude from analysis"),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", help="Neo4j database URI"),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j username"),
    neo4j_password: str = typer.Option("password", help="Neo4j password"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant vector database URL"),
    openai_api_key: Optional[str] = typer.Option(None, help="OpenAI API key for embeddings")
):
    """
    Initialize Codex Fabric analysis for a codebase.
    
    This command will:
    1. Parse the codebase and generate AST representations
    2. Build a knowledge graph in Neo4j
    3. Generate embeddings for code elements
    4. Store the results for further analysis
    """
    
    if not os.path.exists(path):
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold blue]Codex Fabric[/bold blue]\n"
        f"Analyzing codebase: [green]{path}[/green]\n"
        f"Output directory: [green]{output_path}[/green]",
        title="Initialization"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Parse codebase
        task1 = progress.add_task("Parsing codebase...", total=None)
        parser = CodeParser()
        files = parser.parse_directory(path, exclude_patterns=exclude)
        progress.update(task1, description="✅ Codebase parsed", completed=True)
        
        console.print(f"[green]Parsed {len(files)} files[/green]")
        
        # Step 2: Build knowledge graph
        task2 = progress.add_task("Building knowledge graph...", total=None)
        graph_builder = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
        graph_stats = graph_builder.build_graph(files)
        progress.update(task2, description="✅ Knowledge graph built", completed=True)
        
        console.print(f"[green]Built graph with {graph_stats['total_nodes']} nodes and {graph_stats['total_relationships']} relationships[/green]")
        
        # Step 3: Generate embeddings
        task3 = progress.add_task("Generating embeddings...", total=None)
        embedding_service = EmbeddingService(
            qdrant_url=qdrant_url,
            openai_api_key=openai_api_key
        )
        
        # Collect all code nodes
        all_nodes = []
        for file_info in files.values():
            all_nodes.extend(file_info.nodes)
        
        embeddings = embedding_service.embed_code_elements(all_nodes)
        embedding_service.store_embeddings(embeddings)
        progress.update(task3, description="✅ Embeddings generated", completed=True)
        
        console.print(f"[green]Generated {len(embeddings)} embeddings[/green]")
        
        # Step 4: Save analysis data
        task4 = progress.add_task("Saving analysis data...", total=None)
        
        # Save parsed files
        files_data = {}
        for file_path, file_info in files.items():
            files_data[file_path] = {
                "path": file_info.path,
                "language": file_info.language.value,
                "size": file_info.size,
                "lines": file_info.lines,
                "nodes": [
                    {
                        "id": node.id,
                        "name": node.name,
                        "type": node.type,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                        "docstring": node.docstring,
                        "metadata": node.metadata
                    }
                    for node in file_info.nodes
                ],
                "imports": file_info.imports,
                "exports": file_info.exports
            }
        
        with open(output_path / "parsed_files.json", "w") as f:
            json.dump(files_data, f, indent=2)
        
        # Save graph statistics
        with open(output_path / "graph_stats.json", "w") as f:
            json.dump(graph_stats, f, indent=2)
        
        # Save embedding statistics
        embedding_stats = embedding_service.get_embedding_statistics()
        with open(output_path / "embedding_stats.json", "w") as f:
            json.dump(embedding_stats, f, indent=2)
        
        progress.update(task4, description="✅ Analysis data saved", completed=True)
    
    # Display summary
    console.print("\n[bold green]✅ Analysis Complete![/bold green]")
    
    summary_table = Table(title="Analysis Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Files Analyzed", str(len(files)))
    summary_table.add_row("Graph Nodes", str(graph_stats["total_nodes"]))
    summary_table.add_row("Graph Relationships", str(graph_stats["total_relationships"]))
    summary_table.add_row("Embeddings Generated", str(len(embeddings)))
    summary_table.add_row("Languages Found", ", ".join(graph_stats["languages"]))
    
    console.print(summary_table)
    
    console.print(f"\n[blue]Data saved to: {output_path}[/blue]")
    console.print("[yellow]Next steps:[/yellow]")
    console.print("  • Run 'cfabric trace' to analyze git history and signals")
    console.print("  • Run 'cfabric suggest' to get AI-powered insights")
    console.print("  • Run 'cfabric serve' to start the web dashboard")


@app.command()
def trace(
    path: str = typer.Argument(..., help="Path to the codebase to trace"),
    days_back: int = typer.Option(30, help="Number of days to look back in git history"),
    output: str = typer.Option("codex-fabric-signals", help="Output directory for signal data"),
    redis_url: str = typer.Option("redis://localhost:6379", help="Redis URL for caching")
):
    """
    Trace git history and runtime signals for a codebase.
    
    This command will:
    1. Analyze git commit history and patterns
    2. Calculate function churn and volatility scores
    3. Capture runtime execution signals (if available)
    4. Generate insights about code evolution
    """
    
    if not os.path.exists(path):
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        f"[bold blue]Codex Fabric Signal Tracing[/bold blue]\n"
        f"Tracing codebase: [green]{path}[/green]\n"
        f"Looking back: [green]{days_back} days[/green]",
        title="Signal Tracing"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Initialize signal tracer
        task1 = progress.add_task("Initializing signal tracer...", total=None)
        signal_tracer = SignalTracer(path, redis_url)
        progress.update(task1, description="✅ Signal tracer initialized", completed=True)
        
        # Step 2: Trace git history
        task2 = progress.add_task("Tracing git history...", total=None)
        git_signals = signal_tracer.trace_git_history(days_back=days_back)
        progress.update(task2, description="✅ Git history traced", completed=True)
        
        console.print(f"[green]Traced {len(git_signals)} git signals[/green]")
        
        # Step 3: Analyze function churn (if parsed files available)
        task3 = progress.add_task("Analyzing function churn...", total=None)
        
        # Try to load parsed files
        parsed_files_path = Path("codex-fabric-data/parsed_files.json")
        if parsed_files_path.exists():
            with open(parsed_files_path, "r") as f:
                files_data = json.load(f)
            churn_signals = signal_tracer.analyze_function_churn(files_data)
            progress.update(task3, description="✅ Function churn analyzed", completed=True)
            console.print(f"[green]Analyzed churn for {len(churn_signals)} functions[/green]")
        else:
            progress.update(task3, description="⚠️ No parsed files found for churn analysis", completed=True)
            churn_signals = []
            console.print("[yellow]Warning: No parsed files found. Run 'cfabric init' first for churn analysis.[/yellow]")
        
        # Step 4: Capture runtime signals
        task4 = progress.add_task("Capturing runtime signals...", total=None)
        runtime_signals = signal_tracer.capture_runtime_signals()
        progress.update(task4, description="✅ Runtime signals captured", completed=True)
        
        console.print(f"[green]Captured {len(runtime_signals)} runtime signals[/green]")
        
        # Step 5: Calculate priority scores
        task5 = progress.add_task("Calculating priority scores...", total=None)
        priority_scores = signal_tracer.calculate_priority_scores(churn_signals, runtime_signals)
        progress.update(task5, description="✅ Priority scores calculated", completed=True)
        
        # Step 6: Generate insights
        task6 = progress.add_task("Generating insights...", total=None)
        insights = signal_tracer.generate_insights(git_signals, churn_signals, runtime_signals)
        progress.update(task6, description="✅ Insights generated", completed=True)
        
        # Step 7: Save signal data
        task7 = progress.add_task("Saving signal data...", total=None)
        
        # Save git signals
        git_signals_data = [vars(signal) for signal in git_signals]
        with open(output_path / "git_signals.json", "w") as f:
            json.dump(git_signals_data, f, indent=2, default=str)
        
        # Save churn signals
        churn_signals_data = [vars(signal) for signal in churn_signals]
        with open(output_path / "churn_signals.json", "w") as f:
            json.dump(churn_signals_data, f, indent=2, default=str)
        
        # Save runtime signals
        runtime_signals_data = [vars(signal) for signal in runtime_signals]
        with open(output_path / "runtime_signals.json", "w") as f:
            json.dump(runtime_signals_data, f, indent=2, default=str)
        
        # Save priority scores
        with open(output_path / "priority_scores.json", "w") as f:
            json.dump(priority_scores, f, indent=2)
        
        # Save insights
        with open(output_path / "insights.json", "w") as f:
            json.dump(insights, f, indent=2, default=str)
        
        progress.update(task7, description="✅ Signal data saved", completed=True)
    
    # Display summary
    console.print("\n[bold green]✅ Signal Tracing Complete![/bold green]")
    
    summary_table = Table(title="Signal Tracing Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Git Signals", str(len(git_signals)))
    summary_table.add_row("Churn Signals", str(len(churn_signals)))
    summary_table.add_row("Runtime Signals", str(len(runtime_signals)))
    summary_table.add_row("Priority Scores", str(len(priority_scores)))
    
    console.print(summary_table)
    
    # Display key insights
    if insights:
        console.print("\n[bold blue]Key Insights:[/bold blue]")
        
        if insights.get("most_active_developers"):
            console.print(f"👥 Most Active Developer: {insights['most_active_developers'][0][0]} ({insights['most_active_developers'][0][1]} commits)")
        
        if insights.get("most_changed_files"):
            console.print(f"📝 Most Changed File: {insights['most_changed_files'][0][0]} ({insights['most_changed_files'][0][1]} changes)")
        
        if insights.get("recent_activity"):
            recent = insights["recent_activity"]
            console.print(f"📊 Recent Activity: {recent['commits_last_week']} commits, {recent['files_changed_last_week']} files changed")
    
    console.print(f"\n[blue]Signal data saved to: {output_path}[/blue]")


@app.command()
def suggest(
    query: str = typer.Argument(..., help="Question or query about the codebase"),
    openai_api_key: Optional[str] = typer.Option(None, help="OpenAI API key"),
    neo4j_uri: str = typer.Option("bolt://localhost:7687", help="Neo4j database URI"),
    neo4j_user: str = typer.Option("neo4j", help="Neo4j username"),
    neo4j_password: str = typer.Option("password", help="Neo4j password"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant vector database URL")
):
    """
    Get AI-powered insights and suggestions about the codebase.
    
    This command uses the reasoning agent to answer questions about:
    • Code architecture and patterns
    • Implementation details and dependencies
    • Debugging assistance and error analysis
    • Refactoring suggestions and improvements
    """
    
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            console.print("[red]Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --openai-api-key option.[/red]")
            raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold blue]Codex Fabric AI Assistant[/bold blue]\n"
        f"Query: [green]{query}[/green]",
        title="AI Analysis"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Step 1: Initialize services
        task1 = progress.add_task("Initializing AI services...", total=None)
        
        graph_builder = GraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
        embedding_service = EmbeddingService(qdrant_url=qdrant_url)
        
        # Initialize reasoning agent
        reasoning_agent = ReasoningAgent(
            openai_api_key=openai_api_key,
            graph_builder=graph_builder,
            embedding_service=embedding_service
        )
        
        progress.update(task1, description="✅ AI services initialized", completed=True)
        
        # Step 2: Process query
        task2 = progress.add_task("Processing query...", total=None)
        result = reasoning_agent.answer_question(query)
        progress.update(task2, description="✅ Query processed", completed=True)
    
    # Display results
    if result["success"]:
        console.print("\n[bold green]✅ Analysis Complete![/bold green]")
        
        # Display the answer
        console.print("\n[bold blue]Answer:[/bold blue]")
        console.print(Markdown(result["answer"]))
        
        # Display metadata
        if result["tools_used"]:
            console.print(f"\n[blue]Tools used:[/blue] {', '.join(result['tools_used'])}")
        
        # Display context summary
        if result["context"]:
            console.print(f"\n[blue]Analysis type:[/blue] {result['context'].get('question_analysis', {}).get('question_type', 'general')}")
    
    else:
        console.print(f"\n[red]❌ Error: {result['answer']}[/red]")
        if "error" in result:
            console.print(f"[red]Details: {result['error']}[/red]")


@app.command()
def serve(
    host: str = typer.Option("localhost", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development")
):
    """
    Start the Codex Fabric web dashboard.
    
    This will start a FastAPI server with the web interface for
    exploring the knowledge graph and interacting with AI agents.
    """
    
    console.print(Panel.fit(
        f"[bold blue]Codex Fabric Web Dashboard[/bold blue]\n"
        f"Starting server on: [green]http://{host}:{port}[/green]",
        title="Web Server"
    ))
    
    try:
        import uvicorn
        from api.main import app as fastapi_app
        
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    
    except ImportError:
        console.print("[red]Error: FastAPI not installed. Install with 'pip install fastapi uvicorn'[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Check the status of Codex Fabric services and data.
    """
    
    console.print(Panel.fit(
        "[bold blue]Codex Fabric Status Check[/bold blue]",
        title="Status"
    ))
    
    status_table = Table(title="Service Status")
    status_table.add_column("Service", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")
    
    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            status_table.add_row("Neo4j", "✅ Connected", "Graph database ready")
        driver.close()
    except Exception as e:
        status_table.add_row("Neo4j", "❌ Disconnected", str(e))
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient("http://localhost:6333")
        collections = client.get_collections()
        status_table.add_row("Qdrant", "✅ Connected", f"{len(collections.collections)} collections")
    except Exception as e:
        status_table.add_row("Qdrant", "❌ Disconnected", str(e))
    
    # Check Redis
    try:
        import redis
        r = redis.from_url("redis://localhost:6379")
        r.ping()
        status_table.add_row("Redis", "✅ Connected", "Cache ready")
    except Exception as e:
        status_table.add_row("Redis", "❌ Disconnected", str(e))
    
    # Check data files
    data_paths = [
        ("Parsed Files", "codex-fabric-data/parsed_files.json"),
        ("Graph Stats", "codex-fabric-data/graph_stats.json"),
        ("Embeddings", "codex-fabric-data/embedding_stats.json"),
        ("Git Signals", "codex-fabric-signals/git_signals.json")
    ]
    
    for name, path in data_paths:
        if os.path.exists(path):
            size = os.path.getsize(path)
            status_table.add_row(name, "✅ Available", f"{size} bytes")
        else:
            status_table.add_row(name, "❌ Missing", "Not found")
    
    console.print(status_table)


if __name__ == "__main__":
    app() 