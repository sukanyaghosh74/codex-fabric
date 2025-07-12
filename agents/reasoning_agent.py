"""
Reasoning Agent Module

This module implements a multi-step reasoning agent using LangGraph and LangChain
that can answer complex questions about codebases and provide architectural insights.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
import openai


@dataclass
class AgentState:
    """State for the reasoning agent."""
    messages: List[BaseMessage]
    current_step: str
    context: Dict[str, Any]
    tools_used: List[str]
    final_answer: Optional[str] = None


class CodebaseQueryTool(BaseTool):
    """Tool for querying the codebase knowledge graph."""
    
    name = "codebase_query"
    description = "Query the codebase knowledge graph to find information about code elements"
    
    def __init__(self, graph_builder):
        super().__init__()
        self.graph_builder = graph_builder
    
    def _run(self, query: str) -> str:
        """Execute a Cypher query on the knowledge graph."""
        try:
            results = self.graph_builder.query_graph(query)
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error executing query: {e}"


class SemanticSearchTool(BaseTool):
    """Tool for semantic search in the codebase."""
    
    name = "semantic_search"
    description = "Search for similar code elements using semantic similarity"
    
    def __init__(self, embedding_service):
        super().__init__()
        self.embedding_service = embedding_service
    
    def _run(self, query: str) -> str:
        """Search for similar code elements."""
        try:
            results = self.embedding_service.search_similar(query, top_k=5)
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error in semantic search: {e}"


class SignalAnalysisTool(BaseTool):
    """Tool for analyzing code signals and patterns."""
    
    name = "signal_analysis"
    description = "Analyze git history, function churn, and runtime signals"
    
    def __init__(self, signal_tracer):
        super().__init__()
        self.signal_tracer = signal_tracer
    
    def _run(self, analysis_type: str) -> str:
        """Analyze different types of signals."""
        try:
            if analysis_type == "git_history":
                signals = self.signal_tracer.trace_git_history(days_back=30)
                return json.dumps([vars(signal) for signal in signals], indent=2)
            elif analysis_type == "function_churn":
                # This would need the parsed files
                return "Function churn analysis requires parsed files"
            else:
                return f"Unknown analysis type: {analysis_type}"
        except Exception as e:
            return f"Error in signal analysis: {e}"


class ReasoningAgent:
    """
    Multi-step reasoning agent for codebase analysis.
    
    Uses LangGraph to orchestrate complex reasoning tasks involving
    multiple tools and reasoning steps.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 graph_builder=None,
                 embedding_service=None,
                 signal_tracer=None):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        self.graph_builder = graph_builder
        self.embedding_service = embedding_service
        self.signal_tracer = signal_tracer
        
        # Initialize tools
        self.tools = []
        if graph_builder:
            self.tools.append(CodebaseQueryTool(graph_builder))
        if embedding_service:
            self.tools.append(SemanticSearchTool(embedding_service))
        if signal_tracer:
            self.tools.append(SignalAnalysisTool(signal_tracer))
        
        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools)
        
        # Build the agent graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_question", self._analyze_question)
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("reason", self._reason)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        
        # Add edges
        workflow.set_entry_point("analyze_question")
        workflow.add_edge("analyze_question", "gather_context")
        workflow.add_edge("gather_context", "reason")
        workflow.add_edge("reason", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)
        
        return workflow.compile()
    
    def _analyze_question(self, state: AgentState) -> AgentState:
        """Analyze the user's question to understand what information is needed."""
        messages = state.messages
        last_message = messages[-1].content
        
        analysis_prompt = f"""
        Analyze the following question about a codebase and determine what type of information is needed:
        
        Question: {last_message}
        
        Determine:
        1. What type of question this is (architecture, implementation, debugging, etc.)
        2. What specific information needs to be gathered
        3. What tools should be used
        
        Respond with a JSON object containing:
        {{
            "question_type": "string",
            "information_needed": ["list", "of", "required", "info"],
            "tools_to_use": ["list", "of", "tool", "names"],
            "reasoning_steps": ["list", "of", "reasoning", "steps"]
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        
        try:
            analysis = json.loads(response.content)
            state.context["question_analysis"] = analysis
            state.current_step = "gather_context"
        except json.JSONDecodeError:
            state.context["question_analysis"] = {
                "question_type": "general",
                "information_needed": ["codebase_structure"],
                "tools_to_use": ["codebase_query"],
                "reasoning_steps": ["gather_context", "reason", "synthesize"]
            }
            state.current_step = "gather_context"
        
        return state
    
    def _gather_context(self, state: AgentState) -> AgentState:
        """Gather relevant context using available tools."""
        analysis = state.context.get("question_analysis", {})
        tools_to_use = analysis.get("tools_to_use", [])
        
        context_data = {}
        
        for tool_name in tools_to_use:
            if tool_name == "codebase_query":
                # Use graph queries to gather information
                queries = self._generate_context_queries(analysis)
                for query_name, query in queries.items():
                    try:
                        result = self.graph_builder.query_graph(query)
                        context_data[query_name] = result
                        state.tools_used.append(f"codebase_query:{query_name}")
                    except Exception as e:
                        context_data[query_name] = f"Error: {e}"
            
            elif tool_name == "semantic_search":
                # Use semantic search
                last_message = state.messages[-1].content
                try:
                    results = self.embedding_service.search_similar(last_message, top_k=5)
                    context_data["semantic_search"] = results
                    state.tools_used.append("semantic_search")
                except Exception as e:
                    context_data["semantic_search"] = f"Error: {e}"
            
            elif tool_name == "signal_analysis":
                # Analyze signals
                try:
                    git_signals = self.signal_tracer.trace_git_history(days_back=30)
                    context_data["git_signals"] = [vars(signal) for signal in git_signals]
                    state.tools_used.append("signal_analysis")
                except Exception as e:
                    context_data["signal_analysis"] = f"Error: {e}"
        
        state.context["gathered_context"] = context_data
        state.current_step = "reason"
        
        return state
    
    def _generate_context_queries(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate Cypher queries based on the question analysis."""
        question_type = analysis.get("question_type", "general")
        queries = {}
        
        if question_type == "architecture":
            queries.update({
                "file_structure": "MATCH (f:File) RETURN f.path, f.language, f.size ORDER BY f.size DESC LIMIT 20",
                "class_hierarchy": "MATCH (c:Class)-[:INHERITS_FROM]->(parent) RETURN c.name, parent.name",
                "function_dependencies": "MATCH (f:Function)-[:DEPENDS_ON]->(dep) RETURN f.name, dep.name LIMIT 20"
            })
        elif question_type == "implementation":
            queries.update({
                "function_details": "MATCH (f:Function) RETURN f.name, f.file_path, f.line_start, f.line_end LIMIT 20",
                "imports": "MATCH (f:File)-[:IMPORTS]->(imp) RETURN f.path, imp.path LIMIT 20"
            })
        elif question_type == "debugging":
            queries.update({
                "error_prone_functions": "MATCH (f:Function) WHERE f.metadata CONTAINS 'error' RETURN f.name, f.file_path",
                "recent_changes": "MATCH (f:File) WHERE f.last_modified > datetime() - duration('P7D') RETURN f.path, f.last_modified"
            })
        else:
            # General queries
            queries.update({
                "overview": "MATCH (n) RETURN labels(n) as type, count(n) as count",
                "files_by_language": "MATCH (f:File) RETURN f.language, count(f) as count"
            })
        
        return queries
    
    def _reason(self, state: AgentState) -> AgentState:
        """Reason about the gathered context to formulate an answer."""
        messages = state.messages
        last_message = messages[-1].content
        context = state.context.get("gathered_context", {})
        analysis = state.context.get("question_analysis", {})
        
        reasoning_prompt = f"""
        Based on the following context and question, provide a detailed reasoning process:
        
        Question: {last_message}
        Question Type: {analysis.get('question_type', 'general')}
        
        Context Data:
        {json.dumps(context, indent=2)}
        
        Please provide:
        1. Key insights from the context
        2. How the context relates to the question
        3. What conclusions can be drawn
        4. Any additional information that might be needed
        
        Structure your response as a JSON object:
        {{
            "insights": ["list", "of", "key", "insights"],
            "relationships": "how context relates to question",
            "conclusions": ["list", "of", "conclusions"],
            "missing_info": ["any", "missing", "information"],
            "recommendations": ["any", "recommendations"]
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=reasoning_prompt)])
        
        try:
            reasoning = json.loads(response.content)
            state.context["reasoning"] = reasoning
        except json.JSONDecodeError:
            state.context["reasoning"] = {
                "insights": ["Unable to parse reasoning"],
                "relationships": "Context analysis failed",
                "conclusions": ["Need more information"],
                "missing_info": ["Detailed context"],
                "recommendations": ["Re-run with more specific query"]
            }
        
        state.current_step = "synthesize_answer"
        return state
    
    def _synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize the final answer based on reasoning."""
        messages = state.messages
        last_message = messages[-1].content
        reasoning = state.context.get("reasoning", {})
        analysis = state.context.get("question_analysis", {})
        
        synthesis_prompt = f"""
        Synthesize a comprehensive answer to the user's question based on the reasoning:
        
        Original Question: {last_message}
        Question Type: {analysis.get('question_type', 'general')}
        
        Reasoning Results:
        {json.dumps(reasoning, indent=2)}
        
        Provide a clear, well-structured answer that:
        1. Directly addresses the user's question
        2. Includes relevant insights from the analysis
        3. Provides actionable recommendations if applicable
        4. Cites specific code elements or patterns when relevant
        5. Acknowledges any limitations or missing information
        
        Format your response as a comprehensive, well-structured answer suitable for a senior engineer.
        """
        
        response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        state.final_answer = response.content
        state.current_step = "complete"
        
        return state
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the codebase.
        
        Args:
            question: The user's question about the codebase
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=question)],
            current_step="start",
            context={},
            tools_used=[]
        )
        
        # Execute the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "answer": final_state.final_answer,
                "tools_used": final_state.tools_used,
                "context": final_state.context,
                "success": True
            }
        
        except Exception as e:
            return {
                "answer": f"Error processing question: {e}",
                "tools_used": [],
                "context": {},
                "success": False,
                "error": str(e)
            }
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get information about the agent's capabilities."""
        return {
            "tools_available": [tool.name for tool in self.tools],
            "question_types": [
                "architecture_analysis",
                "implementation_details", 
                "debugging_assistance",
                "code_review",
                "refactoring_suggestions",
                "dependency_analysis",
                "performance_analysis"
            ],
            "model": "gpt-4-turbo-preview",
            "workflow_steps": [
                "analyze_question",
                "gather_context", 
                "reason",
                "synthesize_answer"
            ]
        } 