"""
Signal Tracer Module

This module captures various signals from the codebase including git history,
function churn, commit patterns, and runtime traces to enhance the knowledge graph.
"""

import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import git
from git import Repo
import redis
from collections import defaultdict, Counter


@dataclass
class GitSignal:
    """Represents a git-related signal."""
    file_path: str
    commit_hash: str
    author: str
    timestamp: datetime
    lines_added: int
    lines_deleted: int
    message: str
    files_changed: List[str]


@dataclass
class ChurnSignal:
    """Represents function churn information."""
    function_id: str
    file_path: str
    function_name: str
    change_count: int
    last_modified: datetime
    volatility_score: float
    complexity_score: float


@dataclass
class RuntimeSignal:
    """Represents runtime execution signals."""
    function_id: str
    execution_count: int
    average_duration: float
    error_rate: float
    last_executed: datetime
    call_stack: List[str]


class SignalTracer:
    """
    Captures and analyzes various signals from the codebase.
    
    Tracks git history, function churn, and runtime patterns to
    provide insights into code evolution and usage patterns.
    """
    
    def __init__(self, repo_path: str, redis_url: str = "redis://localhost:6379"):
        self.repo_path = Path(repo_path)
        self.redis_client = redis.from_url(redis_url)
        self.repo = None
        self._initialize_repo()
    
    def _initialize_repo(self):
        """Initialize git repository connection."""
        try:
            self.repo = Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            print(f"Warning: {self.repo_path} is not a git repository")
            self.repo = None
    
    def trace_git_history(self, days_back: int = 30) -> List[GitSignal]:
        """
        Trace git history for the specified time period.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of GitSignal objects
        """
        if not self.repo:
            return []
        
        signals = []
        since_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Get commits since the specified date
            commits = list(self.repo.iter_commits(
                since=since_date,
                all=True
            ))
            
            for commit in commits:
                # Get commit stats
                stats = commit.stats
                
                for file_path, file_stats in stats.files.items():
                    signal = GitSignal(
                        file_path=file_path,
                        commit_hash=commit.hexsha,
                        author=commit.author.name,
                        timestamp=datetime.fromtimestamp(commit.committed_date),
                        lines_added=file_stats.get('insertions', 0),
                        lines_deleted=file_stats.get('deletions', 0),
                        message=commit.message.strip(),
                        files_changed=list(stats.files.keys())
                    )
                    signals.append(signal)
        
        except Exception as e:
            print(f"Error tracing git history: {e}")
        
        return signals
    
    def analyze_function_churn(self, files: Dict[str, Any]) -> List[ChurnSignal]:
        """
        Analyze function churn based on git history.
        
        Args:
            files: Dictionary of parsed files from the parser
            
        Returns:
            List of ChurnSignal objects
        """
        if not self.repo:
            return []
        
        churn_signals = []
        
        try:
            # Get all commits
            commits = list(self.repo.iter_commits(all=True))
            
            # Track changes per function
            function_changes = defaultdict(list)
            
            for commit in commits:
                # Get diff for this commit
                if commit.parents:
                    diff = commit.parents[0].diff(commit)
                    
                    for change in diff:
                        if change.a_path and change.b_path:
                            file_path = change.b_path
                            
                            # Find functions in this file
                            if file_path in files:
                                file_info = files[file_path]
                                
                                for node in file_info.nodes:
                                    if node.type in ["function", "method"]:
                                        function_id = node.id
                                        
                                        # Check if this function was modified
                                        if self._function_was_modified(node, change):
                                            function_changes[function_id].append({
                                                'commit': commit.hexsha,
                                                'timestamp': datetime.fromtimestamp(commit.committed_date),
                                                'lines_added': change.stats.get('insertions', 0),
                                                'lines_deleted': change.stats.get('deletions', 0)
                                            })
            
            # Calculate churn metrics for each function
            for function_id, changes in function_changes.items():
                # Find the function info
                function_info = self._find_function_info(function_id, files)
                if not function_info:
                    continue
                
                # Calculate metrics
                change_count = len(changes)
                last_modified = max(change['timestamp'] for change in changes)
                
                # Calculate volatility score (frequency of changes)
                days_since_first = (datetime.now() - min(change['timestamp'] for change in changes)).days
                volatility_score = change_count / max(days_since_first, 1)
                
                # Calculate complexity score (lines of code)
                complexity_score = len(function_info.content.splitlines())
                
                signal = ChurnSignal(
                    function_id=function_id,
                    file_path=function_info.file_path,
                    function_name=function_info.name,
                    change_count=change_count,
                    last_modified=last_modified,
                    volatility_score=volatility_score,
                    complexity_score=complexity_score
                )
                churn_signals.append(signal)
        
        except Exception as e:
            print(f"Error analyzing function churn: {e}")
        
        return churn_signals
    
    def _function_was_modified(self, node: Any, change: Any) -> bool:
        """Check if a function was modified in a git change."""
        try:
            # Get the function's line range
            start_line = node.line_start
            end_line = node.line_end
            
            # Check if any lines in the diff overlap with the function
            for hunk in change.diff.decode('utf-8').split('\n'):
                if hunk.startswith('@@'):
                    # Parse hunk header to get line numbers
                    match = re.search(r'@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@', hunk)
                    if match:
                        old_start = int(match.group(1))
                        new_start = int(match.group(3))
                        
                        # Check for overlap
                        if (old_start <= end_line and 
                            (not match.group(2) or old_start + int(match.group(2)) >= start_line)):
                            return True
            
            return False
        
        except Exception:
            return False
    
    def _find_function_info(self, function_id: str, files: Dict[str, Any]) -> Optional[Any]:
        """Find function information by ID."""
        for file_info in files.values():
            for node in file_info.nodes:
                if node.id == function_id:
                    return node
        return None
    
    def capture_runtime_signals(self, trace_file: str = None) -> List[RuntimeSignal]:
        """
        Capture runtime execution signals.
        
        Args:
            trace_file: Optional file containing runtime traces
            
        Returns:
            List of RuntimeSignal objects
        """
        signals = []
        
        # Try to load from trace file if provided
        if trace_file and os.path.exists(trace_file):
            signals.extend(self._load_runtime_traces(trace_file))
        
        # Try to get signals from Redis cache
        signals.extend(self._get_cached_runtime_signals())
        
        return signals
    
    def _load_runtime_traces(self, trace_file: str) -> List[RuntimeSignal]:
        """Load runtime traces from a file."""
        signals = []
        
        try:
            with open(trace_file, 'r') as f:
                traces = json.load(f)
            
            for trace in traces:
                signal = RuntimeSignal(
                    function_id=trace.get('function_id'),
                    execution_count=trace.get('execution_count', 0),
                    average_duration=trace.get('average_duration', 0.0),
                    error_rate=trace.get('error_rate', 0.0),
                    last_executed=datetime.fromisoformat(trace.get('last_executed')),
                    call_stack=trace.get('call_stack', [])
                )
                signals.append(signal)
        
        except Exception as e:
            print(f"Error loading runtime traces: {e}")
        
        return signals
    
    def _get_cached_runtime_signals(self) -> List[RuntimeSignal]:
        """Get runtime signals from Redis cache."""
        signals = []
        
        try:
            # Get all keys related to runtime signals
            keys = self.redis_client.keys("runtime:*")
            
            for key in keys:
                data = self.redis_client.hgetall(key)
                if data:
                    signal = RuntimeSignal(
                        function_id=data.get(b'function_id', b'').decode('utf-8'),
                        execution_count=int(data.get(b'execution_count', 0)),
                        average_duration=float(data.get(b'average_duration', 0.0)),
                        error_rate=float(data.get(b'error_rate', 0.0)),
                        last_executed=datetime.fromisoformat(
                            data.get(b'last_executed', b'').decode('utf-8')
                        ),
                        call_stack=json.loads(
                            data.get(b'call_stack', b'[]').decode('utf-8')
                        )
                    )
                    signals.append(signal)
        
        except Exception as e:
            print(f"Error getting cached runtime signals: {e}")
        
        return signals
    
    def calculate_priority_scores(self, churn_signals: List[ChurnSignal], 
                                runtime_signals: List[RuntimeSignal]) -> Dict[str, float]:
        """
        Calculate priority scores for functions based on various signals.
        
        Args:
            churn_signals: List of churn signals
            runtime_signals: List of runtime signals
            
        Returns:
            Dictionary mapping function IDs to priority scores
        """
        priority_scores = {}
        
        # Create lookup dictionaries
        churn_lookup = {signal.function_id: signal for signal in churn_signals}
        runtime_lookup = {signal.function_id: signal for signal in runtime_signals}
        
        # Calculate scores for all functions
        all_function_ids = set(churn_lookup.keys()) | set(runtime_lookup.keys())
        
        for function_id in all_function_ids:
            score = 0.0
            
            # Churn-based scoring
            if function_id in churn_lookup:
                churn_signal = churn_lookup[function_id]
                
                # Higher score for more volatile functions
                score += churn_signal.volatility_score * 10
                
                # Higher score for recently modified functions
                days_since_modified = (datetime.now() - churn_signal.last_modified).days
                recency_factor = max(0, 30 - days_since_modified) / 30
                score += recency_factor * 5
            
            # Runtime-based scoring
            if function_id in runtime_lookup:
                runtime_signal = runtime_lookup[function_id]
                
                # Higher score for frequently executed functions
                score += min(runtime_signal.execution_count / 100, 10)
                
                # Higher score for slow functions
                if runtime_signal.average_duration > 1.0:
                    score += min(runtime_signal.average_duration, 10)
                
                # Higher score for error-prone functions
                score += runtime_signal.error_rate * 20
            
            priority_scores[function_id] = score
        
        return priority_scores
    
    def generate_insights(self, git_signals: List[GitSignal], 
                         churn_signals: List[ChurnSignal],
                         runtime_signals: List[RuntimeSignal]) -> Dict[str, Any]:
        """
        Generate insights from collected signals.
        
        Args:
            git_signals: List of git signals
            churn_signals: List of churn signals
            runtime_signals: List of runtime signals
            
        Returns:
            Dictionary containing various insights
        """
        insights = {
            "most_active_developers": self._get_most_active_developers(git_signals),
            "most_changed_files": self._get_most_changed_files(git_signals),
            "most_volatile_functions": self._get_most_volatile_functions(churn_signals),
            "performance_hotspots": self._get_performance_hotspots(runtime_signals),
            "error_prone_functions": self._get_error_prone_functions(runtime_signals),
            "recent_activity": self._get_recent_activity(git_signals),
            "commit_patterns": self._analyze_commit_patterns(git_signals)
        }
        
        return insights
    
    def _get_most_active_developers(self, git_signals: List[GitSignal]) -> List[Tuple[str, int]]:
        """Get the most active developers."""
        author_counts = Counter(signal.author for signal in git_signals)
        return author_counts.most_common(10)
    
    def _get_most_changed_files(self, git_signals: List[GitSignal]) -> List[Tuple[str, int]]:
        """Get the most changed files."""
        file_changes = Counter()
        for signal in git_signals:
            file_changes[signal.file_path] += signal.lines_added + signal.lines_deleted
        return file_changes.most_common(10)
    
    def _get_most_volatile_functions(self, churn_signals: List[ChurnSignal]) -> List[Tuple[str, float]]:
        """Get the most volatile functions."""
        volatility_scores = [(signal.function_name, signal.volatility_score) 
                            for signal in churn_signals]
        return sorted(volatility_scores, key=lambda x: x[1], reverse=True)[:10]
    
    def _get_performance_hotspots(self, runtime_signals: List[RuntimeSignal]) -> List[Tuple[str, float]]:
        """Get performance hotspots."""
        performance_scores = [(signal.function_id, signal.average_duration) 
                             for signal in runtime_signals]
        return sorted(performance_scores, key=lambda x: x[1], reverse=True)[:10]
    
    def _get_error_prone_functions(self, runtime_signals: List[RuntimeSignal]) -> List[Tuple[str, float]]:
        """Get error-prone functions."""
        error_scores = [(signal.function_id, signal.error_rate) 
                       for signal in runtime_signals]
        return sorted(error_scores, key=lambda x: x[1], reverse=True)[:10]
    
    def _get_recent_activity(self, git_signals: List[GitSignal]) -> Dict[str, int]:
        """Get recent activity statistics."""
        now = datetime.now()
        recent_signals = [s for s in git_signals 
                         if (now - s.timestamp).days <= 7]
        
        return {
            "commits_last_week": len(set(s.commit_hash for s in recent_signals)),
            "files_changed_last_week": len(set(s.file_path for s in recent_signals)),
            "lines_added_last_week": sum(s.lines_added for s in recent_signals),
            "lines_deleted_last_week": sum(s.lines_deleted for s in recent_signals)
        }
    
    def _analyze_commit_patterns(self, git_signals: List[GitSignal]) -> Dict[str, Any]:
        """Analyze commit patterns."""
        commit_messages = [signal.message for signal in git_signals]
        
        # Analyze commit message patterns
        patterns = {
            "bug_fixes": len([msg for msg in commit_messages 
                            if any(word in msg.lower() for word in ['fix', 'bug', 'issue'])]),
            "features": len([msg for msg in commit_messages 
                           if any(word in msg.lower() for word in ['add', 'feature', 'implement'])]),
            "refactors": len([msg for msg in commit_messages 
                            if any(word in msg.lower() for word in ['refactor', 'cleanup', 'restructure'])]),
            "documentation": len([msg for msg in commit_messages 
                                if any(word in msg.lower() for word in ['doc', 'readme', 'comment'])])
        }
        
        return patterns
    
    def export_signals(self, output_file: str, signals: Dict[str, Any]):
        """Export signals to a JSON file."""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_signals = {}
            for key, value in signals.items():
                if isinstance(value, list):
                    serializable_signals[key] = []
                    for item in value:
                        if hasattr(item, '__dict__'):
                            item_dict = item.__dict__.copy()
                            # Convert datetime objects
                            for k, v in item_dict.items():
                                if isinstance(v, datetime):
                                    item_dict[k] = v.isoformat()
                            serializable_signals[key].append(item_dict)
                        else:
                            serializable_signals[key].append(item)
                else:
                    serializable_signals[key] = value
            
            with open(output_file, 'w') as f:
                json.dump(serializable_signals, f, indent=2)
            
            print(f"Signals exported to {output_file}")
        
        except Exception as e:
            print(f"Error exporting signals: {e}") 