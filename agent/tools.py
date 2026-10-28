"""
Context Rot Firewall — Agent Tools
Simple tools the baseline agent can call.
These intentionally produce verbose output to generate realistic context growth.
"""
import os
import subprocess
import json
from pathlib import Path


def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    try:
        entries = sorted(Path(path).iterdir())
        lines = []
        for e in entries:
            size = e.stat().st_size if e.is_file() else ""
            lines.append(f"{'[DIR] ' if e.is_dir() else '      '}{e.name}{f'  ({size} bytes)' if size else ''}")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"ERROR listing {path}: {e}"


def run_shell(command: str, cwd: str = ".") -> str:
    """Run a shell command and return stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        output += f"\nEXIT CODE: {result.returncode}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR running command: {e}"


def search_in_file(path: str, query: str) -> str:
    """Search for lines containing query in a file."""
    try:
        lines = Path(path).read_text(encoding="utf-8").split("\n")
        matches = [(i + 1, line) for i, line in enumerate(lines) if query.lower() in line.lower()]
        if not matches:
            return f"No matches for '{query}' in {path}"
        return "\n".join(f"Line {ln}: {txt}" for ln, txt in matches[:20])
    except Exception as e:
        return f"ERROR searching {path}: {e}"


# Tool registry for the agent loop
TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_shell": run_shell,
    "search_in_file": search_in_file,
}

# Tool descriptions for Ollama function-calling format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path (default: '.')"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Working directory (default: '.')"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search for lines containing a query string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["path", "query"],
            },
        },
    },
]
