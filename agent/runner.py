"""
Context Rot Firewall — Agent Runner
A minimal tool-calling loop that runs against a local Ollama model.
Supports two modes:
  - baseline: no context management (raw context growth)
  - firewall: context is processed by FirewallInterceptor on every turn

Usage:
    python -m agent.runner --task "Write a Python script that..." --mode firewall
    python -m agent.runner --task "Write a Python script that..." --mode baseline
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pprint

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools import TOOLS, TOOL_SCHEMAS
from firewall.interceptor import FirewallInterceptor
from firewall.config_loader import cfg

console = Console()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
You must complete the user's task step-by-step using the available tools.
Use tools to gather information, write files, and verify your work.
Always reason about what you need to do next before calling a tool.
When the task is fully complete, say TASK_COMPLETE and summarize what you did."""


def call_ollama(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call Ollama /api/chat and return the parsed response dict."""
    payload = {
        "model": cfg["llm"]["ollama_model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": cfg["llm"]["temperature"],
            "num_ctx": cfg["context"]["token_budget"] + 512,  # slight headroom
        },
    }
    if tools:
        payload["tools"] = tools

    resp = httpx.post(
        f"{cfg['llm']['ollama_url']}/api/chat",
        json=payload,
        timeout=cfg["llm"]["timeout_seconds"],
    )
    resp.raise_for_status()
    return resp.json()


def execute_tool_call(tool_name: str, tool_args: dict) -> str:
    """Execute a tool and return its string output."""
    fn = TOOLS.get(tool_name)
    if fn is None:
        return f"ERROR: Unknown tool '{tool_name}'"
    try:
        return fn(**tool_args)
    except Exception as e:
        return f"ERROR calling {tool_name}: {e}"


def run_agent(
    task: str,
    mode: str = "firewall",
    max_turns: int = 30,
    log_dir: str | None = None,
    initial_messages: list[dict] | None = None,
) -> dict:
    """
    Run the agent on a task.

    Args:
        task: The task description for the agent.
        mode: "baseline" (no firewall) or "firewall" (with context management).
        max_turns: Maximum number of agent turns before stopping.
        log_dir: Directory to write telemetry logs.
        initial_messages: Optional list of pre-seeded messages to inject into the
            history before the task message. Used by the benchmark runner to
            simulate long prior conversations (e.g. Forgotten Requirement scenario).

    Returns:
        Result dict with transcript, token stats, and success flag.
    """
    log_path = Path(log_dir or cfg["storage"]["log_dir"])
    log_path.mkdir(parents=True, exist_ok=True)

    firewall = FirewallInterceptor(enabled=(mode == "firewall"), log_dir=str(log_path))

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Inject preamble messages (for benchmark scenarios)
    if initial_messages:
        messages.extend(initial_messages)

    messages.append({"role": "user", "content": task})

    transcript = []
    total_input_tokens = 0
    turn = 0

    console.print(Panel(f"[bold cyan]Context Rot Firewall Agent[/bold cyan]\nMode: [yellow]{mode}[/yellow]\nTask: {task[:200]}"))

    while turn < max_turns:
        turn += 1
        console.print(f"\n[bold]Turn {turn}[/bold]")

        # Run firewall (or passthrough in baseline mode)
        managed_messages, fw_stats = firewall.process(messages, turn_index=turn)

        if mode == "firewall":
            console.print(
                f"  [dim]Firewall: {fw_stats['assemble']['final_tokens']} tokens "
                f"({fw_stats['assemble']['utilization_pct']}% budget) | "
                f"policy: {fw_stats['policy']} | "
                f"retrieved: {fw_stats['retrieved_count']}[/dim]"
            )

        # Call LLM
        t0 = time.time()
        try:
            response = call_ollama(managed_messages, tools=TOOL_SCHEMAS)
        except Exception as e:
            console.print(f"[red]LLM call failed: {e}[/red]")
            break

        llm_latency = round((time.time() - t0) * 1000, 1)
        resp_msg = response.get("message", {})
        usage = response.get("prompt_eval_count", 0)
        total_input_tokens += usage

        # Append assistant response to history
        messages.append({"role": "assistant", "content": resp_msg.get("content", "")})

        content_text = resp_msg.get("content", "") or ""
        console.print(f"  [green]Assistant:[/green] {content_text[:300]}")

        log_entry = {
            "turn": turn,
            "mode": mode,
            "llm_latency_ms": llm_latency,
            "input_tokens": usage,
            "firewall_stats": fw_stats,
            "response_preview": content_text[:200],
        }
        transcript.append(log_entry)

        # Check for completion
        if "TASK_COMPLETE" in content_text:
            console.print("[bold green]✓ Task complete![/bold green]")
            break

        # Handle tool calls
        tool_calls = resp_msg.get("tool_calls", [])
        if not tool_calls and not content_text.strip():
            console.print("[yellow]No response from LLM — stopping.[/yellow]")
            break

        for tc in tool_calls:
            fn_name = tc.get("function", {}).get("name", "")
            fn_args = tc.get("function", {}).get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}

            console.print(f"  [cyan]Tool call:[/cyan] {fn_name}({fn_args})")
            result = execute_tool_call(fn_name, fn_args)
            console.print(f"  [dim]Result ({len(result)} chars): {result[:150]}[/dim]")

            messages.append({
                "role": "tool",
                "content": result,
                "name": fn_name,
            })

    # Save transcript
    ts = int(time.time())
    transcript_path = log_path / f"transcript_{mode}_{ts}.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "mode": mode,
            "turns": turn,
            "total_input_tokens": total_input_tokens,
            "transcript": transcript,
        }, f, indent=2)

    console.print(f"\n[dim]Transcript saved: {transcript_path}[/dim]")

    return {
        "task": task,
        "mode": mode,
        "turns": turn,
        "total_input_tokens": total_input_tokens,
        "success": any("TASK_COMPLETE" in t.get("response_preview", "") for t in transcript),
        "transcript_path": str(transcript_path),
        "firewall_log": firewall.get_dashboard_data(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Rot Firewall — Agent Runner")
    parser.add_argument("--task", type=str, default="List all files in the current directory, then write a Python hello world script to hello.py and verify it runs.", help="Task for the agent")
    parser.add_argument("--mode", choices=["baseline", "firewall"], default="firewall", help="Run mode")
    parser.add_argument("--max-turns", type=int, default=20, help="Max agent turns")
    args = parser.parse_args()

    result = run_agent(task=args.task, mode=args.mode, max_turns=args.max_turns)
    console.print(f"\n[bold]Result:[/bold] turns={result['turns']}, success={result['success']}, tokens={result['total_input_tokens']}")
