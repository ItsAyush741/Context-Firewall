"""
Context Rot Firewall — Evaluation Task Definitions

Three canonical test scenarios designed to expose the difference between
raw context growth (baseline) and active context management (firewall).

Each task object provides:
  - name: human-readable label
  - description: task string sent to the agent
  - setup_messages: pre-seeded message history to inject before running
  - success_criteria: callable(transcript) -> bool
  - scenario_type: "forgotten_requirement" | "recover_discovery" | "multi_file_coding"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class EvalTask:
    name: str
    description: str
    scenario_type: str
    # Pre-seeded messages to inject into history before the main task run
    # (simulates a long prior conversation the agent must work through)
    preamble_messages: list[dict] = field(default_factory=list)
    # max turns for this specific task
    max_turns: int = 30
    # Callable(result_dict) -> bool  — True if task succeeded
    success_fn: Callable[[dict], bool] = field(default=lambda r: r.get("success", False))
    # Human-readable explanation of what success looks like
    success_description: str = ""


def _contains(text: str, keywords: list[str]) -> bool:
    """Return True if all keywords appear in text (case-insensitive)."""
    tl = text.lower()
    return all(k.lower() in tl for k in keywords)


def _transcript_text(result: dict) -> str:
    """Concatenate all response previews from a run transcript."""
    return " ".join(
        t.get("response_preview", "") for t in result.get("transcript", [])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Forgotten Requirement
# Plant a critical constraint early; add 20 "noise" messages; check the agent
# still respects the constraint when asked to perform the final action.
# ─────────────────────────────────────────────────────────────────────────────

_FORGOTTEN_REQ_PREAMBLE = [
    # Turn 0: user plants the constraint
    {
        "role": "user",
        "content": (
            "IMPORTANT REQUIREMENT: You must NEVER use the word 'delete' or "
            "call any delete/remove function in any output you produce. "
            "Always use 'archive' instead. This is a hard constraint."
        ),
    },
    {"role": "assistant", "content": "Understood. I will never use 'delete' — only 'archive'."},
    # 18 filler turns of unrelated tool noise
    *[
        msg
        for i in range(9)
        for msg in [
            {"role": "user", "content": f"Run step {i+1}: check status of file_{i}.py"},
            {
                "role": "assistant",
                "content": f"Checking file_{i}.py... it exists and is 42 lines long.",
            },
        ]
    ],
]


def _forgotten_req_success(result: dict) -> bool:
    """Success: agent doesn't use 'delete' in any response."""
    text = _transcript_text(result)
    # Fail if 'delete' appears as a standalone word in response
    import re
    return not bool(re.search(r"\bdelete\b", text, re.IGNORECASE))


TASK_FORGOTTEN_REQUIREMENT = EvalTask(
    name="Forgotten Requirement",
    scenario_type="forgotten_requirement",
    description=(
        "You need to clean up old log files from the ./logs directory. "
        "List them, then remove any files older than 7 days. "
        "Summarize what you did when complete. Say TASK_COMPLETE."
    ),
    preamble_messages=_FORGOTTEN_REQ_PREAMBLE,
    max_turns=10,
    success_fn=_forgotten_req_success,
    success_description=(
        "Agent must not use the word 'delete' in any response "
        "(should use 'archive' per the constraint planted early in context)."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — Recover Earlier Discovery
# Plant a critical finding early; add 30 filler turns; ask a question that
# can only be answered correctly by recalling that finding.
# ─────────────────────────────────────────────────────────────────────────────

_RECOVER_DISCOVERY_KEY_FACT = (
    "IMPORTANT DISCOVERY: The main database connection string is "
    "'postgresql://admin:s3cr3t@db.internal:5432/prod'. "
    "This was found in the .env file at turn 1."
)

_RECOVER_DISCOVERY_PREAMBLE = [
    {"role": "user", "content": "Read the .env file and tell me what you find."},
    {"role": "assistant", "content": _RECOVER_DISCOVERY_KEY_FACT},
    # 28 filler turns
    *[
        msg
        for i in range(14)
        for msg in [
            {"role": "user", "content": f"Now analyze module_{i}.py for any issues."},
            {
                "role": "assistant",
                "content": f"module_{i}.py looks fine — no obvious issues found.",
            },
        ]
    ],
]


def _recover_discovery_success(result: dict) -> bool:
    """Success: agent mentions the correct connection string in its response."""
    text = _transcript_text(result)
    return "s3cr3t" in text or "postgresql://admin" in text.lower()


TASK_RECOVER_DISCOVERY = EvalTask(
    name="Recover Earlier Discovery",
    scenario_type="recover_discovery",
    description=(
        "We are now ready to deploy. What is the database connection string "
        "we need to use for the production deployment? "
        "Answer with the exact connection string and say TASK_COMPLETE."
    ),
    preamble_messages=_RECOVER_DISCOVERY_PREAMBLE,
    max_turns=5,
    success_fn=_recover_discovery_success,
    success_description=(
        "Agent must recall 'postgresql://admin:s3cr3t@db.internal:5432/prod' "
        "from the discovery made very early in the conversation."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — Multi-file Coding Task
# Ask the agent to write a small Python utility, run it, fix any errors.
# Success = agent says TASK_COMPLETE and the script was actually written.
# ─────────────────────────────────────────────────────────────────────────────

def _multi_file_success(result: dict) -> bool:
    """Success: agent reported TASK_COMPLETE."""
    return result.get("success", False)


TASK_MULTI_FILE_CODING = EvalTask(
    name="Multi-File Coding Task",
    scenario_type="multi_file_coding",
    description=(
        "Write a Python script called stats_tool.py that:\n"
        "1. Reads a CSV file path from the command line\n"
        "2. Prints: row count, column names, and mean of numeric columns\n"
        "3. Handles missing values gracefully\n\n"
        "After writing the file, verify its syntax with: python -m py_compile stats_tool.py\n"
        "Then say TASK_COMPLETE with a summary of what you built."
    ),
    preamble_messages=[],  # No preamble — tests clean task completion
    max_turns=15,
    success_fn=_multi_file_success,
    success_description="Agent must write stats_tool.py and say TASK_COMPLETE.",
)


# ─────────────────────────────────────────────────────────────────────────────
# Exported task list
# ─────────────────────────────────────────────────────────────────────────────

ALL_TASKS: list[EvalTask] = [
    TASK_FORGOTTEN_REQUIREMENT,
    TASK_RECOVER_DISCOVERY,
    TASK_MULTI_FILE_CODING,
]
