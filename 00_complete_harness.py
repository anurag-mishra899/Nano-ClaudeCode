"""
Complete Unified Agent Harness
================================

A modular, production-ready agent harness that combines all features from the Nano-ClaudeCode
agent implementations (01-16). This harness provides:

CORE FEATURES (Always Active):
- Basic agent loop with tool execution (01)
- File operations: read, write, edit, bash with safe_path (02)
- Permission system with PermissionManager and BashSecurityValidator (07)
- Dynamic system prompt builder with multiple sources (10)

IMPORTANT FEATURES (Default Enabled):
- In-memory task planning with TaskManager (03)
- Context management with CompactState (06)
- Persistent memory across sessions with MemoryManager (09)
- Persistent task board with PersistentTaskManager (11)
- Background task execution with BackgroundManager (12)

OPTIONAL FEATURES (Can Enable/Disable):
- Skill registry for specialized instructions (05)
- Hook system for extensibility (08)
- Team collaboration features (13-16) - future extension points

Architecture:
- Modular managers that can be enabled/disabled via HarnessConfig
- Dynamic tool registration based on enabled features
- Unified system prompt assembled from multiple sources
- Integrated permission checking for all tool calls
- Comprehensive tracing for observability
"""

from utils import get_openai_client, normalize_messages
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import subprocess
import json
import re
import os
import time
import datetime
import uuid
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
from fnmatch import fnmatch

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HarnessConfig:
    """
    Configuration for the AgentHarness.

    Core settings (always active):
    - workdir: Working directory for the agent
    - permission_mode: Permission mode (default, plan, auto)

    Important features (default enabled):
    - enable_planning: In-memory task planning (03)
    - enable_compact: Context management and conversation compaction (06)
    - enable_memory: Persistent memory across sessions (09)
    - enable_persistent_tasks: Persistent task board on disk (11)
    - enable_background: Background task execution (12)

    Optional features (default disabled):
    - enable_skills: Skill registry system (05)
    - enable_hooks: Hook system for extensibility (08)
    - enable_teams: Team collaboration (13-16, future)
    """
    # Core (always active)
    workdir: Path = field(default_factory=Path.cwd)
    permission_mode: str = "default"  # default, plan, auto

    # Important features (default enabled)
    enable_planning: bool = True
    enable_compact: bool = True
    enable_memory: bool = True
    enable_persistent_tasks: bool = True
    enable_background: bool = True

    # Optional features (default disabled)
    enable_skills: bool = False
    enable_hooks: bool = False
    enable_teams: bool = False  # Future: 13-16


# ============================================================================
# Core: Safe Path and Basic Tools
# ============================================================================

def safe_path(workdir: Path, path_str: str) -> Path:
    """
    Validate that a path is within the workspace.
    Prevents directory traversal attacks.
    """
    path = (workdir / path_str).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


@traceable(run_type="tool", name="Bash Executor")
def run_bash(command: str, workdir: Path) -> str:
    """Execute a bash command with timeout and output limits."""
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


@traceable(run_type="tool", name="File Reader")
def run_read(path: str, workdir: Path, limit: Optional[int] = None) -> str:
    """Read file contents with optional line limit."""
    try:
        lines = safe_path(workdir, path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


@traceable(run_type="tool", name="File Writer")
def run_write(path: str, content: str, workdir: Path) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        fp = safe_path(workdir, path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@traceable(run_type="tool", name="File Editor")
def run_edit(path: str, old_text: str, new_text: str, workdir: Path) -> str:
    """Replace exact text in a file (first occurrence only)."""
    try:
        fp = safe_path(workdir, path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# Permission System (07)
# ============================================================================

PERMISSION_MODES = ("default", "plan", "auto")
READ_ONLY_TOOLS = {"read_file", "bash_readonly"}
WRITE_TOOLS = {"write_file", "edit_file", "bash"}


class BashSecurityValidator:
    """
    Validate bash commands for dangerous patterns.
    Returns list of failures or empty list if safe.
    """
    VALIDATORS = [
        ("shell_metachar", r"[;&|`$]"),
        ("sudo", r"\bsudo\b"),
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"),
        ("cmd_substitution", r"\$\("),
        ("ifs_injection", r"\bIFS\s*="),
    ]

    def validate(self, command: str) -> list:
        """Check command against all validators."""
        failures = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures

    def is_safe(self, command: str) -> bool:
        """Returns True only if no validators triggered."""
        return len(self.validate(command)) == 0

    def describe_failures(self, command: str) -> str:
        """Human-readable summary of validation failures."""
        failures = self.validate(command)
        if not failures:
            return "No issues detected"
        parts = [f"{name} (pattern: {pattern})" for name, pattern in failures]
        return "Security flags: " + ", ".join(parts)


DEFAULT_PERMISSION_RULES = [
    {"tool": "bash", "content": "rm -rf /", "behavior": "deny"},
    {"tool": "bash", "content": "sudo *", "behavior": "deny"},
    {"tool": "read_file", "path": "*", "behavior": "allow"},
]


class PermissionManager:
    """
    Manages permission decisions for tool calls.
    Pipeline: bash_validator -> deny_rules -> mode_check -> allow_rules -> ask_user
    """
    def __init__(self, mode: str = "default", rules: list = None):
        if mode not in PERMISSION_MODES:
            raise ValueError(f"Unknown mode: {mode}. Choose from {PERMISSION_MODES}")
        self.mode = mode
        self.rules = rules or list(DEFAULT_PERMISSION_RULES)
        self.validator = BashSecurityValidator()
        self.consecutive_denials = 0
        self.max_consecutive_denials = 3

    @traceable(run_type="tool", name="Permission Check")
    def check(self, tool_name: str, tool_input: dict) -> dict:
        """
        Returns: {"behavior": "allow"|"deny"|"ask", "reason": str}
        """
        # Step 0: Bash security validation
        if tool_name == "bash":
            command = tool_input.get("command", "")
            failures = self.validator.validate(command)
            if failures:
                severe = {"sudo", "rm_rf"}
                severe_hits = [f for f in failures if f[0] in severe]
                if severe_hits:
                    desc = self.validator.describe_failures(command)
                    return {"behavior": "deny", "reason": f"Bash validator: {desc}"}
                desc = self.validator.describe_failures(command)
                return {"behavior": "ask", "reason": f"Bash validator flagged: {desc}"}

        # Step 1: Deny rules
        for rule in self.rules:
            if rule["behavior"] != "deny":
                continue
            if self._matches(rule, tool_name, tool_input):
                return {"behavior": "deny", "reason": f"Blocked by deny rule: {rule}"}

        # Step 2: Mode-based decisions
        if self.mode == "plan":
            if tool_name in WRITE_TOOLS:
                return {"behavior": "deny", "reason": "Plan mode: write operations blocked"}
            return {"behavior": "allow", "reason": "Plan mode: read-only allowed"}

        if self.mode == "auto":
            if tool_name in READ_ONLY_TOOLS or tool_name == "read_file":
                return {"behavior": "allow", "reason": "Auto mode: read-only auto-approved"}

        # Step 3: Allow rules
        for rule in self.rules:
            if rule["behavior"] != "allow":
                continue
            if self._matches(rule, tool_name, tool_input):
                self.consecutive_denials = 0
                return {"behavior": "allow", "reason": f"Matched allow rule: {rule}"}

        # Step 4: Ask user
        return {"behavior": "ask", "reason": f"No rule matched for {tool_name}, asking user"}

    def ask_user(self, tool_name: str, tool_input: dict) -> bool:
        """Interactive approval prompt."""
        preview = json.dumps(tool_input, ensure_ascii=False)[:200]
        print(f"\n  [Permission] {tool_name}: {preview}")
        try:
            answer = input("  Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if answer == "always":
            self.rules.append({"tool": tool_name, "path": "*", "behavior": "allow"})
            self.consecutive_denials = 0
            return True
        if answer in ("y", "yes"):
            self.consecutive_denials = 0
            return True

        self.consecutive_denials += 1
        if self.consecutive_denials >= self.max_consecutive_denials:
            print(f"  [{self.consecutive_denials} consecutive denials - "
                  "consider switching to plan mode]")
        return False

    def _matches(self, rule: dict, tool_name: str, tool_input: dict) -> bool:
        """Check if a rule matches the tool call."""
        if rule.get("tool") and rule["tool"] != "*":
            if rule["tool"] != tool_name:
                return False
        if "path" in rule and rule["path"] != "*":
            path = tool_input.get("path", "")
            if not fnmatch(path, rule["path"]):
                return False
        if "content" in rule:
            command = tool_input.get("command", "")
            if not fnmatch(command, rule["content"]):
                return False
        return True


# ============================================================================
# Task Planning (03) - In-Memory
# ============================================================================

@dataclass
class Task:
    content: str
    status: str = 'pending'  # pending, in_progress, completed
    active_form: str = ''


@dataclass
class TaskList:
    tasks: list[Task] = field(default_factory=list)
    rounds_since_update: int = 0


class TaskManager:
    """In-memory task planning for multi-step work."""
    PLAN_REMINDER_INTERVAL = 3

    def __init__(self):
        self.state = TaskList()

    @traceable(run_type="tool", name="Update Plan")
    def update_plan(self, items: list) -> str:
        """Update the current session plan."""
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0

        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(Task(
                content=content,
                status=status,
                active_form=active_form,
            ))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.tasks = normalized
        self.state.rounds_since_update = 0
        return self.render_plan()

    def note_round_without_update(self) -> None:
        """Track rounds since last plan update."""
        self.state.rounds_since_update += 1

    def reminder(self) -> Optional[str]:
        """Return reminder if plan needs refresh."""
        if not self.state.tasks:
            return None
        if self.state.rounds_since_update < self.PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render_plan(self) -> str:
        """Render the current plan as formatted text."""
        if not self.state.tasks:
            return "No session plan yet."

        lines = []
        for item in self.state.tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]

            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)

        completed = sum(1 for item in self.state.tasks if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.tasks)} completed)")

        return "\n".join(lines)


# ============================================================================
# Compact State (06) - Context Management
# ============================================================================

@dataclass
class CompactState:
    """State tracking for conversation compaction."""
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000


def estimate_context_size(messages: list) -> int:
    """Rough estimate of context size."""
    return len(str(messages))


def track_recent_file(state: CompactState, path: str) -> None:
    """Track recently accessed files for summary."""
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]


@traceable(run_type="tool", name="Persist Large Output")
def persist_large_output(tool_use_id: str, output: str, workdir: Path) -> str:
    """Persist large tool outputs to disk, return preview."""
    if len(output) <= PERSIST_THRESHOLD:
        return output

    tool_results_dir = workdir / ".task_outputs" / "tool-results"
    tool_results_dir.mkdir(parents=True, exist_ok=True)
    stored_path = tool_results_dir / f"{tool_use_id}.txt"

    if not stored_path.exists():
        stored_path.write_text(output)

    preview = output[:PREVIEW_CHARS]
    rel_path = stored_path.relative_to(workdir)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )


@traceable(run_type="tool", name="Micro Compact")
def micro_compact(messages: list) -> list:
    """Compact older tool results to save context."""
    tool_results = [(i, m) for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages

    for _, message in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = message.get("content", "")
        if not isinstance(content, str) or len(content) <= 120:
            continue
        message["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"

    return messages


@traceable(run_type="llm", name="Summarize History")
def summarize_history(messages: list, client, model: str) -> str:
    """Use LLM to summarize conversation history."""
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve:\n"
        "1. The current goal\n"
        "2. Important findings and decisions\n"
        "3. Files read or changed\n"
        "4. Remaining work\n"
        "5. User constraints and preferences\n"
        "Be compact but concrete.\n\n"
        f"{conversation}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


@traceable(run_type="tool", name="Compact History")
def compact_history(
    messages: list,
    state: CompactState,
    workdir: Path,
    client,
    model: str,
    focus: Optional[str] = None
) -> list:
    """Compact conversation history and save transcript."""
    # Save transcript
    transcript_dir = workdir / ".transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"transcript_{int(time.time())}.jsonl"
    with transcript_path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    # Summarize
    summary = summarize_history(messages, client, model)
    if focus:
        summary += f"\n\nFocus to preserve next: {focus}"
    if state.recent_files:
        recent_lines = "\n".join(f"- {path}" for path in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed:\n{recent_lines}"

    state.has_compacted = True
    state.last_summary = summary

    return [{
        "role": "user",
        "content": (
            "This conversation was compacted so the agent can continue working.\n\n"
            f"{summary}"
        ),
    }]


# ============================================================================
# Memory Manager (09) - Persistent Memory
# ============================================================================

MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200


class MemoryManager:
    """Persistent memory across sessions."""

    def __init__(self, workdir: Path):
        self.memory_dir = workdir / ".memory"
        self.memory_index = self.memory_dir / "MEMORY.md"
        self.memories = {}  # name -> {description, type, content}

    @traceable(run_type="tool", name="Memory Loader")
    def load_all(self):
        """Load all memory files from disk."""
        self.memories = {}
        if not self.memory_dir.exists():
            return

        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            parsed = self._parse_frontmatter(md_file.read_text())
            if parsed:
                name = parsed.get("name", md_file.stem)
                self.memories[name] = {
                    "description": parsed.get("description", ""),
                    "type": parsed.get("type", "project"),
                    "content": parsed.get("content", ""),
                    "file": md_file.name,
                }

        count = len(self.memories)
        if count > 0:
            print(f"[Memory loaded: {count} memories from {self.memory_dir}]")

    def load_memory_prompt(self) -> str:
        """Build memory section for injection into system prompt."""
        if not self.memories:
            return ""

        sections = []
        sections.append("# Memories (persistent across sessions)")
        sections.append("")

        for mem_type in MEMORY_TYPES:
            typed = {k: v for k, v in self.memories.items() if v["type"] == mem_type}
            if not typed:
                continue
            sections.append(f"## [{mem_type}]")
            for name, mem in typed.items():
                sections.append(f"### {name}: {mem['description']}")
                if mem["content"].strip():
                    sections.append(mem["content"].strip())
                sections.append("")

        return "\n".join(sections)

    @traceable(run_type="tool", name="Save Memory")
    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """Save a memory to disk."""
        if mem_type not in MEMORY_TYPES:
            return f"Error: type must be one of {MEMORY_TYPES}"

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
        if not safe_name:
            return "Error: invalid memory name"

        self.memory_dir.mkdir(parents=True, exist_ok=True)

        frontmatter = (
            f"---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"type: {mem_type}\n"
            f"---\n"
            f"{content}\n"
        )

        file_name = f"{safe_name}.md"
        file_path = self.memory_dir / file_name
        file_path.write_text(frontmatter)

        self.memories[name] = {
            "description": description,
            "type": mem_type,
            "content": content,
            "file": file_name,
        }

        self._rebuild_index()
        return f"Saved memory '{name}' [{mem_type}] to {file_path}"

    def _rebuild_index(self):
        """Rebuild MEMORY.md index."""
        lines = ["# Memory Index", ""]
        for name, mem in self.memories.items():
            lines.append(f"- {name}: {mem['description']} [{mem['type']}]")
            if len(lines) >= MAX_INDEX_LINES:
                lines.append(f"... (truncated at {MAX_INDEX_LINES} lines)")
                break

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_index.write_text("\n".join(lines) + "\n")

    def _parse_frontmatter(self, text: str) -> Optional[dict]:
        """Parse YAML-like frontmatter."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
        if not match:
            return None
        header, body = match.group(1), match.group(2)
        result = {"content": body.strip()}
        for line in header.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result


# ============================================================================
# Persistent Task Manager (11)
# ============================================================================

class PersistentTaskManager:
    """Persistent task board on disk."""

    def __init__(self, workdir: Path):
        self.dir = workdir / ".tasks"
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """Find highest existing task ID."""
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        """Load task from disk."""
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """Save task to disk."""
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    @traceable(run_type="tool", name="Task Create")
    def create(self, subject: str, description: str = "") -> str:
        """Create a new task."""
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "blocks": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        """Get task details."""
        return json.dumps(self._load(task_id), indent=2)

    @traceable(run_type="tool", name="Task Update")
    def update(
        self,
        task_id: int,
        status: Optional[str] = None,
        owner: Optional[str] = None,
        add_blocked_by: Optional[list] = None,
        add_blocks: Optional[list] = None
    ) -> str:
        """Update task status and dependencies."""
        task = self._load(task_id)

        if owner is not None:
            task["owner"] = owner

        if status:
            if status not in ("pending", "in_progress", "completed", "deleted"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)

        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass

        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        """Remove completed task from blockedBy lists."""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    @traceable(run_type="tool", name="Task List")
    def list_all(self) -> str:
        """List all tasks."""
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))

        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
                "deleted": "[-]"
            }.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{blocked}")

        return "\n".join(lines)


# ============================================================================
# Background Manager (12)
# ============================================================================

class BackgroundManager:
    """Background task execution with notifications."""

    def __init__(self, workdir: Path):
        self.dir = workdir / ".runtime-tasks"
        self.dir.mkdir(exist_ok=True)
        self.tasks = {}  # task_id -> {status, result, command, started_at}
        self._notification_queue = []
        self._lock = threading.Lock()

    def _record_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.json"

    def _output_path(self, task_id: str) -> Path:
        return self.dir / f"{task_id}.log"

    def _persist_task(self, task_id: str, workdir: Path):
        record = dict(self.tasks[task_id])
        self._record_path(task_id).write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )

    def _preview(self, output: str, limit: int = 500) -> str:
        compact = " ".join((output or "(no output)").split())
        return compact[:limit]

    @traceable(run_type="tool", name="Background Run")
    def run(self, command: str, workdir: Path) -> str:
        """Start a background thread, return task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        output_file = self._output_path(task_id)

        self.tasks[task_id] = {
            "id": task_id,
            "status": "running",
            "result": None,
            "command": command,
            "started_at": time.time(),
            "finished_at": None,
            "result_preview": "",
            "output_file": str(output_file.relative_to(workdir)),
        }
        self._persist_task(task_id, workdir)

        thread = threading.Thread(
            target=self._execute,
            args=(task_id, command, workdir),
            daemon=True
        )
        thread.start()

        return (
            f"Background task {task_id} started: {command[:80]} "
            f"(output_file={output_file.relative_to(workdir)})"
        )

    @traceable(run_type="tool", name="Background Executor")
    def _execute(self, task_id: str, command: str, workdir: Path):
        """Thread target: run subprocess, capture output."""
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        final_output = output or "(no output)"
        preview = self._preview(final_output)
        output_path = self._output_path(task_id)
        output_path.write_text(final_output)

        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = final_output
        self.tasks[task_id]["finished_at"] = time.time()
        self.tasks[task_id]["result_preview"] = preview
        self._persist_task(task_id, workdir)

        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "preview": preview,
                "output_file": str(output_path.relative_to(workdir)),
            })

    @traceable(run_type="tool", name="Background Check")
    def check(self, task_id: Optional[str] = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            visible = {
                "id": t["id"],
                "status": t["status"],
                "command": t["command"],
                "result_preview": t.get("result_preview", ""),
                "output_file": t.get("output_file", ""),
            }
            return json.dumps(visible, indent=2, ensure_ascii=False)

        lines = []
        for tid, t in self.tasks.items():
            lines.append(
                f"{tid}: [{t['status']}] {t['command'][:60]} "
                f"-> {t.get('result_preview') or '(running)'}"
            )
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


# ============================================================================
# Skill Registry (05) - Optional
# ============================================================================

@dataclass
class Skill:
    name: str
    description: str
    path: str


@dataclass
class SkillDocument:
    skill: Skill
    body: str


class SkillRegistry:
    """Registry of specialized instruction skills."""

    def __init__(self, workdir: Path):
        self.skills_dir = workdir / "skills"
        self.documents: dict[str, SkillDocument] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all skills from the skills directory."""
        if not self.skills_dir.exists():
            return

        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta, body = self._parse_frontmatter(path.read_text())
            name = meta.get("name", path.parent.name)
            description = meta.get("description", "No description")
            skill = Skill(name=name, description=description, path=str(path))
            self.documents[name] = SkillDocument(skill=skill, body=body.strip())

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """Parse frontmatter from skill file."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        return meta, match.group(2)

    def describe_available(self) -> str:
        """Return description of available skills."""
        if not self.documents:
            return "(no skills available)"
        lines = []
        for name in sorted(self.documents):
            skill = self.documents[name].skill
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def load_full_text(self, name: str) -> str:
        """Load full skill text."""
        document = self.documents.get(name)
        if not document:
            known = ", ".join(sorted(self.documents)) or "(none)"
            return f"Error: Unknown skill '{name}'. Available skills: {known}"
        return (
            f"<skill name=\"{document.skill.name}\">\n"
            f"{document.body}\n"
            "</skill>"
        )


# ============================================================================
# Hook Manager (08) - Optional
# ============================================================================

HOOK_EVENTS = ("PreToolUse", "PostToolUse", "SessionStart")
HOOK_TIMEOUT = 30


class HookManager:
    """Execute hooks from .hooks.json configuration."""

    def __init__(self, workdir: Path, sdk_mode: bool = False):
        self.workdir = workdir
        self.hooks = {"PreToolUse": [], "PostToolUse": [], "SessionStart": []}
        self._sdk_mode = sdk_mode

        config_path = workdir / ".hooks.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                for event in HOOK_EVENTS:
                    self.hooks[event] = config.get("hooks", {}).get(event, [])
                print(f"[Hooks loaded from {config_path}]")
            except Exception as e:
                print(f"[Hook config error: {e}]")

    def _check_workspace_trust(self) -> bool:
        """Check if workspace is trusted."""
        if self._sdk_mode:
            return True
        trust_marker = self.workdir / ".claude" / ".claude_trusted"
        return trust_marker.exists()

    @traceable(run_type="tool", name="Hook Executor")
    def run_hooks(self, event: str, context: Optional[dict] = None) -> dict:
        """
        Execute all hooks for an event.
        Returns: {"blocked": bool, "messages": list[str]}
        """
        result = {"blocked": False, "messages": []}

        if not self._check_workspace_trust():
            if self.hooks.get(event) and not self._sdk_mode:
                trust_marker = self.workdir / ".claude" / ".claude_trusted"
                print(f"  [Hooks] Workspace not trusted - hooks disabled. "
                      f"Create {trust_marker} to enable.")
            return result

        hooks = self.hooks.get(event, [])
        for hook_def in hooks:
            matcher = hook_def.get("matcher")
            if matcher and context:
                tool_name = context.get("tool_name", "")
                if matcher != "*" and matcher != tool_name:
                    continue

            command = hook_def.get("command", "")
            if not command:
                continue

            env = dict(os.environ)
            if context:
                env["HOOK_EVENT"] = event
                env["HOOK_TOOL_NAME"] = context.get("tool_name", "")
                env["HOOK_TOOL_INPUT"] = json.dumps(
                    context.get("tool_input", {}), ensure_ascii=False)[:10000]
                if "tool_output" in context:
                    env["HOOK_TOOL_OUTPUT"] = str(context["tool_output"])[:10000]

            try:
                r = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.workdir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=HOOK_TIMEOUT,
                )

                if r.returncode == 0:
                    output = r.stdout.strip() or r.stderr.strip()
                    if output:
                        print(f"  [hook:{event}] {output[:100]}")
                elif r.returncode == 1:
                    result["blocked"] = True
                    reason = r.stderr.strip() or "Blocked by hook"
                    result["block_reason"] = reason
                    print(f"  [hook:{event}] BLOCKED: {reason[:200]}")
                elif r.returncode == 2:
                    msg = r.stderr.strip()
                    if msg:
                        result["messages"].append(msg)
                        print(f"  [hook:{event}] INJECT: {msg[:200]}")

            except subprocess.TimeoutExpired:
                print(f"  [hook:{event}] Timeout ({HOOK_TIMEOUT}s)")
            except Exception as e:
                print(f"  [hook:{event}] Error: {e}")

        return result


# ============================================================================
# System Prompt Builder (10)
# ============================================================================

class SystemPromptBuilder:
    """Assemble system prompt from multiple sources."""

    def __init__(self, workdir: Path, tools: list, model: str):
        self.workdir = workdir
        self.tools = tools
        self.model = model
        self.skills_dir = workdir / "skills"
        self.memory_dir = workdir / ".memory"
        self.DYNAMIC_BOUNDARY = "=== DYNAMIC_BOUNDARY ==="
        self._cached_prompt = None

    def _build_core(self) -> str:
        """Core instructions."""
        return (
            f"You are a coding agent operating in {self.workdir}.\n"
            "Use the provided tools to explore, read, write, and edit files.\n"
            "Always verify before assuming. Prefer reading files over guessing."
        )

    def _build_tool_listing(self) -> str:
        """List available tools."""
        if not self.tools:
            return ""
        lines = ["# Available tools"]
        for tool in self.tools:
            func = tool.get("function", {})
            props = func.get("parameters", {}).get("properties", {})
            params = ", ".join(props.keys())
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            lines.append(f"- {name}({params}): {desc}")
        return "\n".join(lines)

    def _build_skill_listing(self) -> str:
        """List available skills."""
        if not self.skills_dir.exists():
            return ""
        skills = []
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if not match:
                continue
            meta = {}
            for line in match.group(1).splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", skill_dir.name)
            desc = meta.get("description", "")
            skills.append(f"- {name}: {desc}")
        if not skills:
            return ""
        return "# Available skills\n" + "\n".join(skills)

    def _build_memory_section(self, memory_mgr: Optional[MemoryManager] = None) -> str:
        """Build memory section."""
        if memory_mgr:
            return memory_mgr.load_memory_prompt()
        return ""

    def _build_claude_md(self) -> str:
        """Load CLAUDE.md files in priority order."""
        sources = []

        # User-global
        user_claude = Path.home() / ".claude" / "CLAUDE.md"
        if user_claude.exists():
            sources.append(("user global (~/.claude/CLAUDE.md)", user_claude.read_text()))

        # Project root
        project_claude = self.workdir / "CLAUDE.md"
        if project_claude.exists():
            sources.append(("project root (CLAUDE.md)", project_claude.read_text()))

        # Subdirectory
        cwd = Path.cwd()
        if cwd != self.workdir:
            subdir_claude = cwd / "CLAUDE.md"
            if subdir_claude.exists():
                sources.append((f"subdir ({cwd.name}/CLAUDE.md)", subdir_claude.read_text()))

        if not sources:
            return ""

        parts = ["# CLAUDE.md instructions"]
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "\n\n".join(parts)

    def _build_dynamic_context(self) -> str:
        """Build dynamic context section."""
        lines = [
            f"Current date: {datetime.date.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {self.model}",
            f"Platform: {os.uname().sysname}",
        ]
        return "# Dynamic context\n" + "\n".join(lines)

    @traceable(run_type="tool", name="Build System Prompt")
    def build(
        self,
        force_rebuild: bool = False,
        memory_mgr: Optional[MemoryManager] = None
    ) -> str:
        """Assemble full system prompt."""
        if not force_rebuild and self._cached_prompt:
            return self._cached_prompt

        sections = []

        core = self._build_core()
        if core:
            sections.append(core)

        tools = self._build_tool_listing()
        if tools:
            sections.append(tools)

        skills = self._build_skill_listing()
        if skills:
            sections.append(skills)

        memory = self._build_memory_section(memory_mgr)
        if memory:
            sections.append(memory)

        claude_md = self._build_claude_md()
        if claude_md:
            sections.append(claude_md)

        sections.append(self.DYNAMIC_BOUNDARY)

        dynamic = self._build_dynamic_context()
        if dynamic:
            sections.append(dynamic)

        self._cached_prompt = "\n\n".join(sections)
        return self._cached_prompt

    def invalidate_cache(self):
        """Force next build() to rebuild the prompt."""
        self._cached_prompt = None


# ============================================================================
# Main Agent Harness
# ============================================================================

class AgentHarness:
    """
    Unified agent harness with all features integrated.

    This is the main class that orchestrates:
    - Core tools (bash, read, write, edit)
    - Permission checking
    - Optional features (planning, compact, memory, tasks, background, skills, hooks)
    - Dynamic system prompt building
    - Agent loop with integrated feature support
    """

    def __init__(self, config: Optional[HarnessConfig] = None):
        """Initialize the agent harness with given configuration."""
        self.config = config or HarnessConfig()
        self.workdir = self.config.workdir

        # Initialize OpenAI client
        self.client, self.model = get_openai_client()
        self.client = wrap_openai(self.client)

        # Core: Permission manager (always active)
        self.permissions = PermissionManager(mode=self.config.permission_mode)

        # Important features (conditionally initialized)
        self.task_manager = TaskManager() if self.config.enable_planning else None
        self.compact_state = CompactState() if self.config.enable_compact else None
        self.memory_mgr = None
        if self.config.enable_memory:
            self.memory_mgr = MemoryManager(self.workdir)
            self.memory_mgr.load_all()

        self.persistent_tasks = None
        if self.config.enable_persistent_tasks:
            self.persistent_tasks = PersistentTaskManager(self.workdir)

        self.background_mgr = None
        if self.config.enable_background:
            self.background_mgr = BackgroundManager(self.workdir)

        # Optional features
        self.skill_registry = None
        if self.config.enable_skills:
            self.skill_registry = SkillRegistry(self.workdir)

        self.hook_manager = None
        if self.config.enable_hooks:
            self.hook_manager = HookManager(self.workdir)
            if self.hook_manager:
                self.hook_manager.run_hooks("SessionStart", {
                    "tool_name": "",
                    "tool_input": {}
                })

        # Build tools list
        self.tools = self._build_tools()

        # System prompt builder
        self.prompt_builder = SystemPromptBuilder(
            workdir=self.workdir,
            tools=self.tools,
            model=self.model
        )

    def _build_tools(self) -> list:
        """Build tools list based on enabled features."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["path"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace exact text in file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"}
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                }
            },
        ]

        # Add planning tool
        if self.config.enable_planning:
            tools.append({
                "type": "function",
                "function": {
                    "name": "todo",
                    "description": "Rewrite the current session plan for multi-step work.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {
                                            "type": "string",
                                            "enum": ["pending", "in_progress", "completed"],
                                        },
                                        "activeForm": {"type": "string"},
                                    },
                                    "required": ["content", "status"],
                                },
                            },
                        },
                        "required": ["items"],
                    },
                }
            })

        # Add compact tool
        if self.config.enable_compact:
            tools.append({
                "type": "function",
                "function": {
                    "name": "compact",
                    "description": "Summarize earlier conversation so work can continue in smaller context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "focus": {"type": "string"},
                        },
                    },
                }
            })

        # Add memory tool
        if self.config.enable_memory:
            tools.append({
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save a memory that should persist across sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["user", "feedback", "project", "reference"]
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["name", "description", "type", "content"]
                    }
                },
            })

        # Add persistent task tools
        if self.config.enable_persistent_tasks:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "task_create",
                        "description": "Create a new task.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["subject"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "task_update",
                        "description": "Update a task's status, owner, or dependencies.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "integer"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "deleted"]
                                },
                                "owner": {"type": "string"},
                                "addBlockedBy": {"type": "array", "items": {"type": "integer"}},
                                "addBlocks": {"type": "array", "items": {"type": "integer"}}
                            },
                            "required": ["task_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "task_list",
                        "description": "List all tasks with status summary.",
                        "parameters": {"type": "object", "properties": {}}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "task_get",
                        "description": "Get full details of a task by ID.",
                        "parameters": {
                            "type": "object",
                            "properties": {"task_id": {"type": "integer"}},
                            "required": ["task_id"]
                        }
                    }
                },
            ])

        # Add background tools
        if self.config.enable_background:
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "background_run",
                        "description": "Run command in background. Returns task_id immediately.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"}
                            },
                            "required": ["command"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "check_background",
                        "description": "Check background task status.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"}
                            }
                        }
                    }
                },
            ])

        # Add skill tool
        if self.config.enable_skills:
            tools.append({
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "Load the full body of a named skill into context.",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                }
            })

        return tools

    @traceable(run_type="tool", name="Execute Tool")
    def _execute_tool(self, tool_name: str, tool_args: dict, tool_use_id: str) -> str:
        """Execute a tool with the given arguments."""
        # Core tools
        if tool_name == "bash":
            return run_bash(tool_args["command"], self.workdir)

        elif tool_name == "read_file":
            output = run_read(tool_args["path"], self.workdir, tool_args.get("limit"))
            if self.config.enable_compact:
                output = persist_large_output(tool_use_id, output, self.workdir)
                track_recent_file(self.compact_state, tool_args["path"])
            return output

        elif tool_name == "write_file":
            return run_write(tool_args["path"], tool_args["content"], self.workdir)

        elif tool_name == "edit_file":
            return run_edit(
                tool_args["path"],
                tool_args["old_text"],
                tool_args["new_text"],
                self.workdir
            )

        # Planning tool
        elif tool_name == "todo" and self.task_manager:
            return self.task_manager.update_plan(tool_args["items"])

        # Compact tool
        elif tool_name == "compact":
            return "Compacting conversation..."

        # Memory tool
        elif tool_name == "save_memory" and self.memory_mgr:
            return self.memory_mgr.save_memory(
                tool_args["name"],
                tool_args["description"],
                tool_args["type"],
                tool_args["content"]
            )

        # Persistent task tools
        elif tool_name == "task_create" and self.persistent_tasks:
            return self.persistent_tasks.create(
                tool_args["subject"],
                tool_args.get("description", "")
            )

        elif tool_name == "task_update" and self.persistent_tasks:
            return self.persistent_tasks.update(
                tool_args["task_id"],
                tool_args.get("status"),
                tool_args.get("owner"),
                tool_args.get("addBlockedBy"),
                tool_args.get("addBlocks")
            )

        elif tool_name == "task_list" and self.persistent_tasks:
            return self.persistent_tasks.list_all()

        elif tool_name == "task_get" and self.persistent_tasks:
            return self.persistent_tasks.get(tool_args["task_id"])

        # Background tools
        elif tool_name == "background_run" and self.background_mgr:
            return self.background_mgr.run(tool_args["command"], self.workdir)

        elif tool_name == "check_background" and self.background_mgr:
            return self.background_mgr.check(tool_args.get("task_id"))

        # Skill tool
        elif tool_name == "load_skill" and self.skill_registry:
            return self.skill_registry.load_full_text(tool_args["name"])

        else:
            return f"Unknown tool: {tool_name}"

    @traceable(name="Agent Loop")
    def agent_loop(self, messages: list) -> str:
        """
        Main agent loop with all features integrated.

        Flow:
        1. Inject background notifications (if enabled)
        2. Apply micro-compact (if enabled)
        3. Check for auto-compact (if enabled)
        4. Call LLM
        5. Execute tools with permission checking and hooks
        6. Handle manual compact (if triggered)
        7. Update planning state (if enabled)
        """
        # Prepend system prompt if not present
        if not messages or messages[0].get("role") != "system":
            system_prompt = self.prompt_builder.build(memory_mgr=self.memory_mgr)
            messages.insert(0, {"role": "system", "content": system_prompt})

        while True:
            # Inject background notifications
            if self.config.enable_background and self.background_mgr:
                notifs = self.background_mgr.drain_notifications()
                if notifs and messages:
                    notif_text = "\n".join(
                        f"[bg:{n['task_id']}] {n['status']}: {n['preview']} "
                        f"(output_file={n['output_file']})"
                        for n in notifs
                    )
                    messages.append({
                        "role": "user",
                        "content": f"<background-results>\n{notif_text}\n</background-results>"
                    })

            # Micro-compact (if enabled)
            if self.config.enable_compact:
                messages[:] = micro_compact(messages)

                # Check for auto-compact
                if estimate_context_size(messages) > CONTEXT_LIMIT:
                    print("[auto compact]")
                    messages[:] = compact_history(
                        messages,
                        self.compact_state,
                        self.workdir,
                        self.client,
                        self.model
                    )

            # Normalize and call LLM
            clean_messages = normalize_messages(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=clean_messages,
                tools=self.tools,
                max_tokens=8000,
            )

            assistant_msg = response.choices[0].message

            # Append assistant message
            msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_msg.tool_calls
                ]
            messages.append(msg_dict)

            # Check if done
            if not assistant_msg.tool_calls:
                return assistant_msg.content

            # Execute tool calls
            used_todo = False
            manual_compact = False
            compact_focus = None

            for tc in assistant_msg.tool_calls:
                args = json.loads(tc.function.arguments)
                name = tc.function.name

                # PreToolUse hooks
                if self.config.enable_hooks and self.hook_manager:
                    ctx = {"tool_name": name, "tool_input": args}
                    pre_result = self.hook_manager.run_hooks("PreToolUse", ctx)

                    for msg in pre_result.get("messages", []):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"[Hook message]: {msg}",
                        })

                    if pre_result.get("blocked"):
                        reason = pre_result.get("block_reason", "Blocked by hook")
                        output = f"Tool blocked by PreToolUse hook: {reason}"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output,
                        })
                        continue

                # Permission check
                decision = self.permissions.check(name, args)

                if decision["behavior"] == "deny":
                    output = f"Permission denied: {decision['reason']}"
                    print(f"  [DENIED] {name}: {decision['reason']}")

                elif decision["behavior"] == "ask":
                    if self.permissions.ask_user(name, args):
                        output = self._execute_tool(name, args, tc.id)
                        print(f"> {name}: {str(output)[:200]}")
                    else:
                        output = f"Permission denied by user for {name}"
                        print(f"  [USER DENIED] {name}")

                else:  # allow
                    output = self._execute_tool(name, args, tc.id)
                    print(f"> {name}: {str(output)[:200]}")

                # Track special tools
                if name == "todo":
                    used_todo = True
                if name == "compact":
                    manual_compact = True
                    compact_focus = args.get("focus")

                # PostToolUse hooks
                if self.config.enable_hooks and self.hook_manager:
                    ctx = {
                        "tool_name": name,
                        "tool_input": args,
                        "tool_output": output
                    }
                    post_result = self.hook_manager.run_hooks("PostToolUse", ctx)
                    for msg in post_result.get("messages", []):
                        output += f"\n[Hook note]: {msg}"

                # Append tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output
                })

            # Handle manual compact
            if manual_compact and self.config.enable_compact:
                print("[manual compact]")
                messages[:] = compact_history(
                    messages,
                    self.compact_state,
                    self.workdir,
                    self.client,
                    self.model,
                    focus=compact_focus
                )

            # Update planning state
            if self.config.enable_planning and self.task_manager:
                if used_todo:
                    self.task_manager.state.rounds_since_update = 0
                else:
                    self.task_manager.note_round_without_update()
                    reminder = self.task_manager.reminder()
                    if reminder:
                        messages.append({"role": "user", "content": reminder})

    def run(self, query: str) -> str:
        """
        Entry point for running a single query.
        Creates a fresh message history for each query.
        """
        messages = [{"role": "user", "content": query}]
        return self.agent_loop(messages)

    def run_repl(self):
        """Run the agent in REPL mode."""
        print(f"[Agent Harness initialized at {self.workdir}]")
        print(f"[Permission mode: {self.config.permission_mode}]")
        print(f"[Features enabled: ", end="")
        features = []
        if self.config.enable_planning:
            features.append("planning")
        if self.config.enable_compact:
            features.append("compact")
        if self.config.enable_memory:
            features.append("memory")
        if self.config.enable_persistent_tasks:
            features.append("tasks")
        if self.config.enable_background:
            features.append("background")
        if self.config.enable_skills:
            features.append("skills")
        if self.config.enable_hooks:
            features.append("hooks")
        print(", ".join(features) + "]")
        print()

        history = []

        while True:
            try:
                query = input("Query >> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if query.strip().lower() in ("q", "exit", "quit", ""):
                print("Goodbye!")
                break

            # Handle special commands
            if query.strip() == "/mode":
                print(f"Current permission mode: {self.permissions.mode}")
                print(f"Available modes: {', '.join(PERMISSION_MODES)}")
                continue

            if query.strip().startswith("/mode "):
                new_mode = query.strip().split()[1]
                if new_mode in PERMISSION_MODES:
                    self.permissions.mode = new_mode
                    print(f"[Switched to {new_mode} mode]")
                else:
                    print(f"Invalid mode. Available: {', '.join(PERMISSION_MODES)}")
                continue

            if query.strip() == "/rules":
                for i, rule in enumerate(self.permissions.rules):
                    print(f"  {i}: {rule}")
                continue

            if query.strip() == "/prompt":
                prompt = self.prompt_builder.build(memory_mgr=self.memory_mgr)
                print(f"-- System Prompt ({len(prompt)} chars) --")
                print(prompt)
                print("-- END --")
                continue

            if query.strip() == "/reload":
                self.prompt_builder.invalidate_cache()
                if self.memory_mgr:
                    self.memory_mgr.load_all()
                print("[System prompt and memory reloaded]")
                continue

            # Normal query
            history.append({"role": "user", "content": query})
            final_response = self.agent_loop(history)
            print(final_response)
            print()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Create configuration with default settings (all important features enabled)
    config = HarnessConfig(
        workdir=Path.cwd(),
        permission_mode="default",
        enable_planning=True,
        enable_compact=True,
        enable_memory=True,
        enable_persistent_tasks=True,
        enable_background=True,
        enable_skills=False,  # Optional, enable if you have skills/ directory
        enable_hooks=False,   # Optional, enable if you have .hooks.json
    )

    # Create and run harness
    harness = AgentHarness(config)
    harness.run_repl()
