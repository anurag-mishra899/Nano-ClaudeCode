from utils import get_openai_client, get_prompt, normalize_messages
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import subprocess
import json
import datetime
import re
import os
import time
from pathlib import Path
from dataclasses import dataclass, field

client, AZURE_GPT41_MODEL = get_openai_client()
client = wrap_openai(client)  # Enable LangSmith tracing

WORKDIR = Path.cwd()

class SystemPromptBuilder:
    """
    Assemble the system prompt from independent sections.
    The teaching goal here is clarity:
    each section has one source and one responsibility.
    That makes the prompt easier to reason about, easier to test, and easier
    to evolve as the agent grows new capabilities.
    """
    def __init__(self, workdir: Path = None, tools: list = None):
        self.workdir = workdir or WORKDIR
        self.tools = tools or []
        self.skills_dir = self.workdir / "skills"
        self.memory_dir = self.workdir / ".memory"
        self.DYNAMIC_BOUNDARY = "=== DYNAMIC_BOUNDARY ==="
        self._cached_prompt = None  # Cache the built prompt

    # -- Section 1: Core instructions --
    def _build_core(self) -> str:
        return (
            f"You are a coding agent operating in {self.workdir}.\n"
            "Use the provided tools to explore, read, write, and edit files.\n"
            "Always verify before assuming. Prefer reading files over guessing."
        )
    
    # -- Section 2: Tool listings --
    def _build_tool_listing(self) -> str:
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
    
    # -- Section 3: Skill metadata (layer 1 from s05 concept) --
    def _build_skill_listing(self) -> str:
        if not self.skills_dir.exists():
            return ""
        skills = []
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            text = skill_md.read_text()
            # Parse frontmatter for name + description
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
    
    # -- Section 4: Memory content --
    def _build_memory_section(self) -> str:
        if not self.memory_dir.exists():
            return ""
        memories = []
        for md_file in sorted(self.memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue
            text = md_file.read_text()
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
            if not match:
                continue
            header, body = match.group(1), match.group(2).strip()
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
            name = meta.get("name", md_file.stem)
            mem_type = meta.get("type", "project")
            desc = meta.get("description", "")
            memories.append(f"[{mem_type}] {name}: {desc}\n{body}")
        if not memories:
            return ""
        return "# Memories (persistent)\n\n" + "\n\n".join(memories)
    
    # -- Section 5: CLAUDE.md chain --
    def _build_claude_md(self) -> str:
        """
        Load CLAUDE.md files in priority order (all are included):
        1. ~/.claude/CLAUDE.md (user-global instructions)
        2. <project-root>/CLAUDE.md (project instructions)
        3. <current-subdir>/CLAUDE.md (directory-specific instructions)
        """
        sources = []
        # User-global
        user_claude = Path.home() / ".claude" / "CLAUDE.md"
        if user_claude.exists():
            sources.append(("user global (~/.claude/CLAUDE.md)", user_claude.read_text()))
        # Project root
        project_claude = self.workdir / "CLAUDE.md"
        if project_claude.exists():
            sources.append(("project root (CLAUDE.md)", project_claude.read_text()))
        # Subdirectory -- in real CC, this walks from cwd up to project root
        # Teaching: check cwd if different from workdir
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
    
    # -- Section 6: Dynamic context --
    def _build_dynamic_context(self) -> str:
        lines = [
            f"Current date: {datetime.date.today().isoformat()}",
            f"Working directory: {self.workdir}",
            f"Model: {AZURE_GPT41_MODEL}",
            f"Platform: {os.uname().sysname}",
        ]
        return "# Dynamic context\n" + "\n".join(lines)
    
    # -- Assemble all sections --
    @traceable(run_type="tool", name="Build System Prompt")
    def build(self, force_rebuild: bool = False) -> str:
        """
        Assemble the full system prompt from all sections.
        Static sections (1-5) are separated from dynamic (6) by
        the DYNAMIC_BOUNDARY marker. In real CC, the static prefix
        is cached across turns to save prompt tokens.
        """
        # Return cached prompt unless forced rebuild
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
        memory = self._build_memory_section()
        if memory:
            sections.append(memory)
        claude_md = self._build_claude_md()
        if claude_md:
            sections.append(claude_md)
        # Static/dynamic boundary
        sections.append(self.DYNAMIC_BOUNDARY)
        dynamic = self._build_dynamic_context()
        if dynamic:
            sections.append(dynamic)

        self._cached_prompt = "\n\n".join(sections)
        return self._cached_prompt

    def invalidate_cache(self):
        """Force next build() to rebuild the prompt."""
        self._cached_prompt = None
    


def build_system_reminder(extra: str = None) -> dict:
    """
    Build a system-reminder user message for per-turn dynamic content.
    The teaching version keeps reminders outside the stable system prompt so
    short-lived context does not get mixed into the long-lived instructions.
    """
    parts = []
    if extra:
        parts.append(extra)
    if not parts:
        return None
    content = "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"
    return {"role": "user", "content": content}

# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

@traceable(run_type="tool", name="Bash Executor")
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

@traceable(run_type="tool", name="File Reader")
def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

@traceable(run_type="tool", name="File Writer")
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

@traceable(run_type="tool", name="File Editor")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
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
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
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
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
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
                "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                "required": ["path", "old_text", "new_text"],
            },
        }
    },]

# Global prompt builder
prompt_builder = SystemPromptBuilder(workdir=WORKDIR, tools=TOOLS)

def get_git_status() -> str:
    """Get concise git status for context."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True, text=True, timeout=5, cwd=WORKDIR
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) > 10:
                return f"Git: {len(lines)} files modified"
            return f"Git: {result.stdout.strip()}"
    except Exception:
        pass
    return ""

def get_recent_errors(messages: list) -> str:
    """Extract recent tool errors from message history."""
    errors = []
    # Look at last 5 tool responses
    for msg in reversed(messages[-10:]):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if content.startswith("Error:"):
                errors.append(content[:100])
                if len(errors) >= 3:
                    break
    if errors:
        return "Recent errors:\n" + "\n".join(f"- {e}" for e in errors)
    return ""

@traceable(name="Agent Loop with Dynamic System Prompt")
def agent_loop(messages: list, inject_reminder: bool = True):
    # Inject per-turn system reminder once at the start
    if inject_reminder:
        # Build dynamic reminder with current context
        reminder_parts = []

        # 1. Current timestamp
        reminder_parts.append(f"Current turn time: {datetime.datetime.now().isoformat()}")

        # 2. Git status (if files changed)
        git_status = get_git_status()
        if git_status:
            reminder_parts.append(git_status)

        # 3. Recent errors (if any tool failed recently)
        recent_errors = get_recent_errors(messages)
        if recent_errors:
            reminder_parts.append(recent_errors)

        # 4. Working directory (if it changed)
        current_dir = Path.cwd()
        if current_dir != WORKDIR:
            reminder_parts.append(f"Current working dir: {current_dir}")

        reminder = build_system_reminder(extra="\n".join(reminder_parts))
        if reminder:
            messages.append(reminder)

    while True:
        # Normalize messages before API call
        clean_messages = normalize_messages(messages)

        response = client.chat.completions.create(
            model=AZURE_GPT41_MODEL,
            messages=clean_messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        # Get assistant message
        assistant_msg = response.choices[0].message


        # Append assistant turn
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

        # Check if there are tool calls
        if not assistant_msg.tool_calls:
            return assistant_msg.content

        # Execute each tool call, collect results
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            name = tc.function.name

            
            handler = TOOL_HANDLERS.get(name)
            output = handler(**args) if handler else f"UNKOWN TOOL is called: {name}"
            print(f'>> {name}')

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output
            })


if __name__ == "__main__":
    complete_prompt = prompt_builder.build()
    section_count = complete_prompt.count("\n# ")
    print(f"[System prompt assembled: {len(complete_prompt)} chars, ~{section_count} sections]")

    # Initialize history with system prompt once
    history = [{"role": "system", "content": complete_prompt}]

    while True:
        try:
            query = input("Provide Your Query >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        # Handle commands outside agent loop
        if query.strip() == '/prompt':
            prompt = prompt_builder.build()  # Uses cached version
            print(f'-- System Prompt --')
            print(prompt)
            print('-- END --')
            continue

        if query.strip() == "/sections":
            prompt = prompt_builder.build()  # Uses cached version
            for line in prompt.splitlines():
                if line.startswith("# ") or line == prompt_builder.DYNAMIC_BOUNDARY:
                    print(f"  {line}")
            continue

        if query.strip() == "/reload":
            # Invalidate cache and force rebuild
            prompt_builder.invalidate_cache()
            complete_prompt = prompt_builder.build(force_rebuild=True)
            section_count = complete_prompt.count("\n# ")
            print(f"[System prompt reloaded: {len(complete_prompt)} chars, ~{section_count} sections]")
            # Reset history with new system prompt
            history = [{"role": "system", "content": complete_prompt}]
            continue

        # Add user query to history
        history.append({"role": "user", "content": query})

        # Run agent loop with system reminder injection
        final_response = agent_loop(history, inject_reminder=True)
        print(final_response)
        print()
