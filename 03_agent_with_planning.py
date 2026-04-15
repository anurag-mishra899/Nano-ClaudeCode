from utils import get_openai_client, get_prompt, normalize_messages
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field

client, AZURE_GPT41_MODEL = get_openai_client()
client = wrap_openai(client)  # Enable LangSmith tracing

WORKDIR = Path.cwd()
SYSTEM_PROMPT = get_prompt(prompt_type='planner',workdir=WORKDIR)
PLAN_REMINDER_INTERVAL = 3


@dataclass
class Task:
    content: str 
    status: str = 'pending'
    active_form: str = ''

@dataclass
class TaskList:
    tasks: list[Task] = field(default_factory=list)
    round_since_update: int = 0


class TaskManager:
    def __init__(self):
        self.state = TaskList()

    @traceable(run_type="tool", name="Update Plan")
    def update_plan(self, items: list) -> str:

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

            normalized.append(
                Task(
                    content=content,
                    status=status,
                    active_form=active_form,
                ))
            
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")
        
        self.state.items = normalized
        self.state.rounds_since_update = 0

        return self.render_plan()
    

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"
    
    def render_plan(self) -> str:

        if not self.state.items:
            return "No session plan yet."
        
        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            
            line = f"{marker} {item.content}"
            
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        
        completed = sum(1 for item in self.state.items if item.status == "completed")
        
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        
        return "\n".join(lines)


planner = TaskManager()

def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
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
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    
@traceable(run_type="tool", name="File Reader")
def run_read(path: str, limit: int = 0) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

@traceable(run_type="tool", name="File Writer")
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
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

# -- Concurrency safety classification --
# Read-only tools can safely run in parallel; mutating tools must be serialized.
CONCURRENCY_SAFE = {"read_file"}
CONCURRENCY_UNSAFE = {"write_file", "edit_file"}

# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: planner.update_plan(kw["items"]),
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
    },
    {
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
                                "activeForm": {
                                    "type": "string",
                                    "description": "Optional present-continuous label.",
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["items"],
            },
        }
    },
] ## added 'todo' as tool

@traceable(name="Agent Loop with Planning")
def agent_loop(messages: list):
    # Prepend system message if not already present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

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

        used_todo = False

        # Execute each tool call, collect results
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            name = tc.function.name

            handler = TOOL_HANDLERS.get(name)
            output = handler(**args) if handler else f"UNKOWN TOOL is called: {name}"
            print(f'>> {name}')
            if name == 'todo':
                used_todo = True

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output
            })

        if used_todo:
            planner.state.rounds_since_update = 0
        else:
            planner.note_round_without_update()
            reminder = planner.reminder()
            if reminder:
                messages.insert(0, {"type": "text", "text": reminder})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("Provide Your Query >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        final_response = agent_loop(history)
        print(final_response)
        print()




