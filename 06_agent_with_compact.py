from utils import get_openai_client, get_prompt, normalize_messages
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import subprocess
import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, field

client, AZURE_GPT41_MODEL = get_openai_client()
client = wrap_openai(client)  # Enable LangSmith tracing


WORKDIR = Path.cwd()

SYSTEM_PROMPT = (
    f"You are a coding agent at {WORKDIR}. "
    "Keep working step by step, and use compact if the conversation gets too long."
)


CONTEXT_LIMIT = 50000
KEEP_RECENT_TOOL_RESULTS = 3
PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)

def estimate_context_size(messages: list) -> int:
    return len(str(messages))

def track_recent_file(state: CompactState, path: str) -> None:
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]

def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


@traceable(run_type="tool", name="Persist Large Output")
def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not stored_path.exists():
        stored_path.write_text(output)
    preview = output[:PREVIEW_CHARS]
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>")

@traceable(run_type="tool", name="Collect Tool Results")
def collect_tool_result_blocks(messages: list) -> list[tuple[int, dict]]:
    """Collect tool result messages (OpenAI format: role='tool')"""
    blocks = []
    for message_index, message in enumerate(messages):
        if message.get("role") == "tool":
            blocks.append((message_index, message))
    return blocks

@traceable(run_type="tool", name="Micro Compact")
def micro_compact(messages: list) -> list:
    """Compact older tool results to save context"""
    tool_results = collect_tool_result_blocks(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    
    for _, message in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        content = message.get("content", "")
        if not isinstance(content, str) or len(content) <= 120:
            continue
        message["content"] = "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
    return messages

@traceable(run_type="tool", name="Write Transcript")
def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as handle:
        for message in messages:
            handle.write(json.dumps(message, default=str) + "\n")
    return path


@traceable(run_type="llm", name="Summarize History")
def summarize_history(messages: list) -> str:
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
        model=AZURE_GPT41_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


@traceable(run_type="tool", name="Compact History")
def compact_history(messages: list, state: CompactState, focus: str | None = None) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
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

@traceable(run_type="tool", name="Bash Executor")
def run_bash(command: str, tool_use_id: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as exc:
        return f"Error: {exc}"
    output = (result.stdout + result.stderr).strip() or "(no output)"
    return persist_large_output(tool_use_id, output)
    
@traceable(run_type="tool", name="File Reader")
def run_read(path: str, tool_use_id: str, state: CompactState, limit: int | None = None) -> str:
    try:
        track_recent_file(state, path)
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        output = "\n".join(lines)
        return persist_large_output(tool_use_id, output)
    except Exception as exc:
        return f"Error: {exc}"

@traceable(run_type="tool", name="File Writer")
def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"

@traceable(run_type="tool", name="File Editor")
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"
    
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
            "name": "compact",
            "description": "Summarize earlier conversation so work can continue in a smaller context.",
            "parameters": {
                "type": "object",
                "properties": {
                            "focus": {"type": "string"},
                        },
            },
        }
    },
]

@traceable(run_type="tool", name="Execute Tool")
def execute_tool(tc, state: CompactState) -> str:
    args = json.loads(tc.function.arguments)
    name = tc.function.name
    tool_use_id = tc.id

    print(f">> {name}")

    if name == "bash":
        return run_bash(args["command"], tool_use_id)
    if name == "read_file":
        return run_read(args["path"], tool_use_id, state, args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "compact":
        return "Compacting conversation..."
    return f"Unknown tool: {name}"


@traceable(name="Agent Loop with Context Management")
def agent_loop(messages: list, state: CompactState):
    # Prepend system message if not already present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    while True:

        messages[:] = micro_compact(messages)

        if estimate_context_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages, state)

        # Normalize messages before API call
        #clean_messages = normalize_messages(messages)

        response = client.chat.completions.create(
            model=AZURE_GPT41_MODEL,
            messages=messages,
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
        
        manual_compact = False
        compact_focus = None

        # Execute each tool call, collect results
        for tc in assistant_msg.tool_calls:

            output = execute_tool(tc, state)

            if tc.function.name == "compact":
                manual_compact = True
                compact_focus = (json.loads(tc.function.arguments) or {}).get("focus")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output
            })

        if manual_compact:
            print("[manual compact]")
            messages[:] = compact_history(messages, state, focus=compact_focus)


if __name__ == "__main__":
    history = []
    compact_state = CompactState()
    while True:
        try:
            query = input("Provide Your Query >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        final_response = agent_loop(history,compact_state)
        print(final_response)
        print()

