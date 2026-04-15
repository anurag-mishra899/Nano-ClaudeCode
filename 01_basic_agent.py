from utils import get_openai_client, get_prompt, normalize_messages
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import os
import subprocess
import json

client, AZURE_GPT41_MODEL = get_openai_client()
client = wrap_openai(client)  # Enable LangSmith tracing

SYSTEM_PROMPT = get_prompt('basic')

@traceable(run_type="tool", name="Bash Executor")
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

TOOLS = [{
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
}]

@traceable(name="Agent Loop")
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

        # Execute each tool call, collect results
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\033[33m$ {args['command']}\033[0m")
            output = run_bash(args["command"])

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output
            })
        
        print(messages)

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
