from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import subprocess
import json
try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass


load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY= os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_VERSION=os.getenv('AZURE_OPENAI_VERSION')
AZURE_GPT4O_MODEL=os.getenv('AZURE_GPT4O_MODEL')
AZURE_GPT41_MODEL=os.getenv('AZURE_GPT41_MODEL')

client = AzureOpenAI(
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "type":"function",
    "name": "bash",
    "description": "Run a shell command.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

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
    

def agent_loop(messages: list):
    while True:
        response = client.responses.create(
            model=AZURE_GPT41_MODEL,
            instructions=SYSTEM,
            input=messages,
            tools=TOOLS,
            max_output_tokens=8000,
        )
        # Append assistant turn
        messages += response.output

        #print(messages)

        tool_calls = [item for item in response.output if item.type=='function_call']

        if not tool_calls:
            #print(response)
            return response
        
        # Execute each tool call, collect results

        for block in tool_calls:
            args = json.loads(block.arguments)
            print(f"\033[33m$ {args['command']}\033[0m")
            output = run_bash(args["command"])
            #print(output[:200])
            messages.append({"type": "function_call_output", 
                            "call_id": block.call_id,
                            "output": output})



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
        print(final_response.output_text)
        print()
