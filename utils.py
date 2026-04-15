from openai import AzureOpenAI
from dotenv import load_dotenv
import os
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

SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


def get_openai_client():
    client = AzureOpenAI(
        api_version=AZURE_OPENAI_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    return client, AZURE_GPT41_MODEL


def normalize_messages(messages: list) -> list:
    """Clean up messages before sending to OpenAI API.
    Three jobs:
    1. Strip internal metadata fields the API doesn't understand
    2. Ensure every tool_call has a matching tool result (insert placeholder if missing)
    3. Merge consecutive same-role messages (API requires strict alternation)
    """
    cleaned = []
    for msg in messages:
        # Strip internal metadata (keys starting with _)
        clean = {k: v for k, v in msg.items() if not k.startswith("_")}
        cleaned.append(clean)

    # Collect existing tool result IDs
    existing_results = set()
    for msg in cleaned:
        if msg.get("role") == "tool":
            existing_results.add(msg.get("tool_call_id"))

    # Find orphaned tool_calls and insert placeholder results
    for msg in cleaned:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            if tool_call.get("id") not in existing_results:
                cleaned.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "(cancelled)"
                })

    # Merge consecutive same-role messages (but preserve tool messages and messages with tool_calls)
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        prev = merged[-1]
        # Don't merge tool messages (each needs unique tool_call_id) or messages with tool_calls
        if (msg["role"] == prev["role"] and
            msg["role"] not in ("tool",) and
            not prev.get("tool_calls") and
            not msg.get("tool_calls")):
            # Merge content for user/assistant messages without tool_calls
            prev["content"] = str(prev.get("content", "")) + "\n" + str(msg.get("content", ""))
        else:
            merged.append(msg)
    return merged
