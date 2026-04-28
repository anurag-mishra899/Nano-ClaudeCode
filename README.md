# Nano-ClaudeCode

## Project Overview
Nano-ClaudeCode is a collection of Python scripts and utilities demonstrating various agent-based design patterns, leveraging OpenAI APIs and LangSmith for tracing. The repository includes examples ranging from basic single-agent operation to team-based and autonomous agent systems.

## Python Files Overview

| File | Description |
|------|-------------|
| `00_complete_harness.py` | Complete unified agent harness combining all features from scripts 01-16. Production-ready modular agent with permission system, task management, memory, and hooks. |
| `01_basic_agent.py` | Basic agent loop with tool execution. Foundation for all other agents. |
| `02_tools_use_agent.py` | Agent with file operations (read, write, edit, bash) and safe_path validation for workspace security. |
| `03_agent_with_planning.py` | Agent with in-memory task planning using TaskManager. Demonstrates task decomposition and tracking. |
| `04_agent_with_subagent.py` | Agent with subagent spawning capabilities. Shows hierarchical agent architecture with delegation. |
| `05_agent_with_skills.py` | Agent with skill registry for specialized instructions. Enables domain-specific task handling. |
| `06_agent_with_compact.py` | Agent with context management and CompactState. Handles conversation history compression to stay within context limits. |
| `07_agent_with_permission.py` | Agent with permission system and BashSecurityValidator. Implements safety checks for dangerous operations. |
| `08_agent_with_hooks.py` | Agent with hook system for extensibility. Allows custom pre/post tool execution callbacks. |
| `09_agent_with_session_memeory.py` | Agent with persistent memory across sessions using MemoryManager. Stores learnings in `.memory/` directory. |
| `10_agent_with_dynamic_system_prompt.py` | Agent with dynamic system prompt builder. Assembles prompts from multiple sources (skills, memory, git status). |
| `11_agent_with_task_system.py` | Agent with persistent task board stored in `.tasks/` directory. File-based task tracking system. |
| `12_agent_with_background_tasks.py` | Agent with background task execution using threading. Runs long operations (tests, builds) asynchronously. |
| `13_agent_teams.py` | Team collaboration with persistent named agents and file-based JSONL inboxes. Multiple models coordinated through message passing. |
| `14_team_protocols.py` | Structured handshakes between team members. Implements shutdown and plan approval protocols with request_id correlation. |
| `15_autonomous_agents.py` | Autonomous agents with idle cycle polling. Agents find and claim work from task board without explicit instructions. |
| `16_worktree_task_isolation.py` | Directory-level isolation for parallel task execution. Uses git worktrees for non-colliding parallel work. |
| `utils.py` | Shared utility functions for OpenAI client initialization, prompt management, and message normalization. |
| `hello.py` | Simple test/example file. |

## Installation

### Using UV (Recommended)
[UV](https://docs.astral.sh/uv/) is a fast Python package installer and resolver:

1. Install UV:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository:
   ```bash
   git clone <repo-url>
   cd Nano-ClaudeCode
   ```

3. Create virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

### Alternative Installation Methods
Using pip:
```bash
pip install -r requirements.txt
```

Using Poetry:
```bash
poetry install
```

Using pyproject.toml:
```bash
pip install .
```

## Usage
Each script in the repository demonstrates a different agent use case. Example:
```bash
python 01_basic_agent.py
```
Refer to comments within each file for purpose and usage. Update configurations as required.

## Testing
To ensure proper function, run the scripts and check for expected output. If there are test scripts or notebooks, run:
```bash
pytest
printf '\nOr check test_notebook.ipynb for manual evaluation.'
```

---

*For more project details or customization, please update this README or reach out to the project lead.*
