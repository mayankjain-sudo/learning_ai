# System Monitoring MCP Server

A robust Model Context Protocol (MCP) server built with Python and `FastMCP`. This server provides Large Language Models (LLMs) and agents with real-time access to system statistics, resource usage, process management, and network information via standard tool calls.

## 🌟 Features

The server exposes a comprehensive suite of system monitoring tools to the LLM:

- **System Health & Stats**: `get_system_health`, `get_system_stats`, `get_system_info`
- **Resource Monitoring**: `get_cpu_info`, `get_memory_info`, `get_disk_info`, `get_disk_partitions`
- **Process Management**: `list_processes`, `get_top_processes`, `get_process_info`, `find_processes_by_name`, `get_process_count`, `kill_process`
- **Network Diagnostics**: `get_network_stats`, `get_open_ports` (Note: requires elevated privileges for full visibility)
- **Time & Uptime**: `get_system_time`, `get_boot_time`, `get_uptime`, `get_system_uptime`

The server also includes built-in **Prompts** to easily initiate tasks:
- `system_health_check`: Directs the LLM to run a full diagnostic and summarize resources.
- `troubleshoot_performance`: Directs the LLM to investigate sluggish performance and find processes to optimize or kill.

## 📋 Prerequisites

Before installing, ensure you have the following installed on your machine:
- **Python**: Version `3.13` or higher.
- **uv**: The blazing-fast Python package manager. (Install via `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux).
- **Node.js/npm**: Required only if you want to use the web-based MCP Inspector to test the tools.

## 🚀 Installation

1. **Clone the repository** (or download the source code) and navigate to the project directory:
   ```bash
   cd learning_ai/mcp_server/sys_monitoring_mcp
   ```

2. **Install dependencies** using `uv`. The project relies on `mcp` and `psutil`:
   ```bash
   # This will create an isolated virtual environment and install dependencies from uv.lock
   uv sync
   ```
   *(Note: If you run into compiler errors for `psutil` on macOS, ensure you have Xcode command line tools installed by running `xcode-select --install`).*

## 💻 How to Run and Test

Because MCP servers communicate over `stdio` (Standard Input/Output) using JSON-RPC, running `uv run main.py` directly in the terminal will appear "stuck" as it waits for incoming client messages in the background. 

To interact with the server, use one of the following methods:

### Option 1: Use the official MCP Inspector (Web UI)
The easiest way to test your tools is using the official MCP Inspector. Run the following command in your terminal:

```bash
npx @modelcontextprotocol/inspector uv run main.py
```
*This will start a local web server. Open the provided `localhost` URL in your browser to interactively test the system monitoring tools.*

### Option 2: Use the FastMCP Dev Server
Since the project uses `FastMCP`, you can launch the built-in development inspector:

```bash
uv run mcp dev main.py
```

### Option 3: Integrating with Claude Desktop
To allow Claude to monitor your system directly, add this server to your Claude Desktop configuration file.

1. Open your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following configuration to the `mcpServers` object:

```json
{
  "mcpServers": {
    "sys-monitoring": {
      "command": "uv",
      "args": [
        "--directory",
        "/<YOUR_FOLDER_PATH>/learning_ai/mcp_server/sys_monitoring_mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```
3. **Restart Claude Desktop**. You should now see the system monitoring tools available (indicated by a hammer icon) when chatting with Claude.

## ⚠️ Troubleshooting & Notes

- **AccessDenied Errors on Open Ports**: 
  The `get_open_ports` tool requires elevated system privileges to view network connections belonging to other users' processes. If run without admin/sudo rights, it will safely catch the error and return an `AccessDenied` message to the LLM. 
  - *Workaround*: If you need full visibility, run the server with elevated privileges (e.g., `sudo uv run main.py`).
- **Missing `psutil` Module**: 
  If you encounter `ModuleNotFoundError: No module named 'psutil'`, ensure you are running the script via `uv run main.py` so it utilizes the isolated environment where `psutil` is installed, rather than your global python installation.

## 📄 License

This project is open-source and available for modification.