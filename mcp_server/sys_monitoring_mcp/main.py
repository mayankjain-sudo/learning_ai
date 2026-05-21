import time
from datetime import datetime, timezone

import psutil  # type: ignore
from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent
import signal

mcp = FastMCP("sys-monitoring-mcp")
              


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"

def _format_uptime(seconds: int) -> str:
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    parts : list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:        
        parts.append(f"{seconds}s")
        
    return " ".join(parts)

@mcp.tool()
def get_system_stats() -> dict:
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    uptime_seconds = int(time.time() - psutil.boot_time())
    
    return {
        "cpu_usage": f"{cpu_usage:.1f}%",
        "memory_usage": f"{memory_info.percent:.1f}% ({_format_bytes(memory_info.used)} / {_format_bytes(memory_info.total)})",
        "disk_usage": f"{disk_info.percent:.1f}% ({_format_bytes(disk_info.used)} / {_format_bytes(disk_info.total)})",
        "uptime": _format_uptime(uptime_seconds)
    }

@mcp.tool()
def list_processes() -> list[dict]:
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append({
                "pid": proc.info['pid'],
                "name": proc.info['name'],
                "cpu_percent": f"{proc.info['cpu_percent'] or 0.0:.1f}%",
                "memory_percent": f"{proc.info['memory_percent'] or 0.0:.1f}%"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

@mcp.tool()
def kill_process(pid: int) -> str:
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=3)
        return f"Process {pid} terminated successfully."
    except psutil.NoSuchProcess:
        return f"Process {pid} does not exist."
    except psutil.AccessDenied:
        return f"Permission denied to terminate process {pid}."
    except psutil.TimeoutExpired:
        return f"Failed to terminate process {pid} within timeout."
    
@mcp.tool()
def get_system_uptime() -> str:
    uptime_seconds = int(time.time() - psutil.boot_time())
    return _format_uptime(uptime_seconds)  

@mcp.tool()
def get_system_time() -> str:
    now = datetime.now(timezone.utc)
    return now.isoformat()  

@mcp.tool()
def get_system_info() -> dict:
    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_total": _format_bytes(psutil.virtual_memory().total),
        "disk_total": _format_bytes(psutil.disk_usage('/').total)
    }

@mcp.tool()
def get_top_processes(n: int = 5) -> list[dict]:
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append({
                "pid": proc.info['pid'],
                "name": proc.info['name'],
                "cpu_percent": proc.info['cpu_percent'] or 0.0,
                "memory_percent": proc.info['memory_percent'] or 0.0
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    processes.sort(key=lambda p: (p['cpu_percent'], p['memory_percent']), reverse=True)
    return processes[:n]

@mcp.tool()
def get_process_info(pid: int) -> dict:
    try:
        proc = psutil.Process(pid)
        return {
            "pid": proc.pid,
            "name": proc.name(),
            "status": proc.status(),
            "cpu_percent": f"{proc.cpu_percent(interval=0.1):.1f}%",
            "memory_percent": f"{proc.memory_percent():.1f}%"
        }
    except psutil.NoSuchProcess:
        return {"error": f"Process {pid} does not exist."}
    except psutil.AccessDenied:
        return {"error": f"Permission denied to access process {pid}."}

@mcp.tool()
def get_disk_partitions() -> list[dict]:
    partitions = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            partitions.append({
                "device": part.device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "total": _format_bytes(usage.total),
                "used": _format_bytes(usage.used),
                "free": _format_bytes(usage.free),
                "percent": f"{usage.percent:.1f}%"
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return partitions

@mcp.tool()
def get_network_stats() -> dict:
    net_io = psutil.net_io_counters()
    return {
        "bytes_sent": _format_bytes(net_io.bytes_sent),
        "bytes_recv": _format_bytes(net_io.bytes_recv),
        "packets_sent": net_io.packets_sent,
        "packets_recv": net_io.packets_recv
    }

@mcp.tool()
def get_boot_time() -> str:
    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    return boot_time.isoformat()

@mcp.tool()
def get_uptime() -> str:
    uptime_seconds = int(time.time() - psutil.boot_time())
    return _format_uptime(uptime_seconds)

@mcp.tool()
def get_cpu_info() -> dict:
    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_freq": f"{psutil.cpu_freq().current:.1f} MHz",
        "cpu_usage": f"{psutil.cpu_percent(interval=1):.1f}%"
    }

@mcp.tool()
def get_memory_info() -> dict:
    memory_info = psutil.virtual_memory()
    return {
        "total": _format_bytes(memory_info.total),
        "available": _format_bytes(memory_info.available),
        "used": _format_bytes(memory_info.used),
        "free": _format_bytes(memory_info.free),
        "percent": f"{memory_info.percent:.1f}%"
    }

@mcp.tool()
def get_disk_info() -> dict:
    disk_info = psutil.disk_usage('/')
    return {
        "total": _format_bytes(disk_info.total),
        "used": _format_bytes(disk_info.used),
        "free": _format_bytes(disk_info.free),
        "percent": f"{disk_info.percent:.1f}%"
    }
    
@mcp.tool()
def get_process_count() -> int:
    return len(psutil.pids())

@mcp.tool()
def find_processes_by_name(name: str) -> list[dict]:
    processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if name.lower() in proc.info['name'].lower():
                processes.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

@mcp.tool()
def get_open_ports() -> list[dict]:
    ports = []
    try:
        for conn in psutil.net_connections():
            if conn.status == 'LISTEN':
                ports.append({
                    "pid": conn.pid,
                    "local_address": f"{conn.laddr.ip}:{conn.laddr.port}",
                    "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                    "status": conn.status
                })
    except psutil.AccessDenied:
        return [{"error": "AccessDenied: Viewing network connections requires elevated privileges (e.g., root/sudo)."}]
    return ports

@mcp.tool()
def get_system_health() -> dict:
    return {
        "cpu_usage": f"{psutil.cpu_percent(interval=1):.1f}%",
        "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
        "disk_usage": f"{psutil.disk_usage('/').percent:.1f}%"
    }

@mcp.prompt(name="system_health_check", description="A prompt to request a full system health report")
def prompt_system_health_check() -> list[PromptMessage]:
    return [
        PromptMessage(
            role="user", 
            content=TextContent(
                type="text", 
                text="Please run a complete system health check using the available monitoring tools. Summarize CPU, Memory, Disk usage, and list the top 5 resource-heavy processes."
            )
        )
    ]

@mcp.prompt(name="troubleshoot_performance", description="A prompt to investigate a slow or unresponsive system")
def prompt_troubleshoot_performance() -> list[PromptMessage]:
    return [
        PromptMessage(
            role="user", 
            content=TextContent(
                type="text", 
                text="My computer is feeling sluggish. Can you analyze the current system performance, find out if any specific processes are hogging the CPU or Memory, and suggest what I might want to kill or investigate?"
            )
        )
    ]

def main() -> None:
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
