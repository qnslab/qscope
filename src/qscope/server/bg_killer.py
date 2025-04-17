import json
from datetime import datetime
from pathlib import Path

import psutil
from loguru import logger


def get_servers_dir() -> Path:
    """Get the directory for storing server PID files."""
    base_dir = Path.home() / ".qscope"
    servers_dir = base_dir / "running_servers"
    servers_dir.mkdir(parents=True, exist_ok=True)
    return servers_dir


def list_running_servers() -> list[dict]:
    """Get info about all currently running servers."""
    servers = []
    for pid_file in get_servers_dir().glob("server_*.json"):
        try:
            with pid_file.open() as f:
                server_info = json.load(f)
            # Check if process exists and add running status
            try:
                proc = psutil.Process(server_info["pid"])
                server_info["running"] = True
            except psutil.NoSuchProcess:
                server_info["running"] = False
            servers.append(server_info)
        except:
            continue
    return servers


def kill_qscope_servers() -> int:
    """Find and kill all running Qscope Server processes."""
    killed = 0
    servers_dir = get_servers_dir()

    if not servers_dir.exists():
        return 0

    for pid_file in servers_dir.glob("server_*.json"):
        try:
            with pid_file.open() as f:
                server_info = json.load(f)

            pid = server_info["pid"]
            try:
                proc = psutil.Process(pid)
                logger.info(
                    f"Killing server PID {pid} started at {server_info['timestamp']}"
                )
                proc.kill()
                killed += 1
            except psutil.NoSuchProcess:
                logger.debug(f"Server PID {pid} no longer exists")

            # Clean up stale PID file
            pid_file.unlink()

        except Exception as e:
            logger.error(f"Error processing {pid_file}: {e}")
            continue

    return killed


def cleanup_stale_servers():
    """Remove PID files for servers that no longer exist."""
    for pid_file in get_servers_dir().glob("server_*.json"):
        try:
            with pid_file.open() as f:
                server_info = json.load(f)
            try:
                psutil.Process(server_info["pid"])
            except psutil.NoSuchProcess:
                pid_file.unlink()
        except:
            # If we can't read the file, consider it stale
            try:
                pid_file.unlink()
            except:
                pass


if __name__ == "__main__":
    killed = kill_qscope_servers()
    logger.info(f"Killed {killed} Qscope Server processes")

# =============================================================================
# === Improvements
# =============================================================================

'''
Here's an enhanced version with those additional features:

1. First, let's modify `server_script.py` to include a timestamp in the title:

```python
from datetime import datetime
from setproctitle import setproctitle

# Format: "Qscope Server (2024-01-20 15:30:45)"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
setproctitle(f'Qscope Server ({timestamp})')
```

2. Here's an enhanced version of `scripts/kill_bg_servers_script.py`:

```python
#!/usr/bin/env python
import argparse
from datetime import datetime, timedelta
import psutil
from loguru import logger

def parse_timestamp(proc_name: str) -> datetime | None:
    """Extract timestamp from process name."""
    try:
        # Extract "YYYY-MM-DD HH:MM:SS" from "Qscope Server (YYYY-MM-DD HH:MM:SS)"
        timestamp_str = proc_name.split('(')[1].split(')')[0]
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except (IndexError, ValueError):
        return None

def kill_qscope_servers(force: bool = False, max_age_hours: float = 24.0) -> tuple[int, list[str]]:
    """Find and kill Qscope Server processes.

    Parameters
    ----------
    force : bool, optional
        If True, kill all servers regardless of age, by default False
    max_age_hours : float, optional
        Maximum allowed age in hours before force killing, by default 24.0

    Returns
    -------
    tuple[int, list[str]]
        Number of processes killed and list of log messages
    """
    killed = 0
    logs = []
    now = datetime.now()
    max_age = timedelta(hours=max_age_hours)

    for proc in psutil.process_iter(['name', 'pid', 'create_time']):
        try:
            proc_name = proc.name()
            if not proc_name.startswith('Qscope Server'):
                continue

            pid = proc.pid
            create_time = datetime.fromtimestamp(proc.create_time())
            age = now - create_time
            
            # Get timestamp from process title if available
            title_timestamp = parse_timestamp(proc_name)
            if title_timestamp:
                age = now - title_timestamp

            msg = f"Found server PID {pid} (age: {age})"
            
            # Decision logic for killing
            should_kill = False
            if force:
                should_kill = True
                msg += " - Force killing"
            elif age > max_age:
                should_kill = True
                msg += f" - Exceeds max age of {max_age_hours}h"
            
            logs.append(msg)
            logger.info(msg)

            if should_kill:
                proc.kill()
                killed += 1
                kill_msg = f"Killed server process {pid}"
                logs.append(kill_msg)
                logger.info(kill_msg)
            else:
                warn_msg = f"Skipping server {pid} (use --force to kill)"
                logs.append(warn_msg)
                logger.warning(warn_msg)

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            err_msg = f"Error accessing process: {e}"
            logs.append(err_msg)
            logger.error(err_msg)
            continue

    return killed, logs

def main():
    parser = argparse.ArgumentParser(description="Kill orphaned Qscope Server processes")
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Kill all servers regardless of age"
    )
    parser.add_argument(
        "--max-age", 
        type=float,
        default=24.0,
        help="Maximum allowed server age in hours before force killing"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="kill_servers.log",
        help="Log file path"
    )

    args = parser.parse_args()

    # Add file handler to logger
    logger.add(args.log_file, rotation="1 week")

    try:
        killed, logs = kill_qscope_servers(args.force, args.max_age)
        summary = f"Killed {killed} Qscope Server process(es)"
        logger.info(summary)
        
        if killed == 0 and not args.force:
            logger.info("Use --force to kill all servers regardless of age")
            
    except Exception as e:
        logger.exception("Error during server cleanup")
        raise

if __name__ == '__main__':
    main()
```

3. Enhanced `dodo.py` task:

```python
def task_kill_servers():
    """Kill any orphaned Qscope Server processes.

    Usage:
        doit kill_servers              # Normal cleanup (24h max age)
        doit kill_servers --force      # Force kill all servers
        doit kill_servers --max-age=2  # Kill servers older than 2 hours
    """
    def cmd_builder(force: bool, max_age: float):
        cmd = ["python", "scripts/kill_bg_servers_script.py"]
        if force:
            cmd.append("--force")
        if max_age != 24.0:
            cmd.extend(["--max-age", str(max_age)])
        return " ".join(cmd)

    return {
        "actions": [CmdAction(cmd_builder)],
        "params": [
            {
                "name": "force",
                "short": "f", 
                "type": bool,
                "default": False,
                "help": "Force kill all servers"
            },
            {
                "name": "max_age",
                "short": "m",
                "type": float,
                "default": 24.0,
                "help": "Max server age in hours"
            }
        ],
        "verbosity": 2
    }
```

This enhanced version provides:

1. Process titles with timestamps for better tracking
2. Age-based cleanup (default 24h max age)
3. Force kill option
4. Detailed logging of all actions
5. Command-line arguments for flexibility
6. Rotation of log files
7. Better error handling and reporting

You can use it like:
```bash
doit kill_servers              # Normal cleanup
doit kill_servers --force      # Force kill all
doit kill_servers --max-age=2  # Kill servers older than 2h
```

The script will:
- Log all actions to both console and file
- Warn about servers that weren't killed
- Provide detailed information about each server found
- Handle errors gracefully
- Create rotating log files
'''
