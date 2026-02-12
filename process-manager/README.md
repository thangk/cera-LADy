# Python Process Manager

A clean, terminal-based interface to view and manage Python processes running on your system. Designed to provide a neat alternative to cluttered `ps aux | grep python` output.

## Features

- 📋 **Clean Interface**: Shows only relevant process information in a formatted table
- 🎯 **Smart Filtering**: Local mode shows only processes from current project directory
- 🌐 **Global Mode**: Option to view all Python processes by your user
- ⚡ **Interactive TUI**: Navigate with arrow keys, kill processes with confirmation
- 🔄 **Real-time Updates**: Refresh process list on demand
- 💻 **Simple List Mode**: Non-interactive output for scripting
- 🗂️ **Process Grouping**: Automatically groups processes by their parent bash script
- 📊 **Sorting Options**: Sort by PID, CPU, memory, state, time, or command
- 🎛️ **Hierarchical View**: Collapsible groups with aggregate statistics
- 🚀 **Bulk Operations**: Kill entire process groups or individual processes

## Installation

No installation required! Just run the script directly:

```bash
# Make it executable
chmod +x process_manager.py

# Run from the process-manager directory
python process_manager.py
```

## Usage

### Interactive Mode (Default)

```bash
# Show processes from current project directory only
python process_manager.py

# Show all Python processes by your user
python process_manager.py -g
python process_manager.py --global
```

### Interactive Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate through list |
| `k` | Mark selected item for killing (turns red) |
| `Enter` | Confirm kill of marked item (process or entire group) |
| `K` (Shift+K) | Kill ALL processes (requires typing "YES") |
| `Space` | Expand/collapse selected group (in group view) |
| `r` | Refresh process list |
| `g` | Toggle between local/global mode |
| `G` (Shift+G) | Toggle between flat and grouped view |
| `q` / `ESC` | Quit the application |

**Sorting Options:**
| Key | Sort By | Default Order |
|-----|---------|---------------|
| `p` | Process ID (PID) | Ascending |
| `c` | CPU Usage | Descending (highest first) |
| `m` | Memory Usage | Descending (highest first) |
| `s` | Process State | Ascending |
| `t` | CPU Time | Descending (longest first) |
| `P` (Shift+P) | Command/Path | Ascending |

*Press the same sort key twice to reverse the sort order. Current sort is shown in the header.*

**Process Management Workflow:**
1. Navigate to a process or group with `↑`/`↓`
2. Press `k` to mark it for killing (line turns red)
3. Press `Enter` to confirm and kill the process/group
4. Press any other key to cancel the kill marking

**Group View Features:**
- **Automatic Grouping**: Processes grouped by their parent bash script
- **Collapsible Groups**: Use `Space` to expand/collapse group contents  
- **Group Statistics**: See total CPU, memory usage, and process count per group
- **Bulk Kill**: Select a group and kill all its processes at once
- **Visual Indicators**: ▼ (expanded) and ▶ (collapsed) show group state

### Simple List Mode

For non-interactive use or scripting:

```bash
# List processes (local)
python process_manager.py --list

# List all processes (global)  
python process_manager.py -g --list
```

## Examples

### Basic Usage

```bash
# Interactive mode - shows processes from current directory
cd /your/project/directory
python process_manager.py
```

**Flat View Sample:**
```
Python Process Manager (LOCAL/FLAT) - 5 processes - Sort: PID↑
Directory: /home/user/my-project
================================================================================
PID      CPU%   MEM%   STAT   TIME     SCRIPT               COMMAND
12345    1.2    2.5    S      00:05:23 main.py             python src/main.py -data input.xml
12346    0.8    1.9    S      00:02:10 server.py           python src/web/backend/app.py
12347    15.4   8.2    R      01:23:45 train_model.py      python experiments/train_model.py --epochs 100
```

**Group View Sample:**
```
Python Process Manager (LOCAL/GROUPED) - 5 processes - Sort: PID↑
Directory: /home/user/my-project
Groups: 2
================================================================================
PID      CPU%   MEM%   STAT   TIME     SCRIPT               COMMAND
▼ run_experiment.sh (3 processes) - CPU: 17.4% MEM: 12.6%
  12345  1.2    2.5   S      00:05:23  main.py             python src/main.py -data input.xml
  12347  15.4   8.2   R      01:23:45  train_model.py      python experiments/train_model.py
  12348  0.8    1.9   S      00:02:10  eval.py             python src/eval.py --model bert
▶ web_server.sh (1 processes) - CPU: 0.8% MEM: 1.9%
ungrouped (1 processes) - CPU: 2.1% MEM: 3.2%
  12349  2.1    3.2   R      00:01:05  debug.py            python debug.py
```

### Global Mode

```bash
# Show all Python processes by your user
python process_manager.py --global
```

### Quick List

```bash
# Just show the list without interactive interface
python process_manager.py --list
```

## Command Line Options

```
usage: process_manager.py [-h] [-g] [--list] [--version]

Manage Python processes with a clean terminal interface

options:
  -h, --help     show this help message and exit
  -g, --global   Show all Python processes by user (not just current directory)
  --list         Simple list view instead of interactive mode
  --version      show program's version number and exit
```

## Process Information

The interface displays the following information for each process:

- **PID**: Process ID
- **CPU%**: Current CPU usage percentage  
- **MEM%**: Memory usage percentage
- **STAT**: Process state (R=Running, S=Sleeping, etc.)
- **TIME**: Total CPU time used
- **SCRIPT**: Just the script filename (e.g., `main.py`)
- **COMMAND**: Full command with arguments (truncated for display)

## Local vs Global Mode

### Local Mode (Default)
- Shows only Python processes running from the current directory
- Perfect for managing processes related to your current project
- Reduces clutter when working on specific projects

### Global Mode (`-g` flag)
- Shows all Python processes owned by your user
- Useful for system-wide process management
- Can be toggled interactively with the `g` key

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)
- Unix-like system with `ps` command (Linux, macOS, WSL)

## Troubleshooting

### No processes shown
- Make sure you're running Python scripts from the current directory (local mode)
- Try global mode with `-g` flag
- Verify processes are actually running: `ps aux | grep python`

### Permission denied when killing processes
- You can only kill processes owned by your user
- Some system processes may require elevated permissions

### Display issues
- Ensure your terminal supports cursor movement
- Minimum terminal width of 80 characters recommended
- Use `--list` mode if interactive mode has issues

## License

This tool is provided as-is for educational and development purposes.