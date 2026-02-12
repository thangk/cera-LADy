#!/usr/bin/env python3
"""
Process Manager for Python Scripts
A terminal-based interface to view and manage Python processes
"""

import argparse
import os
import signal
import subprocess
import sys
from typing import List, Dict, Tuple
import curses
from datetime import datetime


class ProcessInfo:
    def __init__(self, pid: str, ppid: str, cpu: str, mem: str, vsz: str, rss: str, 
                 tty: str, stat: str, start: str, time: str, command: str):
        self.pid = pid
        self.ppid = ppid
        self.cpu = cpu
        self.mem = mem
        self.vsz = vsz
        self.rss = rss
        self.tty = tty
        self.stat = stat
        self.start = start
        self.time = time
        self.command = command
        self.script_name = self._extract_script_name()
        self.parent_script = None  # Will be populated by process manager
        self.children = []  # Child processes
        
    def _extract_script_name(self) -> str:
        """Extract just the script name from the full command"""
        parts = self.command.split()
        if len(parts) >= 2:
            script_path = parts[1]
            return os.path.basename(script_path)
        return "python"
    
    def get_short_command(self, max_length: int = 60) -> str:
        """Get a shortened version of the command for display"""
        if len(self.command) <= max_length:
            return self.command
        return self.command[:max_length-3] + "..."


class ProcessGroup:
    def __init__(self, parent_script: str, parent_pid: str = None):
        self.parent_script = parent_script  # The bash script name
        self.parent_pid = parent_pid
        self.processes: List[ProcessInfo] = []
        self.is_expanded = True  # Whether to show child processes
        
    def add_process(self, process: ProcessInfo):
        self.processes.append(process)
        process.parent_script = self.parent_script
        
    def get_total_cpu(self) -> float:
        return sum(float(p.cpu) for p in self.processes)
        
    def get_total_mem(self) -> float:
        return sum(float(p.mem) for p in self.processes)
        
    def get_process_count(self) -> int:
        return len(self.processes)


class ProcessManager:
    def __init__(self, global_mode: bool = False):
        self.global_mode = global_mode
        self.current_dir = self._find_project_root()
        self.processes: List[ProcessInfo] = []
        self.process_groups: Dict[str, ProcessGroup] = {}
        self.display_items = []  # Groups and individual processes for display
        self.selected_index = 0
        self.marked_for_kill = False  # Track if current process is marked for killing
        self.sort_by = 'pid'  # Default sort: pid, cpu, mem, stat, time, command
        self.sort_reverse = False  # Sort direction
        self.show_grouped = False  # Toggle between grouped and flat view
        
    def _find_project_root(self) -> str:
        """Find the git repository root, fallback to current directory"""
        try:
            # Try to find git root
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Fallback to current directory
        return os.getcwd()
    
    def _get_process_hierarchy(self) -> Dict[str, str]:
        """Get mapping of PID -> parent command for process hierarchy"""
        try:
            # Get process tree with parent info
            cmd = ["ps", "-eo", "pid,ppid,comm"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {}
                
            hierarchy = {}
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    pid = parts[0]
                    ppid = parts[1]
                    comm = parts[2]
                    hierarchy[pid] = {'ppid': ppid, 'comm': comm}
                    
            return hierarchy
        except Exception:
            return {}
    
    def _find_parent_script(self, pid: str, hierarchy: Dict[str, str]) -> str:
        """Find the originating bash script for a Python process"""
        visited = set()
        current_pid = pid
        
        while current_pid and current_pid != '1' and current_pid not in visited:
            visited.add(current_pid)
            
            if current_pid not in hierarchy:
                break
                
            parent_info = hierarchy[current_pid]
            parent_pid = parent_info['ppid']
            parent_comm = parent_info['comm']
            
            # If parent is bash, check if we can find the script name
            if parent_comm in ['bash', 'sh', 'zsh']:
                # Try to get full command line for the parent
                try:
                    with open(f'/proc/{parent_pid}/cmdline', 'r') as f:
                        cmdline = f.read().replace('\x00', ' ').strip()
                        if '.sh' in cmdline or 'bash' in cmdline:
                            # Extract script name
                            for part in cmdline.split():
                                if part.endswith('.sh'):
                                    return os.path.basename(part)
                                elif '/' in part and not part.startswith('-'):
                                    script_name = os.path.basename(part)
                                    if script_name and not script_name.startswith('-'):
                                        return script_name
                except:
                    pass
                    
            current_pid = parent_pid
            
        return "ungrouped"
        
    def get_python_processes(self) -> List[ProcessInfo]:
        """Get all Python processes for the current user and build groups"""
        try:
            # Get detailed process information
            cmd = ["ps", "aux"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return []
                
            # Get process hierarchy for grouping
            hierarchy = self._get_process_hierarchy()
                
            processes = []
            self.process_groups.clear()  # Reset groups
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                parts = line.split()
                if len(parts) < 11:
                    continue
                    
                user = parts[0]
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                vsz = parts[4]
                rss = parts[5]
                tty = parts[6]
                stat = parts[7]
                start = parts[8]
                time = parts[9]
                command = ' '.join(parts[10:])
                
                # Filter for current user and Python processes
                current_user = os.getenv('USER', 'unknown')
                if user == current_user and 'python' in command.lower():
                    # Skip the process manager itself
                    if 'process_manager.py' in command:
                        continue
                        
                    # If not global mode, only show processes from current project
                    if not self.global_mode:
                        # Check if the process belongs to this project
                        is_local = False
                        
                        # Method 1: Check if project path appears in the command
                        if self.current_dir in command:
                            is_local = True
                        else:
                            # Method 2: Check the actual working directory of the process
                            # This is more reliable as it shows where the process was started from
                            try:
                                cwd_path = f"/proc/{pid}/cwd"
                                if os.path.exists(cwd_path):
                                    # Read the symlink to get actual working directory
                                    process_cwd = os.readlink(cwd_path)
                                    # Check if process is running from within our project tree
                                    if process_cwd.startswith(self.current_dir):
                                        is_local = True
                            except (OSError, PermissionError):
                                # If we can't read /proc, fall back to command string analysis
                                # Check if any script file in the command exists in our project
                                for part in command.split()[1:]:
                                    if part.endswith('.py'):
                                        # Try to find this file in our project
                                        for root, dirs, files in os.walk(self.current_dir):
                                            if os.path.basename(part) in files:
                                                is_local = True
                                                break
                                        if is_local:
                                            break
                        
                        if not is_local:
                            continue
                    
                    proc_info = ProcessInfo(pid, "", cpu, mem, vsz, rss, tty, stat, start, time, command)
                    
                    # Find parent script
                    parent_script = self._find_parent_script(pid, hierarchy)
                    
                    # Add to appropriate group
                    if parent_script not in self.process_groups:
                        self.process_groups[parent_script] = ProcessGroup(parent_script)
                    
                    self.process_groups[parent_script].add_process(proc_info)
                    processes.append(proc_info)
                    
            return processes
            
        except Exception as e:
            print(f"Error getting processes: {e}")
            return []
    
    def kill_process(self, pid: str) -> bool:
        """Kill a process by PID"""
        # Safety check: never kill our own process
        if not pid or not pid.isdigit():
            return False
        if int(pid) == os.getpid():
            return False
            
        try:
            os.kill(int(pid), signal.SIGTERM)
            return True
        except (OSError, ValueError):
            try:
                # Try SIGKILL if SIGTERM fails
                os.kill(int(pid), signal.SIGKILL)
                return True
            except (OSError, ValueError):
                return False
    
    def kill_all_processes(self) -> Tuple[int, int]:
        """Kill all processes in the current list. Returns (success_count, total_count)"""
        if not self.processes:
            return 0, 0
            
        success_count = 0
        total_count = len(self.processes)
        
        for proc in self.processes:
            if self.kill_process(proc.pid):
                success_count += 1
                
        return success_count, total_count
    
    def kill_group(self, group_name: str) -> Tuple[int, int]:
        """Kill all processes in a group. Returns (success_count, total_count)"""
        if group_name not in self.process_groups:
            return 0, 0
            
        group = self.process_groups[group_name]
        success_count = 0
        total_count = len(group.processes)
        
        for proc in group.processes:
            if self.kill_process(proc.pid):
                success_count += 1
                
        return success_count, total_count
    
    def _build_display_items(self):
        """Build the display items list based on current view mode"""
        self.display_items.clear()
        
        if self.show_grouped:
            # Grouped view: show groups and their children
            for group_name, group in sorted(self.process_groups.items()):
                # Add group header
                self.display_items.append(('group', group))
                
                # Add processes if group is expanded
                if group.is_expanded:
                    for process in group.processes:
                        self.display_items.append(('process', process))
        else:
            # Flat view: show all processes
            for process in self.processes:
                self.display_items.append(('process', process))
    
    def _sort_processes(self):
        """Sort processes based on current sort criteria"""
        if not self.processes:
            return
            
        def sort_key(proc):
            if self.sort_by == 'pid':
                return int(proc.pid)
            elif self.sort_by == 'cpu':
                return float(proc.cpu)
            elif self.sort_by == 'mem':
                return float(proc.mem)
            elif self.sort_by == 'stat':
                return proc.stat
            elif self.sort_by == 'time':
                return proc.time
            elif self.sort_by == 'command':
                return proc.command.lower()
            else:
                return int(proc.pid)
        
        self.processes.sort(key=sort_key, reverse=self.sort_reverse)
    
    def refresh_processes(self, reset_marking=True):
        """Refresh the process list"""
        self.processes = self.get_python_processes()
        self._sort_processes()  # Sort after getting processes
        self._build_display_items()  # Build display items for current view
        if self.selected_index >= len(self.display_items):
            self.selected_index = max(0, len(self.display_items) - 1)
        if reset_marking:
            self.marked_for_kill = False  # Reset kill marking when explicitly refreshing
    
    def run_interactive(self):
        """Run the interactive terminal interface"""
        try:
            curses.wrapper(self._curses_main)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    def _curses_main(self, stdscr):
        """Main curses interface"""
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # Header
        curses.init_pair(3, curses.COLOR_RED, -1)    # Kill message
        curses.init_pair(4, curses.COLOR_YELLOW, -1) # Warning
        
        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Refresh process list (don't reset marking during normal loop)
            self.refresh_processes(reset_marking=False)
            
            # Header
            mode_text = "GLOBAL" if self.global_mode else "LOCAL"
            view_text = "GROUPED" if self.show_grouped else "FLAT"
            sort_indicator = "↓" if self.sort_reverse else "↑"
            header = f"Python Process Manager ({mode_text}/{view_text}) - {len(self.processes)} processes - Sort: {self.sort_by.upper()}{sort_indicator}"
            stdscr.addstr(0, 0, header, curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(1, 0, f"Directory: {self.current_dir}", curses.color_pair(2))
            if self.show_grouped:
                stdscr.addstr(2, 0, f"Groups: {len(self.process_groups)}", curses.color_pair(2))
            stdscr.addstr(2 if not self.show_grouped else 3, 0, "=" * min(width-1, 80), curses.color_pair(2))
            
            # Column headers
            headers_row = 3 if not self.show_grouped else 4
            headers = f"{'PID':<8} {'CPU%':<6} {'MEM%':<6} {'STAT':<6} {'TIME':<8} {'SCRIPT':<20} {'COMMAND'}"
            stdscr.addstr(headers_row, 0, headers, curses.A_BOLD)
            
            # Display items (groups and/or processes)
            start_row = headers_row + 1
            visible_rows = height - start_row - 3  # Leave space for footer
            
            if not self.display_items:
                stdscr.addstr(start_row, 0, "No Python processes found.", curses.color_pair(4))
            else:
                for i, (item_type, item) in enumerate(self.display_items[:visible_rows]):
                    row = start_row + i
                    if row >= height - 3:
                        break
                    
                    if item_type == 'group':
                        # Format group header
                        group = item
                        expand_char = "▼" if group.is_expanded else "▶"
                        group_info = f"{expand_char} {group.parent_script} ({group.get_process_count()} processes) - CPU: {group.get_total_cpu():.1f}% MEM: {group.get_total_mem():.1f}%"
                        
                        # Highlight selected group
                        if i == self.selected_index:
                            if self.marked_for_kill:
                                stdscr.addstr(row, 0, group_info[:width-1], curses.color_pair(3) | curses.A_BOLD)
                            else:
                                stdscr.addstr(row, 0, group_info[:width-1], curses.color_pair(1) | curses.A_BOLD)
                        else:
                            stdscr.addstr(row, 0, group_info[:width-1], curses.color_pair(2) | curses.A_BOLD)
                            
                    else:  # item_type == 'process'
                        # Format process info
                        proc = item
                        short_cmd = proc.get_short_command(width - 60)
                        indent = "  " if self.show_grouped else ""
                        line = f"{indent}{proc.pid:<6} {proc.cpu:<6} {proc.mem:<6} {proc.stat:<6} {proc.time:<8} {proc.script_name:<20} {short_cmd}"
                        
                        # Highlight selected process
                        if i == self.selected_index:
                            if self.marked_for_kill:
                                # Show selected process marked for killing in red
                                stdscr.addstr(row, 0, line[:width-1], curses.color_pair(3) | curses.A_BOLD)
                            else:
                                # Regular selection highlighting
                                stdscr.addstr(row, 0, line[:width-1], curses.color_pair(1))
                        else:
                            stdscr.addstr(row, 0, line[:width-1])
            
            # Footer with instructions
            footer_row = height - 2
            if self.marked_for_kill and self.display_items:
                item_type, item = self.display_items[self.selected_index] if self.selected_index < len(self.display_items) else (None, None)
                if item_type == 'group':
                    stdscr.addstr(footer_row, 0, f"READY TO KILL GROUP {item.parent_script} ({item.get_process_count()} processes) - Press ENTER to confirm | Any other key to cancel", curses.color_pair(3) | curses.A_BOLD)
                else:
                    stdscr.addstr(footer_row, 0, "READY TO KILL PROCESS - Press ENTER to confirm kill | Any other key to cancel", curses.color_pair(3) | curses.A_BOLD)
            else:
                footer1 = "Controls: ↑/↓ Navigate | k Mark | K Kill All | r Refresh | g Global | G Group View | q Quit"
                footer2 = "Sort: p(PID) c(CPU) m(MEM) s(STAT) t(TIME) P(PATH) | SPACE expand/collapse groups"
                stdscr.addstr(footer_row - 1, 0, footer2, curses.color_pair(4))
                stdscr.addstr(footer_row, 0, footer1, curses.A_BOLD)
            
            stdscr.refresh()
            
            # Handle input
            try:
                key = stdscr.getch()
                
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('r'):  # Refresh
                    self.refresh_processes(reset_marking=True)  # Explicit refresh resets marking
                    continue
                elif key == ord('g'):  # Toggle global mode
                    self.global_mode = not self.global_mode
                    self.selected_index = 0
                elif key == ord('G'):  # Toggle grouped view (Shift+G)
                    self.show_grouped = not self.show_grouped
                    self._build_display_items()
                    self.selected_index = 0
                elif key == ord('p'):  # Sort by PID
                    if self.sort_by == 'pid':
                        self.sort_reverse = not self.sort_reverse
                    else:
                        self.sort_by = 'pid'
                        self.sort_reverse = False
                    self.selected_index = 0
                elif key == ord('c'):  # Sort by CPU
                    if self.sort_by == 'cpu':
                        self.sort_reverse = not self.sort_reverse
                    else:
                        self.sort_by = 'cpu'
                        self.sort_reverse = True  # High CPU first by default
                    self.selected_index = 0
                elif key == ord('m'):  # Sort by memory
                    if self.sort_by == 'mem':
                        self.sort_reverse = not self.sort_reverse
                    else:
                        self.sort_by = 'mem'
                        self.sort_reverse = True  # High memory first by default
                    self.selected_index = 0
                elif key == ord('s'):  # Sort by state
                    if self.sort_by == 'stat':
                        self.sort_reverse = not self.sort_reverse
                    else:
                        self.sort_by = 'stat'
                        self.sort_reverse = False
                    self.selected_index = 0
                elif key == ord('t'):  # Sort by time
                    if self.sort_by == 'time':
                        self.sort_reverse = not self.sort_reverse
                    else:
                        self.sort_by = 'time'
                        self.sort_reverse = True  # Longest time first by default
                    self.selected_index = 0
                elif key == ord('P'):  # Sort by command/path (Shift+P to avoid conflict)
                    if self.sort_by == 'command':
                        self.sort_reverse = not self.sort_reverse
                    else:
                        self.sort_by = 'command'
                        self.sort_reverse = False
                    self.selected_index = 0
                elif key == curses.KEY_UP and self.selected_index > 0:
                    self.selected_index -= 1
                    self.marked_for_kill = False  # Reset marking when navigating
                elif key == curses.KEY_DOWN and self.selected_index < len(self.display_items) - 1:
                    self.selected_index += 1
                    self.marked_for_kill = False  # Reset marking when navigating
                elif key == ord(' ') and self.show_grouped and self.display_items:  # Space to expand/collapse groups
                    if 0 <= self.selected_index < len(self.display_items):
                        item_type, item = self.display_items[self.selected_index]
                        if item_type == 'group':
                            item.is_expanded = not item.is_expanded
                            self._build_display_items()
                            # Keep selection on the same group
                            if self.selected_index >= len(self.display_items):
                                self.selected_index = len(self.display_items) - 1
                elif key == ord('K') and self.processes:  # Shift+K for kill all
                    # Kill all processes
                    if self._confirm_kill_all(stdscr):
                        success_count, total_count = self.kill_all_processes()
                        if success_count == total_count:
                            self._show_message(stdscr, f"All {total_count} processes killed successfully", curses.color_pair(2))
                        else:
                            self._show_message(stdscr, f"Killed {success_count}/{total_count} processes", curses.color_pair(4))
                elif key == ord('k') and self.display_items:
                    # Mark selected item for killing
                    if 0 <= self.selected_index < len(self.display_items):
                        self.marked_for_kill = True
                elif key == ord('\n') and self.display_items and self.marked_for_kill:
                    # Confirm kill of marked item
                    if 0 <= self.selected_index < len(self.display_items):
                        item_type, item = self.display_items[self.selected_index]
                        if item_type == 'group':
                            # Kill all processes in group
                            success_count, total_count = self.kill_group(item.parent_script)
                            if success_count == total_count:
                                self._show_message(stdscr, f"All {total_count} processes in {item.parent_script} killed successfully", curses.color_pair(2))
                            else:
                                self._show_message(stdscr, f"Killed {success_count}/{total_count} processes in {item.parent_script}", curses.color_pair(4))
                        else:  # process
                            proc = item
                            success = self.kill_process(proc.pid)
                            if success:
                                self._show_message(stdscr, f"Process {proc.pid} killed successfully", curses.color_pair(2))
                            else:
                                self._show_message(stdscr, f"Failed to kill process {proc.pid}", curses.color_pair(3))
                        self.marked_for_kill = False
                elif self.marked_for_kill:
                    # Any other key cancels the kill marking
                    self.marked_for_kill = False
                            
            except KeyboardInterrupt:
                break
    
    def _confirm_kill(self, stdscr, proc: ProcessInfo) -> bool:
        """Show confirmation dialog for killing a process"""
        height, width = stdscr.getmaxyx()
        
        # Create confirmation window
        msg = f"Kill process {proc.pid} ({proc.script_name})? [y/N]: "
        stdscr.addstr(height - 1, 0, " " * (width - 1))
        stdscr.addstr(height - 1, 0, msg, curses.color_pair(3) | curses.A_BOLD)
        stdscr.refresh()
        
        key = stdscr.getch()
        return key == ord('y') or key == ord('Y')
    
    def _confirm_kill_all(self, stdscr) -> bool:
        """Show confirmation dialog for killing all processes"""
        height, width = stdscr.getmaxyx()
        
        # Create confirmation window
        mode_text = "LOCAL" if not self.global_mode else "GLOBAL"
        msg = f"Kill ALL {len(self.processes)} {mode_text} processes? Type 'YES' to confirm: "
        stdscr.addstr(height - 1, 0, " " * (width - 1))
        stdscr.addstr(height - 1, 0, msg, curses.color_pair(3) | curses.A_BOLD)
        stdscr.refresh()
        
        # Get user input for "YES"
        curses.echo()
        curses.curs_set(1)  # Show cursor
        input_str = ""
        
        try:
            while True:
                key = stdscr.getch()
                if key == ord('\n'):  # Enter
                    break
                elif key == 27:  # ESC
                    input_str = ""
                    break
                elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                    if input_str:
                        input_str = input_str[:-1]
                        # Redraw the line
                        stdscr.addstr(height - 1, 0, " " * (width - 1))
                        stdscr.addstr(height - 1, 0, msg + input_str, curses.color_pair(3) | curses.A_BOLD)
                        stdscr.refresh()
                elif 32 <= key <= 126:  # Printable characters
                    input_str += chr(key)
                    if len(input_str) <= 10:  # Limit input length
                        stdscr.addstr(height - 1, len(msg), input_str, curses.color_pair(3) | curses.A_BOLD)
                        stdscr.refresh()
        finally:
            curses.noecho()
            curses.curs_set(0)  # Hide cursor
        
        return input_str.upper() == "YES"
    
    def _show_message(self, stdscr, message: str, color_pair):
        """Show a temporary message"""
        height, width = stdscr.getmaxyx()
        stdscr.addstr(height - 1, 0, " " * (width - 1))
        stdscr.addstr(height - 1, 0, message, color_pair)
        stdscr.refresh()
        curses.napms(1500)  # Show for 1.5 seconds
    
    def list_processes_simple(self):
        """Simple list view for non-interactive mode"""
        processes = self.get_python_processes()
        
        if not processes:
            print("No Python processes found.")
            return
        
        mode_text = "GLOBAL" if self.global_mode else "LOCAL"
        print(f"\nPython Processes ({mode_text}) - {len(processes)} found:")
        print(f"Directory: {self.current_dir}")
        print("=" * 80)
        print(f"{'PID':<8} {'CPU%':<6} {'MEM%':<6} {'STAT':<6} {'TIME':<8} {'SCRIPT':<20} {'COMMAND'}")
        print("-" * 80)
        
        for proc in processes:
            short_cmd = proc.get_short_command(40)
            print(f"{proc.pid:<8} {proc.cpu:<6} {proc.mem:<6} {proc.stat:<6} {proc.time:<8} {proc.script_name:<20} {short_cmd}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Python processes with a clean terminal interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_manager.py              # Interactive mode (local processes)
  python process_manager.py -g           # Interactive mode (global processes)  
  python process_manager.py --list       # Simple list view (local)
  python process_manager.py -g --list    # Simple list view (global)
  
Interactive Controls:
  ↑/↓        Navigate through processes
  ENTER/k    Kill selected process
  r          Refresh process list
  g          Toggle between local/global mode
  q/ESC      Quit
        """
    )
    
    parser.add_argument('-g', '--global', action='store_true', dest='global_mode',
                       help='Show all Python processes by user (not just current directory)')
    parser.add_argument('--list', action='store_true',
                       help='Simple list view instead of interactive mode')
    parser.add_argument('--version', action='version', version='Process Manager 1.0')
    
    args = parser.parse_args()
    
    manager = ProcessManager(global_mode=args.global_mode)
    
    if args.list:
        manager.list_processes_simple()
    else:
        manager.run_interactive()


if __name__ == "__main__":
    main()