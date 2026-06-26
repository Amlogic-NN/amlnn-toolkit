#!/usr/bin/env python3

#
# Copyright (C) 2024–2025 Amlogic, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import subprocess
import re
try:
    import curses
except Exception:
    curses = None
import time
import traceback
import os
import platform
import sys

__version__ = "1.2"

logo = [
    r"                   __            _     ",
    r"  ____ _____ ___  / /___  ____ _(_)____",
    r" / __ `/ __ `__ \/ / __ \/ __ `/ / ___/",
    r"/ /_/ / / / / / / / /_/ / /_/ / / /__  ",
    r"\__,_/_/ /_/ /_/_/\____/\__, /_/\___/  ",
    r"                       /____/          "
]

SOC_OTHERS = "/sys/class/adla/adla0/device/debug/utilization"
SOC_C302 = "/proc/mbp/adla_utilization"
NN_LOADING_OTHERS = "/sys/class/adla/adla0/device/debug/nn_loading"
NN_LOADING_C302 = "/proc/mbp/adla_nn_loading"    # to do, not support for C302.

# Detection System Architecture
def detect_architecture():
    """Detect system architecture and return True if arm64, False if x86."""
    machine = platform.machine().lower()
    return machine in ['aarch64', 'arm64', 'armv8l']

# Global variable: Whether it is an arm64 architecture
IS_ARM64 = detect_architecture()

# CPU statistics cache, used to avoid waiting for each call
_cpu_stats_cache = None
_cpu_stats_cache_time = 0

def execute_shell(cmd: str):
    """Execute commands based on architecture: arm64 executes directly, x86 via adb."""
    if IS_ARM64:
        # arm64
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    else:
        # Execute via ADB on x86
        p = subprocess.run(
            ["adb", "shell", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    out = (p.stdout or b"").decode(errors="ignore").strip()
    err = (p.stderr or b"").decode(errors="ignore").strip()
    return p.returncode, out, err

def path_exists(path: str) -> bool:
    """Check whether the path exists based on the architecture."""
    _, out, _ = execute_shell(f'[ -e "{path}" ] && echo YES || echo NO')
    return out.strip().endswith("YES")

def is_android_or_yocto_device():
    return path_exists(SOC_OTHERS)

def enable_npu_utilization():
    target = SOC_OTHERS if path_exists(SOC_OTHERS) else SOC_C302
    rc, _, err = execute_shell(f'echo 1 > "{target}"')
    if rc != 0 or "Permission denied" in err:
        execute_shell(f'su -c \'echo 1 > "{target}"\'')

def get_nn_loading_node():
    """Get the nn_loading node path based on platform"""
    if path_exists(NN_LOADING_OTHERS):
        return NN_LOADING_OTHERS
    elif path_exists(NN_LOADING_C302):
        return NN_LOADING_C302
    else:
        return None

def enable_nn_loading():
    """Enable nn_loading recording"""
    node = get_nn_loading_node()
    if node:
        rc, _, err = execute_shell(f'echo 1 > "{node}"')
        if rc != 0 or "Permission denied" in err:
            execute_shell(f'su -c \'echo 1 > "{node}"\'')

def disable_nn_loading():
    """Disable nn_loading recording"""
    node = get_nn_loading_node()
    if node:
        rc, _, err = execute_shell(f'echo 0 > "{node}"')
        if rc != 0 or "Permission denied" in err:
            execute_shell(f'su -c \'echo 0 > "{node}"\'')

def get_nn_loading():
    """Get nn_loading percentage (0-100).

    output format:
    - "72" / "72 %" / "72%"
    - "NPU0 50%, NPU1 60%" (multiple values output by one cat command)
    """
    node = get_nn_loading_node()
    if not node:
        return None
    
    _, out, err = _read_node(node)
    
    if err and "No such file" in err:
        return None
    
    # clean output, remove leading and trailing whitespace
    out = out.strip()
    
    # first parse multiple values with NPU label: NPU0 50%, NPU1 60%
    matches = re.findall(r"NPU\s*(\d+)\s*[:=]?\s*(\d+)\s*%?", out, re.I)
    if matches:
        pairs = [(int(i), int(v)) for i, v in matches]
        pairs.sort(key=lambda x: x[0])
        return [max(0, min(100, v)) for _, v in pairs]

    # then fallback to single value: match "90 %" or "90%" or "90"
    m = re.search(r"(\d+)\s*%?", out)
    if not m:
        return None
    try:
        value = int(m.group(1))
        return max(0, min(100, value))
    except ValueError:
        return None

def _read_node(path: str):
    return execute_shell(f'cat "{path}"')

def get_npu_utilization():
    node = SOC_OTHERS if path_exists(SOC_OTHERS) else SOC_C302
    _, out, err = _read_node(node)

    if 'please' in out.lower():
        enable_npu_utilization()
        _, out, err = _read_node(node)

    if err and "No such file" in err:
        return [0]

    m = re.search(r'(?:NPU\s*load|adla\s*utilization)?\s*:?\s*(-?\d+)\s*%', out, re.I)
    if not m:
        m = re.search(r'(-?\d+)', out)
    if m:
        v = int(m.group(1))
        return [0] if v < 0 else [v]

    return [0]

# get CPU loading
def read_cpu_times():
    """Read CPU time statistics, returning (idle_time, total_time) for each CPU."""
    if IS_ARM64:
        # arm64 directly reads local files
        with open('/proc/stat', 'r') as f:
            lines = f.read().splitlines()
    else:
        # Reading via ADB on x86
        result = subprocess.run(['adb', 'shell', 'cat', '/proc/stat'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return []
        lines = result.stdout.decode().splitlines()

    cpu_stats = []
    for line in lines:
        parts = line.strip().split()
        if not parts or not parts[0].startswith("cpu"):
            continue
        
        # Skip the “cpu” total row and process only cpu0, cpu1, cpu2...
        if parts[0] == "cpu":
            continue
            
        # /proc/stat: cpu0 user nice system idle iowait irq softirq steal guest guest_nice
        # At least 5 fields are required.(user, nice, system, idle, iowait)
        if len(parts) < 5:
            continue
            
        try:
            values = list(map(int, parts[1:]))
            # idle is the 4th field (index 3), iowait is the 5th field (index 4)
            # If the field is missing, use the default value 0.
            #
            # Align with `top -h` expectation: treat `iowait` as "busy" instead
            # of folding it into idle time.
            idle = values[3] if len(values) > 3 else 0
            # iowait intentionally not added into idle_time.
            idle_time = idle
            total_time = sum(values)
            cpu_stats.append((idle_time, total_time))
        except (ValueError, IndexError):
            continue

    return cpu_stats

def get_cpu_usage():
    """Calculate CPU usage, utilizing a caching mechanism to avoid waiting each time and improve refresh speed."""
    global _cpu_stats_cache, _cpu_stats_cache_time
    
    current_time = time.time()
    cpu_stats1 = None
    
    # If the cache exists and more than 0.5 seconds have passed since the last sample, use the cache.
    if _cpu_stats_cache is not None and (current_time - _cpu_stats_cache_time) >= 0.5:
        cpu_stats1 = _cpu_stats_cache
    else:
        # First call or cache expiration requires resampling.
        cpu_stats1 = read_cpu_times()
        if not cpu_stats1:
            return []
        # Wait 0.5 seconds for the second sampling (balancing accuracy and response speed)
        time.sleep(0.5)
    
    cpu_stats2 = read_cpu_times()
    if not cpu_stats2 or len(cpu_stats2) != len(cpu_stats1):
        return []

    # Update Cache
    _cpu_stats_cache = cpu_stats2
    _cpu_stats_cache_time = time.time()

    usage = []
    for i in range(len(cpu_stats1)):
        idle1, total1 = cpu_stats1[i]
        idle2, total2 = cpu_stats2[i]
        
        idle_delta = idle2 - idle1
        total_delta = total2 - total1
        
        if total_delta <= 0:
            usage.append(0.0)
        else:
            # CPU usage = (1 - idle_time / total_time) * 100
            usage_percent = 100.0 * (1.0 - idle_delta / total_delta)
            # Limited to the range of 0-100%
            usage_percent = max(0.0, min(100.0, usage_percent))
            usage.append(round(usage_percent, 1))

    return usage

def get_npudriver_version():
    if path_exists(SOC_OTHERS):
        version_node = "/sys/class/adla/adla0/device/kmd_version"
    else:
        version_node = "/proc/mbp/adla_version"

    rc, out, err = execute_shell(f'cat "{version_node}"')
    if rc != 0 or not out:
        return f"Failed to read version: {err}"

    # supports both regular and debug versions
    match = re.search(r'ADLA\s*Version\s*:\s*([\d\.]+(?:-[\w]+)?)', out, re.I)
    if match:
        return match.group(1)
    return None

def get_memory_usage():
    """Obtain memory usage with a more accurate calculation method"""
    if IS_ARM64:
        # arm64 directly reads local files
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_info = f.read()
        except Exception as e:
            return None
    else:
        # Reading via ADB on x86
        result = subprocess.run(['adb', 'shell', 'cat', '/proc/meminfo'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return None
        mem_info = result.stdout.decode()

    mem_total = None
    mem_free = None
    mem_available = None
    buffers = None
    cached = None
    
    for line in mem_info.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
            
        key = parts[0].rstrip(':')
        value = int(parts[1])
        
        if key == "MemTotal":
            mem_total = value
        elif key == "MemFree":
            mem_free = value
        elif key == "MemAvailable":
            mem_available = value
        elif key == "Buffers":
            buffers = value
        elif key == "Cached":
            cached = value

    if mem_total is None or mem_total == 0:
        return None
    
    # Using the same calculation method as htop: MemTotal - MemFree - Buffers - Cached
    if mem_free is not None:
        # Calculate the actual memory used
        if buffers is not None and cached is not None:
            # Used Memory = Total Memory - Free Memory - Puffer - Cache
            mem_used = mem_total - mem_free - buffers - cached
        else:
            # If Buffers and Cached information are unavailable, use MemFree for calculation.
            mem_used = mem_total - mem_free
        
        # Calculate utilization rate
        memory_usage_percent = 100.0 * mem_used / mem_total
    elif mem_available is not None:
        # Alternative: If MemFree is unavailable, use MemAvailable.
        memory_usage_percent = 100.0 * (mem_total - mem_available) / mem_total
    else:
        return None
    
    # Limited to the range of 0-100%
    memory_usage_percent = max(0.0, min(100.0, memory_usage_percent))
    return round(memory_usage_percent, 1)

# update
def matplotShow(frame):
    # get NPU utilization
    npu_utilization = get_npu_utilization()
    for i, bar in enumerate(bar_npu):
        if npu_utilization[i] > 50:
            bar.set_color('red')
        else:
            bar.set_color('green')
        bar.set_height(npu_utilization[i])

    # get CPU loading
    cpu_load = get_cpu_usage()
    for i, bar in enumerate(bar_cpu):
        if cpu_load[i] > 50:
            bar.set_color('red')
        else:
            bar.set_color('green')
        bar.set_height(cpu_load[i])

    return bar_npu + bar_cpu

# display
def draw_bar(win, y, x, value, label, max_width=40):
    bar_length = int((value / 100) * max_width)
    color = 1 if value < 50 else 2
    win.addstr(y, x, label + ": " + f"{value}% | ".rjust(9))
    win.addstr(y, x + len(label) + 11, "|" * bar_length, curses.color_pair(color))
    win.addstr(y, x + len(label) + 11 + bar_length, " " * (max_width - bar_length))
    win.addstr(y, x + len(label) + 11 + max_width + 1, "|")

def draw_bar_vertical(win, flag, y, x, value, colorid, ch):
    if flag == 0 :
        for i in range(value):
            if colorid != 0 :
                win.addstr(y + i, x, ch, curses.color_pair(colorid))
            else :
                win.addstr(y + i, x, ch)
    elif flag == 1:
        for i in range(value):
            if colorid != 0 :
                win.addstr(y - i, x, ch, curses.color_pair(colorid))
            else :
                win.addstr(y - i, x, ch)

def draw_logo(stdscr):
    max_y, max_x = stdscr.getmaxyx()  # Get window size
    for i, line in enumerate(logo):
        line_len = len(line)
        # Calculate the starting position of each row to center it.
        start_x = (max_x - line_len) // 2
        stdscr.addstr(i, start_x, line)  # Center each line in the window

def draw_usage_bar(stdscr, y, x, label, value, width=40):
    # Handle None / non-numeric to avoid TypeError
    if value is None:
        value = 0.0
    try:
        value = float(value)
    except Exception:
        value = 0.0
    value = max(0.0, min(100.0, value))

    bar_len = int(value / 100.0 * width)
    bar_start = "["
    bar_end = f"] {value:>5.1f}%"
    empty_part = " " * (width - bar_len)

    # Color matching
    color_pair = curses.color_pair(1) if value <= 50 else curses.color_pair(2)

    stdscr.addstr(y, x, f"{label:<8} ")
    stdscr.addstr(y, x + 9, bar_start)
    stdscr.addstr(y, x + 10, "|" * bar_len, color_pair)  # colored bar
    stdscr.addstr(y, x + 10 + bar_len, empty_part + bar_end)

def terminalShow(stdscr):
    curses.curs_set(0)  # Do not display the cursor
    stdscr.nodelay(1)   # No input required
    stdscr.timeout(50)  # 50ms timeout, matching actual ADB sampling rate

    # Initialize color
    curses.start_color() 
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Load below 50% uses green
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Load exceeding 50% indicates red
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  

    # Enable nn_loading recording when starting
    enable_nn_loading()

    try:
        while True:
            height, width = stdscr.getmaxyx()

            npu_utilization = get_npu_utilization()
            cpu_load = get_cpu_usage()
            memory_usage_percent = get_memory_usage()
            nn_loading_percent = get_nn_loading()
            npu_driver_version = "AMLNPU Driver: v" + get_npudriver_version()
            tool_version_line = f"AMLNPU Utilization Monitor v{__version__}"

            stdscr.clear()
            max_y, max_x = stdscr.getmaxyx()

            # === Draw logo ===
            logo_y = 1
            draw_logo(stdscr)
            logo_height = len(logo)

            # === Draw Driver Info ===
            stdscr.addstr(logo_height + logo_y, 0, "-" * max_x)
            stdscr.addstr(logo_height + logo_y + 1, max_x // 2 - len(tool_version_line) // 2, tool_version_line)
            stdscr.addstr(logo_height + logo_y + 2, max_x // 2 - len(npu_driver_version) // 2, npu_driver_version)
            stdscr.addstr(logo_height + logo_y + 3, 0, "-" * max_x)

            # === Draw NPU utilizations ===
            y = logo_height + logo_y + 5
            stdscr.addstr(y, 2, "[NPU Utilization]")
            y += 1
            for i, utilization in enumerate(npu_utilization):
                draw_usage_bar(stdscr, y, 4, f"NPU{i}", utilization)
                y += 1

            # === Draw NPU Loading ===
            if nn_loading_percent is not None:
                y += 1
                stdscr.addstr(y, 2, "[NPU Loading]")
                y += 1
                if isinstance(nn_loading_percent, list):
                    for i, v in enumerate(nn_loading_percent):
                        draw_usage_bar(stdscr, y, 4, f"NPU{i}", v)
                        y += 1
                else:
                    draw_usage_bar(stdscr, y, 4, "NPU", nn_loading_percent)
                    y += 1

            # === Draw CPU Loads (Display up to 8 cores) ===
            if cpu_load and len(cpu_load) > 0:
                y += 1
                stdscr.addstr(y, 2, "[CPU Loading]")
                y += 1
                for i, load in enumerate(cpu_load[:8]):
                    draw_usage_bar(stdscr, y, 4, f"CPU{i+1}", load)
                    y += 1

            # === Draw Memory Usage ===
            y += 1
            stdscr.addstr(y, 2, "[Memory Usage]")
            y += 1
            draw_usage_bar(stdscr, y, 4, "Memory", memory_usage_percent)

            # After all drawings are complete, add an exit prompt.
            exit_hint = "hold 'q' to exit"
            exit_y = y + 3  # Two rows from the bottom
            exit_x = (max_x - len(exit_hint)) // 2

            # Display a prompt at the specified location
            stdscr.addstr(exit_y, exit_x, exit_hint, curses.color_pair(3))

            # Refresh the screen
            stdscr.refresh()

            # Check if the ‘q’ key is pressed to exit - Detect before entering sleep mode to avoid missing the key press
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):  # Supports case-sensitive search
                break

            time.sleep(0.05)   # 50ms sampling interval, matching the actual ADB communication speed
            
            # After sleep, perform another check to enhance responsiveness.
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                break
    finally:
        # Disable nn_loading recording when exiting
        disable_nn_loading()

def textShow():
    """Text mode display, used when curses is unavailable"""
    print(f"AMLNPU Utilization Monitor v{__version__} (Text Mode)")
    print("=" * 50)
    print("Press Ctrl+C to exit\n")
    
    # Enable nn_loading recording when starting
    enable_nn_loading()
    
    try:
        while True:
            npu_utilization = get_npu_utilization()
            cpu_load = get_cpu_usage()
            memory_usage_percent = get_memory_usage()
            nn_loading_percent = get_nn_loading()
            npu_driver_version = get_npudriver_version()
            
            # Clear Screen (Using ANSI Escape Sequences)
            if os.name == "nt":
                os.system("cls")
            else:
                print("\033[2J\033[H", end="")
            
            # Display Logo
            for line in logo:
                print(line)
            print()
            
            # Display NPU Driver Version
            print("-" * 50)
            print(f"AMLNPU Driver: v{npu_driver_version}")
            print(f"AMLNPU Utilization Monitor v{__version__}")
            print("-" * 50)
            print()
            
            # Display NPU Utilization
            print("[NPU Utilization]")
            for i, utilization in enumerate(npu_utilization):
                bar = "|" * int(utilization / 2)  # Simplified Progress Bar
                print(f"NPU{i:>4}: [{bar:<50}] {utilization:>5.1f}%")
            print()
            
            # Display NPU Loading
            if nn_loading_percent is not None:
                print("[NPU Loading]")
                if isinstance(nn_loading_percent, list):
                    for i, v in enumerate(nn_loading_percent):
                        bar = "|" * int(v / 2)
                        print(f"NPU{i:>4}: [{bar:<50}] {float(v):>5.1f}%")
                    print()
                else:
                    bar = "|" * int(nn_loading_percent / 2)
                    print(f"NPU   0: [{bar:<50}] {nn_loading_percent:>5.1f}%")
                    print()
            
            # Display CPU load
            if cpu_load and len(cpu_load) > 0:
                print("[CPU Loading]")
                for i, load in enumerate(cpu_load[:8]):
                    bar = "|" * int(load / 2)
                    print(f"CPU{i+1:>4}: [{bar:<50}] {load:>5.1f}%")
                print()
            
            # Display memory usage
            print("[Memory Usage]")
            if memory_usage_percent is not None:
                bar = "|" * int(memory_usage_percent / 2)
                print(f"Memory: [{bar:<50}] {memory_usage_percent:>5.1f}%")
            else:
                print("Memory: [Failed to read memory info]")
            
            time.sleep(0.5)  # Refreshes every 0.5 seconds
            
    except KeyboardInterrupt:
        print("\n\nExiting...")

if __name__ == "__main__":
    # Windows: default to text mode unless curses is available.
    if os.name == "nt":
        if curses is None:
            textShow()
        else:
            try:
                curses.wrapper(terminalShow)
            except Exception:
                textShow()
    else:
        # Verify that you are in a valid terminal environment.
        term = os.environ.get("TERM", "")
        is_tty = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, "fileno") else False

        # If TERM is not set or is not a valid terminal, use text mode.
        if curses is None or not term or term == "unknown" or not is_tty:
            print("Warning: Terminal not detected, using text mode.")
            print("To use curses mode, ensure TERM environment variable is set correctly.")
            print()
            textShow()
        else:
            try:
                curses.wrapper(terminalShow)
            except Exception:
                print("Error initializing curses, falling back to text mode...")
                print()
                textShow()
