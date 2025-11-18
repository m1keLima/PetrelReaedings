# OVERVIEW
# This file is a utility script designed to create timestamped backups of all Python (.py) files 
# in its current directory. When executed, it copies each .py file—including itself—into a subfolder 
# named "Backup code", appending the current date and time to each filename. The script also 
# provides colored terminal output for better user feedback and attempts to clear the terminal at 
# the start for a cleaner display.

import os
import shutil
from datetime import datetime

# Font colors
DOMANDA = "\033[1;32m"  # Bold green
REGOLARE = "\033[0m" # Regular white
HIGHLIGHT = "\033[1;38;5;208m" #Bold orange
ERRORE = "\033[1;31m" #Bold red

# Set source and destination folders
src_folder = os.path.dirname(os.path.abspath(__file__))
dst_folder = os.path.join(src_folder, "Backup code")

# Create destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)

# Get current date and time in the format yyyymmdd.hhmm
timestamp = datetime.now().strftime("%Y%m%d-%H%M")

# Clear terminal
import contextlib
def clear_terminal():
    """Attempt to clear the terminal, but fail gracefully if not possible."""
    with contextlib.suppress(Exception):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
clear_terminal()  # Clear terminal (optional, fails gracefully)

# Copy all .py, .json, and .csv files including this backup script
for fname in os.listdir(src_folder):
    if fname.lower().endswith((".py", ".json", ".csv")):
        base, ext = os.path.splitext(fname)
        new_name = f"{base}_{timestamp}{ext}"
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, new_name)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {fname} -> {dst_path}")

print(f"{HIGHLIGHT}Backup complete.{REGOLARE}")