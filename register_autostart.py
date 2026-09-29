"""
Registers JARVIS into Windows Startup folder so it runs completely in the
background on Windows boot without ever needing to open or click any file.
Also starts the background service immediately.
"""

import os
import sys
import subprocess

STARTUP_DIR = os.path.join(
    os.environ["APPDATA"],
    r"Microsoft\Windows\Start Menu\Programs\Startup"
)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(CURRENT_DIR, "jarvis_ui.pyw")
PYTHONW_PATH = sys.executable.replace("python.exe", "pythonw.exe")

if not os.path.exists(PYTHONW_PATH):
    # Fallback to local python program path
    PYTHONW_PATH = r"C:\Users\Avinash Reddy\AppData\Local\Programs\Python\Python313\pythonw.exe"

VBS_CONTENT = f'''Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "{CURRENT_DIR}"
WshShell.Run """{PYTHONW_PATH}"" ""{SCRIPT_PATH}""", 0, False
'''

vbs_file_path = os.path.join(STARTUP_DIR, "JARVIS_Assistant.vbs")

with open(vbs_file_path, "w", encoding="utf-8") as f:
    f.write(VBS_CONTENT)

print(f"[SUCCESS] Registered JARVIS to Windows Startup at:\n{vbs_file_path}")

# Launch the VBS script now so JARVIS is running in the background right now
subprocess.Popen(["wscript.exe", vbs_file_path], cwd=CURRENT_DIR)
print("[SUCCESS] JARVIS is now active in the background. Say 'Hey Jarvis' to summon.")
