import os
import sys
import subprocess

VENV_DIR = ".venv"
REQUIREMENTS = "requirements.txt"
TARGET_SCRIPT = "main.py"

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def get_python_exec():
    return os.path.join(VENV_DIR, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV_DIR, "bin", "python")

def is_venv_ready():
    if not os.path.exists(VENV_DIR):
        return False

    python_exec = get_python_exec()

    if os.path.exists(REQUIREMENTS):
        try:
            result = subprocess.run(
                [python_exec, "-m", "pip", "check"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    return True

def ensure_venv():
    if not os.path.exists(VENV_DIR):
        print("Stvaranje virtualne okoline...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

    python_exec = get_python_exec()

    if os.path.exists(REQUIREMENTS):
        print("Instalacija ovisnosti...")
        subprocess.run([python_exec, "-m", "pip", "install", "-r", REQUIREMENTS], check=True)

def run_script():
    clear_screen()
    python_exec = get_python_exec()
    subprocess.run([python_exec, TARGET_SCRIPT], check=True)

if __name__ == "__main__":
    if is_venv_ready():
        run_script()
    else:      
        ensure_venv()
        run_script()
