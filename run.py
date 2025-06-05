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

def parse_requirements():
    if not os.path.exists(REQUIREMENTS):
        return set()
    
    required_packages = set()
    with open(REQUIREMENTS, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pkg_name = line.split('==')[0].split('>')[0].split('<')[0].split('~')[0].split('[')[0].strip()
            required_packages.add(pkg_name.lower())
    return required_packages

def get_installed_packages(python_exec):
    try:
        result = subprocess.run(
            [python_exec, "-m", "pip", "freeze"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        installed_packages = set()
        for line in result.stdout.splitlines():
            if '==' in line:
                pkg_name = line.split('==')[0].lower()
                installed_packages.add(pkg_name)
        return installed_packages
    except subprocess.CalledProcessError:
        return set()

def is_venv_ready():
    if not os.path.exists(VENV_DIR):
        return False

    python_exec = get_python_exec()
    if not os.path.exists(python_exec):
        return False

    required_packages = parse_requirements()
    if not required_packages:
        return True

    try:
        installed_packages = get_installed_packages(python_exec)
        missing_packages = required_packages - installed_packages
        if missing_packages:
            print(f"Nedostaju ovisnosti: {', '.join(missing_packages)}")
        return len(missing_packages) == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

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