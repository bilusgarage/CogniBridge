#!/usr/bin/env python3
import os
import subprocess
import sys

def run_command(command, description):
    """Runs a terminal command and prints the status."""
    print(f"\n{'-'*50}")
    print(f"⏳ STEP: {description}")
    print(f"💻 RUNNING: {' '.join(command)}")
    print(f"{'-'*50}")
    
    try:
        # We use check=True so the script stops if a command fails
        subprocess.run(command, check=True)
        print(f"✅ SUCCESS: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to execute {description}.")
        print("Please check the terminal output above for clues.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ ERROR: Command not found. Make sure Conda is installed and added to your PATH.")
        sys.exit(1)

def main():
    print("🚀 Starting CogniBridge Installation Protocol...\n")
    
    # 1. Create the NLP Brain (cogni39)
    # The '-y' flag automatically says "yes" to the installation prompts
    run_command(
        ["conda", "create", "-n", "cogni39", "python=3.9.11", "-y"],
        "Creating 'cogni39' environment for Qwen Text Generation"
    )
    
    # 2. Install requirements into the NLP Brain
    # 'conda run -n' executes the pip command INSIDE the environment without needing to activate it!
    run_command(
        ["conda", "run", "-n", "cogni39", "pip", "install", "-r", "requirements_main.txt"],
        "Installing main dependencies into 'cogni39'"
    )
    
    # 3. Create the Vision Brain (mindocr_env)
    run_command(
        ["conda", "create", "-n", "mindocr_env", "python=3.9.11", "-y"],
        "Creating 'mindocr_env' environment for Optical Character Recognition"
    )
    
    # 4. Install requirements into the Vision Brain
    # Assuming you have a requirements_ocr.txt or you point this directly to mindocr/requirements.txt
    run_command(
        ["conda", "run", "-n", "mindocr_env", "pip", "install", "-r", "mindocr/requirements.txt"],
        "Installing OCR dependencies into 'mindocr_env'"
    )
    
    print("\n" + "="*50)
    print("🎉 INSTALLATION COMPLETE! 🎉")
    print("Your 'Two-Brain' architecture is fully configured.")
    print("To run the project, activate the main environment:")
    print("👉 conda activate cogni39")
    print("👉 python src/CogniBridge.py")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Ensure the script is being run from the root directory so it finds the requirements files
    if not os.path.exists("requirements_main.txt"):
        print("❌ ERROR: Could not find 'requirements_main.txt'.")
        print("Please run this script from the root of the CogniBridge repository.")
        sys.exit(1)
        
    main()