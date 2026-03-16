import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main function to run the application"""
    print("🏥 dAIgnoQ: Quantum-AI Medical Diagnosis")
    print("=" * 40)
    
    # Base directory of the project
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Path to the main application file
    APP_FILE = BASE_DIR / "dAIgnoQ" / "app" / "main.py"
    
    if not APP_FILE.exists():
        print(f"❌ {APP_FILE} not found!")
        sys.exit(1)
    
    print("🚀 Starting Streamlit application...")
    
    try:
        # Run Streamlit application
        # Setting PYTHONPATH to BASE_DIR so that dAIgnoQ can be imported
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(APP_FILE)
        ], env=env)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
