"""
Simple launcher script for the Student Score Prediction Dashboard.
This script will automatically train the model if needed and launch the dashboard.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_model_exists():
    """Check if the trained model exists."""
    model_path = Path('models/student_score_model.pkl')
    return model_path.exists()

def train_model_if_needed():
    """Train the model if it doesn't exist."""
    if not check_model_exists():
        print("🔄 Model not found. Training model first...")
        try:
            result = subprocess.run([sys.executable, 'run_pipeline.py'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Error training model:")
                print(result.stderr)
                return False
            print("✅ Model trained successfully!")
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    else:
        print("✅ Model found!")
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("🚀 Launching dashboard...")
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function."""
    print("🎓 Student Score Prediction Dashboard Launcher")
    print("=" * 50)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Check and train model if needed
    if not train_model_if_needed():
        print("❌ Failed to prepare model. Exiting.")
        sys.exit(1)
    
    # Launch dashboard
    print("\n🌐 Starting web dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    launch_dashboard()

if __name__ == "__main__":
    main()