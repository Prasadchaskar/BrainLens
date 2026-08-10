import os
from pathlib import Path

# Define the directory structure relative to the current directory
structure = [
    # Main service folders
    "services/data-download-service",
    "services/data-validation-service",
    "services/preprocessing-service",
    "services/training-service",
    "services/evaluation-service",
    "services/model-registry-service",
    "services/prediction-service",
    "services/drift-monitor-service",
    
    # Root level folders
    "airflow",
    "infrastructure",
    "shared"
]

# Define root level files to create
files = [
    "README.md"
]

def create_boilerplate():
    print("🚀 Starting folder structure creation inside BrainLens...")
    
    # Create directories
    for folder in structure:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder}")
        
    # Create blank files
    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            file_path.touch()
            print(f"Created file: {file}")
        else:
            print(f"File already exists, skipping: {file}")
            
    print("\n✅ Setup complete! Your MLOps pipeline structure is ready.")

if __name__ == "__main__":
    create_boilerplate()
