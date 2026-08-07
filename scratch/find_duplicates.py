import os
from collections import defaultdict

def scan_files():
    workspace = r"d:\BrainTumorProject\UNeXt-pytorch"
    py_files = defaultdict(list)
    css_files = defaultdict(list)
    
    for root, dirs, files in os.walk(workspace):
        # Exclude virtual environment and git
        if ".venv" in root or ".git" in root or ".gemini" in root or "build" in root or "dist" in root:
            continue
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, workspace)
            if file.endswith(".py"):
                py_files[file].append(rel_path)
            elif file.endswith(".css"):
                css_files[file].append(rel_path)
                
    print("--- DUPLICATE PYTHON MODULES / FILE NAMES ---")
    duplicate_py = {k: v for k, v in py_files.items() if len(v) > 1}
    if not duplicate_py:
        print("None found!")
    else:
        for name, paths in duplicate_py.items():
            print(f"{name}:")
            for p in paths:
                print(f"  - {p}")
                
    print("\n--- ALL CSS FILES ---")
    for name, paths in css_files.items():
        print(f"{name}: {paths}")

if __name__ == "__main__":
    scan_files()
