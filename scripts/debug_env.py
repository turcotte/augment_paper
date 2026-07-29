#!/usr/bin/env python3
"""
Diagnostic script to inspect the Python environment and path resolution 
when running inside a SLURM allocation on DRAC.
"""

import sys
import os
import platform

def print_section(title):
    print(f"\n{'-' * 80}")
    print(f" {title} ".center(80, '='))
    print(f"{'-' * 80}")

def main():
    print_section("SYSTEM INFORMATION")
    print(f"Hostname:       {platform.node()}")
    print(f"Python Version: {sys.version.replace(chr(10), ' ')}")
    print(f"Python Exec:    {sys.executable}")
    print(f"Current Dir:    {os.getcwd()}")
    
    print_section("PYTHON SYS.PATH")
    for i, p in enumerate(sys.path):
        print(f"[{i:02d}] {p}")

    print_section("RELEVANT ENVIRONMENT VARIABLES")
    keys_of_interest = [
        "PYTHONPATH", 
        "VIRTUAL_ENV", 
        "PATH", 
        "SLURM_JOB_ID", 
        "SLURM_SUBMIT_DIR",
        "SLURM_JOB_NODELIST",
        "PROJECT_DIR"
    ]
    
    for k in keys_of_interest:
        print(f"{k:<18}: {os.environ.get(k, '<NOT SET>')}")

    print_section("MODULE RESOLUTION TEST")
    try:
        import src
        print("[SUCCESS] Successfully imported 'src'")
        if hasattr(src, '__file__'):
            print(f"          Resolved path: {src.__file__}")
        elif hasattr(src, '__path__'):
            print(f"          Namespace path: {list(src.__path__)}")
        else:
            print("          (No __file__ or __path__ attribute found)")
    except ImportError as e:
        print(f"[FAILED]  Could not import 'src'")
        print(f"          Error: {e}")
        
    try:
        import src.data
        print("[SUCCESS] Successfully imported 'src.data'")
        if hasattr(src.data, '__file__'):
            print(f"          Resolved path: {src.data.__file__}")
    except ImportError as e:
        print(f"[FAILED]  Could not import 'src.data'")
        print(f"          Error: {e}")

if __name__ == "__main__":
    main()
