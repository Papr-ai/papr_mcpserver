#!/usr/bin/env python3
"""
Script to keep the two paprmcp.py files in sync.
This script copies the package version to the root version with import adjustments.
"""

import os
import shutil
from pathlib import Path

def sync_paprmcp_files():
    """Sync the package paprmcp.py to the root paprmcp.py with import adjustments."""
    
    # Paths
    package_file = Path("papr_memory_mcp/paprmcp.py")
    root_file = Path("paprmcp.py")
    
    if not package_file.exists():
        print(f"Error: Package file {package_file} not found!")
        return False
    
    # Read the package file
    with open(package_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the import to work in root context
    # Change from: from .services.logging_config import get_logger
    # To: from services.logging_config import get_logger
    content = content.replace(
        "from .services.logging_config import get_logger",
        "from services.logging_config import get_logger"
    )
    
    # Write to root file
    with open(root_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Synced {package_file} → {root_file}")
    print("   - Updated import: .services.logging_config → services.logging_config")
    return True

def sync_root_to_package():
    """Sync the root paprmcp.py to the package version with import adjustments."""
    
    # Paths
    root_file = Path("paprmcp.py")
    package_file = Path("papr_memory_mcp/paprmcp.py")
    
    if not root_file.exists():
        print(f"Error: Root file {root_file} not found!")
        return False
    
    # Read the root file
    with open(root_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the import to work in package context
    # Change from: from services.logging_config import get_logger
    # To: from .services.logging_config import get_logger
    content = content.replace(
        "from services.logging_config import get_logger",
        "from .services.logging_config import get_logger"
    )
    
    # Write to package file
    with open(package_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Synced {root_file} → {package_file}")
    print("   - Updated import: services.logging_config → .services.logging_config")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "reverse":
        # Sync root to package
        sync_root_to_package()
    else:
        # Default: sync package to root
        sync_paprmcp_files()
    
    print("\n💡 Usage:")
    print("   python sync_files.py          # Sync package → root")
    print("   python sync_files.py reverse  # Sync root → package")
