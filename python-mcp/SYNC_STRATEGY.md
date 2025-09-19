# Code Synchronization Strategy

## Problem
You have two `paprmcp.py` files that need to stay in sync:
- `paprmcp.py` (root) - for direct execution
- `papr_memory_mcp/paprmcp.py` (package) - for PyPI installation

## Solution: Single Source of Truth

### 1. **Shared Core Module** ✅ IMPLEMENTED
- **File**: `papr_memory_mcp/core.py`
- **Contains**: All the main logic, classes, and functions
- **Smart imports**: Handles both relative and absolute imports automatically

### 2. **Thin Wrapper Files** ✅ IMPLEMENTED
- **Root file** (`paprmcp.py`): Imports from `papr_memory_mcp.core`
- **Package file** (`papr_memory_mcp/paprmcp.py`): Imports from `.core`

### 3. **Sync Script** ✅ IMPLEMENTED
- **File**: `sync_files.py`
- **Usage**: 
  ```bash
  python sync_files.py          # Sync package → root
  python sync_files.py reverse  # Sync root → package
  ```

## Benefits

✅ **Single source of truth**: All logic in `core.py`
✅ **No duplication**: Both files are just thin wrappers
✅ **Automatic sync**: Changes to core.py affect both files
✅ **Import flexibility**: Core handles both relative and absolute imports
✅ **Maintainable**: Only one file to edit for features

## Workflow

### For Development:
1. **Edit**: `papr_memory_mcp/core.py` (the main file)
2. **Test**: Both files automatically use the updated core
3. **No sync needed**: Changes are automatically available

### For Manual Sync (if needed):
```bash
# If you ever need to sync manually
python sync_files.py
```

## File Structure

```
papr_memory_mcp/
├── core.py              # 🎯 MAIN LOGIC (edit this file)
├── paprmcp.py          # 📦 Package wrapper (auto-synced)
└── services/
    └── logging_config.py

paprmcp.py              # 🏠 Root wrapper (auto-synced)
sync_files.py           # 🔄 Sync script (backup method)
```

## Import Strategy in Core

The `core.py` file uses smart imports that work in both contexts:

```python
try:
    from .services.logging_config import get_logger
except ImportError:
    try:
        from services.logging_config import get_logger
    except ImportError:
        # Fallback to basic logging
        def get_logger(name):
            # ... basic logging setup
```

This handles:
- ✅ Package context: `from .services.logging_config`
- ✅ Root context: `from services.logging_config`
- ✅ Fallback: Basic logging if neither works

## Best Practices

1. **Always edit**: `papr_memory_mcp/core.py`
2. **Never edit**: The wrapper files directly
3. **Test both**: Root and package versions
4. **Use sync script**: Only if you need to manually sync

## Verification

To verify everything works:

```bash
# Test root version
python paprmcp.py

# Test package version
python -m papr_memory_mcp.paprmcp

# Test installed version (after pip install)
papr-mcp
```

All three should work identically! 🎉
