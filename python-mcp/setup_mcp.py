#!/usr/bin/env python3
"""
Setup script for Papr Memory MCP in various applications
This script can be run after package installation to automatically configure MCP clients
"""

import os
import json
import sys
import subprocess
from pathlib import Path
import platform

def get_application_config_path(app_name):
    """Get the MCP configuration file path for different applications"""
    if app_name.lower() == "cursor":
        if sys.platform == "win32":
            return Path.home() / ".cursor" / "mcp.json"
        elif sys.platform == "darwin":
            # Use ~/.cursor/mcp.json for macOS
            return Path.home() / ".cursor" / "mcp.json"
        else:
            config_dir = Path.home() / ".config" / "cursor" / "mcp"
            return config_dir / "mcp.json"
    elif app_name.lower() == "claude":
        if sys.platform == "win32":
            return Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        elif sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        else:
            return Path.home() / ".config" / "claude" / "claude_desktop_config.json"
    else:
        # Generic path for other applications
        return Path.home() / f".{app_name.lower()}" / "mcp.json"

def check_application_installed(app_name):
    """Check if an application is installed"""
    try:
        if app_name.lower() == "cursor":
            if sys.platform == "win32":
                cursor_paths = [
                    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "cursor",
                    Path(os.environ.get("PROGRAMFILES", "")) / "Cursor",
                    Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Cursor"
                ]
                return any(path.exists() for path in cursor_paths)
            elif sys.platform == "darwin":
                return Path("/Applications/Cursor.app").exists()
            else:
                result = subprocess.run(["which", "cursor"], capture_output=True)
                return result.returncode == 0
        elif app_name.lower() == "claude":
            if sys.platform == "win32":
                # Check multiple possible Claude installation paths
                claude_paths = [
                    # Standard installation paths
                    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Claude",
                    Path(os.environ.get("PROGRAMFILES", "")) / "Claude",
                    Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Claude",
                    # Alternative names (cluely, etc.)
                    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "cluely",
                    Path(os.environ.get("PROGRAMFILES", "")) / "cluely",
                    Path(os.environ.get("PROGRAMFILES(X86)", "")) / "cluely",
                    # Check for claude.exe in PATH
                ]
                # Check if any of the paths exist
                if any(path.exists() for path in claude_paths):
                    return True
                # Also check if claude or cluely executable is in PATH
                try:
                    result = subprocess.run(["where", "claude"], capture_output=True, shell=True)
                    if result.returncode == 0:
                        return True
                    result = subprocess.run(["where", "cluely"], capture_output=True, shell=True)
                    return result.returncode == 0
                except:
                    return False
            elif sys.platform == "darwin":
                return Path("/Applications/Claude.app").exists()
            else:
                result = subprocess.run(["which", "claude"], capture_output=True)
                return result.returncode == 0
        else:
            # For other applications, try to find them
            if sys.platform == "win32":
                app_paths = [
                    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / app_name,
                    Path(os.environ.get("PROGRAMFILES", "")) / app_name,
                    Path(os.environ.get("PROGRAMFILES(X86)", "")) / app_name
                ]
                return any(path.exists() for path in app_paths)
            elif sys.platform == "darwin":
                return Path(f"/Applications/{app_name}.app").exists()
            else:
                result = subprocess.run(["which", app_name.lower()], capture_output=True)
                return result.returncode == 0
    except:
        return False

def check_uv_installed():
    """Check if uv is installed"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def select_application():
    """Let user select which application to configure"""
    print("🚀 Papr Memory MCP Setup")
    print("Which application would you like to configure?")
    print()
    print("1. Cursor")
    print("2. Claude") 
    print("3. Other (provide JSON configuration)")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return "cursor"
        elif choice == "2":
            return "claude"
        elif choice == "3":
            return "other"
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def create_mcp_config(app_name, api_key, use_env_var=True):
    """Create or update MCP configuration for the specified application"""
    if app_name == "other":
        return create_generic_config(api_key, use_env_var)
    
    config_path = get_application_config_path(app_name)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config if it exists
    existing_config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
        except:
            existing_config = {}
    
    # Ensure mcpServers exists
    if "mcpServers" not in existing_config:
        existing_config["mcpServers"] = {}
    
    # Create clean configuration for PyPI package using uv
    if use_env_var:
        # Use environment variable placeholder (recommended)
        papr_memory_config = {
            "command": "uv",
            "args": ["run", "--with", "papr-memory-mcp", "papr-memory-mcp"],
            "env": {
                "PAPR_API_KEY": "${PAPR_API_KEY}",
                "MEMORY_SERVER_URL": "${MEMORY_SERVER_URL:-https://memory.papr.ai}"
            }
        }
    else:
        # Embed API key directly (less secure but works without env vars)
        papr_memory_config = {
            "command": "uv",
            "args": ["run", "--with", "papr-memory-mcp", "papr-memory-mcp"],
            "env": {
                "PAPR_API_KEY": api_key,
                "MEMORY_SERVER_URL": "https://memory.papr.ai"
            }
        }
    
    # Add or update papr-memory configuration (overwrites any existing config)
    existing_config["mcpServers"]["papr-memory"] = papr_memory_config
    
    # Write the updated configuration
    try:
        with open(config_path, 'w') as f:
            json.dump(existing_config, f, indent=2)
        
        return str(config_path)
    except Exception as e:
        print(f"❌ Error writing MCP config: {e}")
        return None

def create_generic_config(api_key, use_env_var=True):
    """Create a generic MCP configuration for other applications"""
    if use_env_var:
        config = {
            "mcpServers": {
                "papr-memory": {
                    "command": "uv",
                    "args": ["run", "--with", "papr-memory-mcp", "papr-memory-mcp"],
                    "env": {
                        "PAPR_API_KEY": "${PAPR_API_KEY}",
                        "MEMORY_SERVER_URL": "${MEMORY_SERVER_URL:-https://memory.papr.ai}"
                    }
                }
            }
        }
    else:
        config = {
            "mcpServers": {
                "papr-memory": {
                    "command": "uv",
                    "args": ["run", "--with", "papr-memory-mcp", "papr-memory-mcp"],
                    "env": {
                        "PAPR_API_KEY": api_key,
                        "MEMORY_SERVER_URL": "https://memory.papr.ai"
                    }
                }
            }
        }
    
    print("\n📋 **Generic MCP Configuration**")
    print("Copy this JSON configuration to your application's MCP config file:")
    print()
    print("```json")
    print(json.dumps(config, indent=2))
    print("```")
    print()
    print("💡 **Where to place this configuration:**")
    print("- Check your application's documentation for MCP configuration location")
    print("- Common locations:")
    print("  - ~/.config/[app-name]/mcp.json")
    print("  - ~/.local/share/[app-name]/mcp.json")
    print("  - Application's settings/preferences directory")
    
    return "generic"

def set_system_environment_variable(key, value):
    """Set system-wide environment variable"""
    try:
        if platform.system() == "Windows":
            # Use setx command for Windows
            result = subprocess.run(
                ["setx", key, value], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True, "System-wide environment variable set successfully"
        else:
            # For Unix-like systems, add to shell profile
            shell_profile = None
            if os.path.exists(os.path.expanduser("~/.bashrc")):
                shell_profile = os.path.expanduser("~/.bashrc")
            elif os.path.exists(os.path.expanduser("~/.zshrc")):
                shell_profile = os.path.expanduser("~/.zshrc")
            elif os.path.exists(os.path.expanduser("~/.profile")):
                shell_profile = os.path.expanduser("~/.profile")
            
            if shell_profile:
                # Add export statement to shell profile
                export_line = f'export {key}="{value}"\n'
                with open(shell_profile, 'a') as f:
                    f.write(f"\n# Papr Memory MCP API Key\n{export_line}")
                return True, f"Environment variable added to {shell_profile}. Please restart your terminal or run 'source {shell_profile}'"
            else:
                return False, "Could not find shell profile file"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to set system environment variable: {e}"
    except Exception as e:
        return False, f"Error setting system environment variable: {e}"

def get_api_key():
    """Get API key from user input or environment"""
    api_key = os.getenv("PAPR_API_KEY")
    saved_to_env = False
    
    if not api_key:
        print("🔑 Papr Memory API Key Required")
        print("   You can get your API key from: https://papr.ai/dashboard")
        print()
        
        while True:
            api_key = input("Enter your PAPR_API_KEY: ").strip()
            if api_key:
                # Ask if user wants to save it to environment
                print("\n💡 **Environment Variable Recommendation:**")
                print("   ✅ **Recommended (y)**: Store API key securely in environment variables")
                print("      - More secure (key not stored in config files)")
                print("      - Easy to change without editing config")
                print("      - Follows security best practices")
                print("   ⚠️  **Alternative (n)**: Embed API key directly in config file")
                print("      - Less secure but simpler setup")
                print("      - Key stored in plain text in config file")
                print()
                save_env = input("Save to environment variables? (y/n): ").strip().lower()
                if save_env in ['y', 'yes']:
                    # Try to set system-wide environment variable
                    success, message = set_system_environment_variable("PAPR_API_KEY", api_key)
                    if success:
                        print(f"✅ {message}")
                        saved_to_env = True
                    else:
                        print(f"⚠️  {message}")
                        print("   Falling back to session-only environment variable")
                        os.environ["PAPR_API_KEY"] = api_key
                        saved_to_env = True
                else:
                    print("⚠️  API key not saved to environment variables")
                    print("   The MCP server will need the API key to be set manually")
                break
            else:
                print("❌ API key cannot be empty. Please try again.")
    else:
        saved_to_env = True  # API key already in environment
    
    return api_key, saved_to_env

def main():
    """Main setup function"""
    # Let user select application
    app_name = select_application()
    
    if app_name == "other":
        print("\n🔧 Generic MCP Configuration")
        print("This will provide you with a JSON configuration to copy to your application.")
    else:
        print(f"\n🚀 Setting up Papr Memory MCP for {app_name.title()}...")
        
        # Check if application is installed
        if not check_application_installed(app_name):
            print(f"❌ {app_name.title()} not found. Please install {app_name.title()} first:")
            if app_name == "cursor":
                print("   https://cursor.sh/")
            elif app_name == "claude":
                print("   https://claude.ai/")
            return False
        
        print(f"✅ {app_name.title()} found")
    
    # Check if uv is installed
    if not check_uv_installed():
        print("❌ uv not found. Please install uv first:")
        print("   Windows: https://docs.astral.sh/uv/getting-started/installation/#windows")
        print("   macOS: brew install uv")
        print("   Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    
    print("✅ uv found")
    
    # Get API key from user
    api_key, saved_to_env = get_api_key()
    
    # Create MCP configuration
    config_path = create_mcp_config(app_name, api_key, saved_to_env)
    if not config_path:
        print("❌ Failed to create MCP configuration")
        return False
    
    if app_name == "other":
        print("\n🎉 Setup complete!")
        return True
    
    print(f"✅ MCP configuration created/updated at: {config_path}")
    
    print("\n📋 Configuration Summary:")
    print(f"   📁 Config file: {config_path}")
    print(f"   🔧 Server: papr-memory")
    print(f"   ⚡ Command: uv run --with papr-memory-mcp papr-memory-mcp")
    print(f"   🔑 API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"   🌍 Environment Variable: {'✅ Saved' if saved_to_env else '❌ Not saved'}")
    
    print(f"\n🔧 Next Steps:")
    print(f"1. Restart {app_name.title()}")
    print(f"2. The papr-memory tools will be available in {app_name.title()}")
    print("3. uv will automatically install papr-memory-mcp from PyPI when first used")
    
    if saved_to_env:
        print(f"\n💡 The API key is saved in environment variables and will be used by the MCP server.")
    else:
        print(f"\n⚠️  The API key is embedded in the config file (less secure but works).")
        print(f"   For better security, set PAPR_API_KEY environment variable and update the config.")
    
    print("\n🎉 Setup complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
