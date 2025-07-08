import os
import json
import platform
from pathlib import Path
import subprocess
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install project dependencies using uv"""
    logger.info("Installing project dependencies...")
    try:
        # Get virtual environment name from user
        global venv_name  # Make venv_name accessible to other functions
        venv_name = input("Enter virtual environment name (press Enter for default '.venv'): ").strip() or ".venv"
        # Create and activate virtual environment if it doesn't exist
        venv_path = Path(venv_name)
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            subprocess.run(['uv', 'venv'], check=True)
        
        # Install project dependencies with all extras
        logger.info("Installing dependencies from pyproject.toml...")
        subprocess.run(['uv', 'pip', 'install', '.[all]'], check=True)
        
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error installing dependencies: {e}")
        return False

def get_default_mcp_path():
    """Get the default MCP config path and filename based on OS"""
    """Get the default MCP config path based on OS"""
    system = platform.system().lower()
    home = Path.home()
    
    logger.debug(f"Operating System: {system}")
    logger.debug(f"Home Directory: {home}")
    # Ask for client name
    client_name = input("Enter client name ('claude' or 'cursor', press Enter for default 'claude'): ").strip().lower()
    if client_name not in ['claude', 'cursor']:
        client_name = 'none'
    logger.debug(f"Client name: {client_name}")

    # Set path based on client
    if client_name.lower() == "cursor":
        path = Path("~/.cursor").expanduser()
    elif client_name.lower() == "claude":
         if system == "darwin":  # macOS
            path = home / "Library" / "Application Support" / client_name
         elif system == "windows": 
            path = home / "AppData" / "Roaming" / client_name
         else:  # Linux and others
            path = home / ".config" / client_name
    else:  # Default to Claude path structure
        path = None
        config_file = None

    logger.debug(f"MCP Config Path: {path}")
    if path:
        path.mkdir(parents=True, exist_ok=True)
        config_file = path / ("mcp.json" if client_name.lower() == "cursor" else "claude_desktop_config.json")
        logger.debug(f"Config file path: {config_file}")
    return path, config_file

def read_env_api_key():
    """Read API key from .env file"""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('PAPR_API_KEY='):
                    return line.strip().split('=', 1)[1]
    return None

def update_env_file(api_key):
    """Update or create .env file with API key"""
    env_path = Path(".env")
    logger.debug(f"Updating .env file at: {env_path.absolute()}")
    
    # Read existing env file if it exists
    env_vars = {}
    if env_path.exists():
        logger.debug("Found existing .env file")
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
                    logger.debug(f"Found existing env var: {key}")

    # Update API key
    env_vars['PAPR_API_KEY'] = api_key
    env_vars['MEMORY_SERVER_URL'] = 'https://memory.papr.ai'

    # Write back to .env file
    try:
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        logger.debug("Successfully updated .env file")
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")
        raise

def update_mcp_config(api_key):
    """Update MCP config for Claude"""
    path, config_file = get_default_mcp_path()
    logger.debug(f"Creating MCP directory at: {path}")
    
    try:
        # Read existing config if it exists
        existing_config = {}
        
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                existing_config = json.load(f)
        
        
        
        # Get current working directory and virtual environment path
        cwd = Path.cwd()
        venv_path = cwd / ".venv"
        python_path = str(venv_path / "bin" / "python")
        
        # MCP Server configuration
        mcp_server_config = {
            "Papr MCP Server": {
                "command": "uv",
                "timeout": 300,
                "args": [
                    "run",
                    "--with",
                    "fastmcp",
                    "--python",
                    python_path,
                    "fastmcp",
                    "run",
                    str(cwd / "paprmcp.py")
                ],
                "env": {
                    "PYTHONPATH": str(cwd),
                    "PAPR_API_KEY": api_key                   
                }
            }
        }
        if path:
            # Update or add mcpServers section
            if "mcpServers" in existing_config:
                existing_config["mcpServers"].update(mcp_server_config)
            else:
                existing_config["mcpServers"] = mcp_server_config
            
            # Write updated config
            with open(config_file, 'w') as f:
                json.dump(existing_config, f, indent=2)
            logger.debug("Successfully wrote config file")
            
            print(f"Updated MCP config at: {config_file}")
        else:
            # dump mcp_server_config to the console, 
            # print to use instruction to copy paste to the client mcp config file            
            print("Please copy paste the following to the client mcp config file:")
            print(json.dumps(mcp_server_config, indent=2))
    except Exception as e:
        logger.error(f"Error updating MCP config: {e}")
        raise

def run_paprmcp():
    """Run paprmcp.py using uv run"""
    logger.debug("Attempting to start paprmcp.py")
    try:
        # Use uv run to ensure virtual environment is used
        cmd = ['uv', 'run', 'python', 'paprmcp.py']
        logger.debug(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running paprmcp.py: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error running paprmcp.py: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Setup MCP configuration')
    parser.add_argument('--use-existing', action='store_true', help='Use existing configuration if available')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--run-server', action='store_true', help='Skip setup and run server directly')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.run_server:
        print("Starting MCP server...")
        run_paprmcp()
        return

    print("Welcome to MCP Setup!")
    print("--------------------")
    
    try:
        # Check if uv is installed
        logger.debug("Checking if uv is installed")
        if platform.system() == "Windows":
            uv_check = subprocess.run(['where', 'uv'], capture_output=True)
            if uv_check.returncode != 0:
                logger.info("Installing uv on Windows")
                subprocess.run(['powershell', '-File', 'install_uv.ps1'], check=True)
        else:
            uv_check = subprocess.run(['which', 'uv'], capture_output=True)
            if uv_check.returncode != 0:
                logger.info("Installing uv on Unix-like system")
                # Make install script executable first
                subprocess.run(['chmod', '+x', './install_uv.sh'], check=True)
                subprocess.run(['bash', './install_uv.sh'], check=True)

        # Install dependencies
        if not args.skip_deps:
            print("\nInstalling dependencies...")
            if not install_dependencies():
                print("Error: Failed to install dependencies")
                sys.exit(1)
            print("✓ Dependencies installed")
        
        # Get API key from user or .env file
        api_key = read_env_api_key()
        if api_key:
            print(f"Found existing API key in .env file")
            use_existing = input("Use existing API key? (y/n): ").lower().strip()
            if use_existing != 'y':
                api_key = input("Please enter your Papr API key: ").strip()
        else:
            api_key = input("Please enter your Papr API key: ").strip()
        
        if not api_key:
            logger.error("No API key provided")
            print("Error: API key is required")
            sys.exit(1)
        
        # Update .env file
        print("\nUpdating environment configuration...")
        update_env_file(api_key)
        print("✓ Environment configuration updated")
        
        # Update MCP config
        print("\nUpdating MCP configuration...")
        update_mcp_config(api_key)
        print("✓ MCP configuration updated")
        
        # Ask to run paprmcp.py
        should_run = input("\nWould you like to start the MCP server now? (y/n): ").lower()
        if should_run == 'y':
            print("\nStarting MCP server...")
            run_paprmcp()
        else:
            print("\nYou can start the server later by running: uv run python paprmcp.py")
        
        print("\nSetup complete! 🎉")
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\nError: Setup failed - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 