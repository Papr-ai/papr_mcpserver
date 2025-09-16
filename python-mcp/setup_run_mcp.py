# Copyright 2024 Papr AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def install_nodejs_windows():
    """Install Node.js on Windows if not found"""
    logger.info("Checking for Node.js installation on Windows")
    try:
        # Check if Node.js is already installed
        node_check = subprocess.run(['where', 'node'], capture_output=True)
        if node_check.returncode == 0:
            logger.info("Node.js is already installed")
            return True
        
        # Try to install Node.js using winget
        logger.info("Installing Node.js using winget...")
        result = subprocess.run(
            ['winget', 'install', 'OpenJS.NodeJS', '--accept-package-agreements', '--accept-source-agreements'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        if result.returncode == 0:
            logger.info("Node.js installed successfully")
            # Refresh PATH
            setup_windows_path()
            return True
        else:
            logger.warning(f"Node.js installation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("Node.js installation timed out")
        return False
    except Exception as e:
        logger.warning(f"Error installing Node.js: {e}")
        return False

def setup_windows_path():
    """Setup Windows PATH for uv and Node.js"""
    logger.info("Setting up Windows PATH for uv and Node.js")
    try:
        # Add uv to PATH if not already there
        uv_path = Path.home() / ".local" / "bin"
        if uv_path.exists() and str(uv_path) not in os.environ.get("PATH", ""):
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{uv_path};{current_path}"
            logger.info(f"Added uv path to environment: {uv_path}")
        
        # Add Node.js to PATH if not already there
        node_paths = [
            "C:\\Program Files\\nodejs\\",
            "C:\\Program Files (x86)\\nodejs\\"
        ]
        for node_path in node_paths:
            if Path(node_path).exists() and node_path not in os.environ.get("PATH", ""):
                current_path = os.environ.get("PATH", "")
                os.environ["PATH"] = f"{node_path};{current_path}"
                logger.info(f"Added Node.js path to environment: {node_path}")
                break
        
        return True
    except Exception as e:
        logger.warning(f"Error setting up Windows PATH: {e}")
        return False

def install_uv_windows():
    """Install uv on Windows with timeout and error handling"""
    logger.info("Installing uv on Windows")
    try:
        # Try multiple installation methods with timeout
        methods = [
            # Method 1: Try winget with auto-accept
            {
                'name': 'winget',
                'cmd': ['winget', 'install', '--id', 'Astral.uv', '--accept-package-agreements', '--accept-source-agreements'],
                'timeout': 300  # 5 minutes
            },
            # Method 2: Try direct PowerShell installer
            {
                'name': 'powershell installer',
                'cmd': ['powershell', '-ExecutionPolicy', 'Bypass', '-File', 'install_uv.ps1'],
                'timeout': 300  # 5 minutes
            },
            # Method 3: Try pip as fallback
            {
                'name': 'pip',
                'cmd': ['python', '-m', 'pip', 'install', 'uv'],
                'timeout': 180  # 3 minutes
            }
        ]
        
        for method in methods:
            logger.info(f"Trying {method['name']} installation method...")
            try:
                result = subprocess.run(
                    method['cmd'], 
                    capture_output=True, 
                    text=True, 
                    timeout=method['timeout']
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed uv using {method['name']}")
                    # Verify installation - try both direct command and python module
                    verify_result = subprocess.run(['where', 'uv'], capture_output=True)
                    if verify_result.returncode == 0:
                        logger.info("uv installation verified!")
                        return True
                    else:
                        # Try python module verification for pip installations
                        if method['name'] == 'pip':
                            python_verify = subprocess.run(['python', '-m', 'uv', '--version'], capture_output=True)
                            if python_verify.returncode == 0:
                                logger.info("uv installation verified via python module!")
                                return True
                        logger.warning(f"Installation succeeded but verification failed for {method['name']}")
                        continue
                else:
                    logger.warning(f"Installation failed with {method['name']}: {result.stderr}")
                    continue
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Installation timed out with {method['name']}")
                continue
            except Exception as e:
                logger.warning(f"Error with {method['name']}: {e}")
                continue
        
        logger.error("All installation methods failed")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during uv installation: {e}")
        return False

def check_dependencies_available():
    """Check if key dependencies are available for running the server"""
    logger.debug("Checking if key dependencies are available")
    try:
        # Check if uv is available to use virtual environment
        uv_check = subprocess.run(['where' if platform.system() == "Windows" else 'which', 'uv'], capture_output=True)
        use_uv = uv_check.returncode == 0
        
        # If direct command not available, try python module
        if not use_uv:
            python_uv_check = subprocess.run(['python', '-m', 'uv', '--version'], capture_output=True)
            use_uv = python_uv_check.returncode == 0
        
        if use_uv:
            # Check if .venv exists and has dependencies
            venv_path = Path('.venv')
            if venv_path.exists():
                # Use the virtual environment directly
                python_exe = venv_path / ('Scripts' if platform.system() == "Windows" else 'bin') / 'python.exe'
                if not python_exe.exists():
                    python_exe = venv_path / ('Scripts' if platform.system() == "Windows" else 'bin') / 'python'
                
                if python_exe.exists():
                    key_dependencies = ['fastmcp', 'httpx', 'pydantic']
                    
                    for dep in key_dependencies:
                        try:
                            result = subprocess.run([str(python_exe), '-c', f'import {dep}'], 
                                                  capture_output=True, timeout=10)
                            if result.returncode == 0:
                                logger.debug(f"Dependency {dep} is available in virtual environment")
                            else:
                                logger.warning(f"Dependency {dep} is not available in virtual environment")
                                return False
                        except (subprocess.TimeoutExpired, Exception) as e:
                            logger.warning(f"Error checking dependency {dep}: {e}")
                            return False
                    
                    logger.info("All key dependencies are available in virtual environment")
                    return True
                else:
                    logger.warning("Virtual environment exists but Python executable not found")
                    return False
            else:
                logger.warning("Virtual environment .venv does not exist")
                return False
        else:
            # Fallback to checking in current environment
            key_dependencies = ['fastmcp', 'httpx', 'pydantic']
            
            for dep in key_dependencies:
                try:
                    __import__(dep)
                    logger.debug(f"Dependency {dep} is available")
                except ImportError:
                    logger.warning(f"Dependency {dep} is not available")
                    return False
            
            logger.info("All key dependencies are available")
            return True
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return False

def install_dependencies():
    """Install project dependencies using uv or pip fallback"""
    logger.info("Installing project dependencies...")
    try:
        # Check if uv is available (try both direct command and python module)
        uv_check = subprocess.run(['where' if platform.system() == "Windows" else 'which', 'uv'], capture_output=True)
        use_uv = uv_check.returncode == 0
        
        # If direct command not available, try python module
        if not use_uv:
            python_uv_check = subprocess.run(['python', '-m', 'uv', '--version'], capture_output=True)
            use_uv = python_uv_check.returncode == 0
        
        if use_uv:
            logger.info("Using uv for dependency installation")
            # Determine uv command (direct or python module)
            uv_cmd = ['uv'] if uv_check.returncode == 0 else ['python', '-m', 'uv']
            
            # Get virtual environment name from user
            global venv_name  # Make venv_name accessible to other functions
            venv_name = input("Enter virtual environment name (press Enter for default '.venv'): ").strip() or ".venv"
            # Create and activate virtual environment if it doesn't exist
            venv_path = Path(venv_name)
            if not venv_path.exists():
                logger.info("Creating virtual environment...")
                subprocess.run(uv_cmd + ['venv'], check=True)
            
            # Install project dependencies with all extras
            logger.info("Installing dependencies from pyproject.toml...")
            subprocess.run(uv_cmd + ['pip', 'install', '.[all]'], check=True)
        else:
            logger.info("uv not available, using pip for dependency installation")
            # Use pip to install dependencies
            logger.info("Installing dependencies using pip...")
            subprocess.run(['python', '-m', 'pip', 'install', '-e', '.[all]'], check=True)
        
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
        python_path = str(venv_path / ('Scripts' if platform.system() == "Windows" else 'bin') / 'python')
        
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
    """Run paprmcp.py using uv run or direct python"""
    logger.debug("Attempting to start paprmcp.py")
    try:
        # Check if dependencies are available first
        if not check_dependencies_available():
            print("\n❌ Error: Required dependencies are not available.")
            print("Please install dependencies first:")
            print("1. Run without --skip-deps: python setup_run_mcp.py")
            print("2. Or install manually: python -m uv pip install .[all]")
            print("3. Or use uv run with dependencies: python -m uv run --with fastapi python paprmcp.py")
            sys.exit(1)
        
        # Check if uv is available (try both direct command and python module)
        uv_check = subprocess.run(['where' if platform.system() == "Windows" else 'which', 'uv'], capture_output=True)
        use_uv = uv_check.returncode == 0
        
        # If direct command not available, try python module
        if not use_uv:
            python_uv_check = subprocess.run(['python', '-m', 'uv', '--version'], capture_output=True)
            use_uv = python_uv_check.returncode == 0
        
        if use_uv:
            # Check if .venv exists and use it directly
            venv_path = Path('.venv')
            if venv_path.exists():
                # Use the virtual environment directly
                python_exe = venv_path / ('Scripts' if platform.system() == "Windows" else 'bin') / 'python.exe'
                if not python_exe.exists():
                    python_exe = venv_path / ('Scripts' if platform.system() == "Windows" else 'bin') / 'python'
                
                if python_exe.exists():
                    cmd = [str(python_exe), 'paprmcp.py']
                    logger.debug(f"Running command: {' '.join(cmd)} (using virtual environment)")
                else:
                    # Fallback to uv run with dependencies
                    uv_cmd = ['uv'] if uv_check.returncode == 0 else ['python', '-m', 'uv']
                    cmd = uv_cmd + ['run', '--with', 'fastapi', 'python', 'paprmcp.py']
                    logger.debug(f"Running command: {' '.join(cmd)} (uv run with fastapi)")
            else:
                # Fallback to uv run with dependencies
                uv_cmd = ['uv'] if uv_check.returncode == 0 else ['python', '-m', 'uv']
                cmd = uv_cmd + ['run', '--with', 'fastapi', 'python', 'paprmcp.py']
                logger.debug(f"Running command: {' '.join(cmd)} (uv run with fastapi)")
        else:
            # Fallback to direct python execution
            cmd = ['python', 'paprmcp.py']
            logger.debug(f"Running command: {' '.join(cmd)} (uv not available)")
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running paprmcp.py: {e}")
        print("\n❌ Server failed to start. This might be due to missing dependencies.")
        print("Try running: python -m uv run --with fastapi python paprmcp.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error running paprmcp.py: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Setup MCP configuration')
    parser.add_argument('--use-existing', action='store_true', help='Use existing configuration if available')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies only and exit')
    parser.add_argument('--run-server', action='store_true', help='Skip setup and run server directly')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.install_deps:
        print("Installing dependencies only...")
        if install_dependencies():
            print("✓ Dependencies installed successfully!")
        else:
            print("❌ Failed to install dependencies")
            sys.exit(1)
        return
    
    if args.run_server:
        print("Starting MCP server...")
        run_paprmcp()
        return

    print("Welcome to MCP Setup!")
    print("--------------------")
    
    try:
        # Setup Windows PATH and install Node.js if needed
        if platform.system() == "Windows":
            setup_windows_path()
            # Check and install Node.js if needed (for MCP inspector)
            install_nodejs_windows()
        
        # Check if uv is installed
        logger.debug("Checking if uv is installed")
        uv_installed = False
        
        if platform.system() == "Windows":
            uv_check = subprocess.run(['where', 'uv'], capture_output=True)
            if uv_check.returncode == 0:
                uv_installed = True
                logger.info("uv is already installed")
            else:
                logger.info("uv is not installed. Attempting to install...")
                uv_installed = install_uv_windows()
        else:
            uv_check = subprocess.run(['which', 'uv'], capture_output=True)
            if uv_check.returncode == 0:
                uv_installed = True
                logger.info("uv is already installed")
            else:
                logger.info("Installing uv on Unix-like system")
                # Make install script executable first
                subprocess.run(['chmod', '+x', './install_uv.sh'], check=True)
                subprocess.run(['bash', './install_uv.sh'], check=True)
                uv_installed = True
        
        if not uv_installed:
            print("\n⚠️  Warning: uv installation failed or was skipped.")
            print("You can install uv manually using one of these methods:")
            print("1. Run: winget install --id Astral.uv")
            print("2. Download from: https://github.com/astral-sh/uv/releases")
            print("3. Use pip: pip install uv")
            print("\nAfter installing uv, run this script again with --skip-deps")
            print("Or continue without uv (some features may not work properly)")
            
            continue_without_uv = input("\nContinue without uv? (y/n): ").lower().strip()
            if continue_without_uv != 'y':
                print("Setup cancelled. Please install uv and try again.")
                sys.exit(1)

        # Install dependencies
        if not args.skip_deps:
            print("\nInstalling dependencies...")
            if not install_dependencies():
                print("Error: Failed to install dependencies")
                sys.exit(1)
            print("✓ Dependencies installed")
        else:
            print("\nSkipping dependency installation (--skip-deps flag used)")
            # Check if dependencies are available for running the server
            if not check_dependencies_available():
                print("⚠️  Warning: Some dependencies may not be available.")
                print("If you encounter errors when running the server, try running without --skip-deps")
                print("or install dependencies manually: python -m uv pip install .[all]")
        
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
            # Check if uv is available for the message (try both direct command and python module)
            uv_check = subprocess.run(['where' if platform.system() == "Windows" else 'which', 'uv'], capture_output=True)
            use_uv = uv_check.returncode == 0
            
            # If direct command not available, try python module
            if not use_uv:
                python_uv_check = subprocess.run(['python', '-m', 'uv', '--version'], capture_output=True)
                use_uv = python_uv_check.returncode == 0
            
            if use_uv:
                uv_cmd = "uv" if uv_check.returncode == 0 else "python -m uv"
                print(f"\nYou can start the server later by running: {uv_cmd} run python paprmcp.py")
            else:
                print("\nYou can start the server later by running: python paprmcp.py")
        
        print("\nSetup complete! 🎉")
        print("\n📝 Notes:")
        print("- MCP server is configured and ready to use")
        print("- Node.js and uv are installed for MCP inspector support")
        print("- You can use 'fastmcp dev paprmcp.py' to run with MCP inspector")
        print("- Or use 'uv run python paprmcp.py' for direct server mode")
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\nError: Setup failed - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 