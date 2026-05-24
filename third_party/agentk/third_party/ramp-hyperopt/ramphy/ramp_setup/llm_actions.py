import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import ramphy as rh
from ramphy import ramp_setup as rs


def execute_script(script_path: str | Path, env_args: Dict, script_args: Dict, hydra_args: Dict) -> bool:
    """executes the required python script

    Args:
        script_path (str | Path): _description_
        env_args (Dict): _description_
        script_args (Dict): _description_
        hydra_args (Dict): _description_

    Returns:
        bool: True if terminated properly, False otherwise
    """
    # Ensure script_path is a Path object
    script_path = str(script_path)

    # Add environment args
    env = os.environ.copy()
    for env_variable in env_args:
        env[env_variable] = str(env_args[env_variable])

    # Construct the command to run the script with its arguments
    cmd = ["python", script_path]
    for key, value in script_args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    # Add any arguments for hydra (these are handled differently than script)
    for key, value in hydra_args.items():
        cmd.append(f"{key}={value} ")

    try:
        # Run the script
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Optionally, you can print or log the output
        print(result.stdout)
        print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        # Handle errors in the script execution
        print(f"Script failed with exit code {e.returncode}")
        print(e.output)
        return False
    except Exception as e:
        # Handle other potential exceptions
        print(f"An error occurred: {e}")
        return False
