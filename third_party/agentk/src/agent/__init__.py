from pathlib import Path

from agent import agents

_agent_path = Path(__file__).parent
assert (
    _agent_path.name == "agent" and _agent_path.parent.name == "src"
), "agent must be installed editably with `pip install -e .` !"

PROJECT_ROOT = _agent_path.parent.parent
