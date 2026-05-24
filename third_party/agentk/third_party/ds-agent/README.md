# Agent K

This package collects the external scripts necessary for the data-science pipeline.

## Installation

Begin by activating your python virtual environment with agent, then run the following commands (from project root):
```shell
pip install -e ./third_party/ds-agent
```

## Adding a competition
- add the competition ID to the enum class `CompetitionID`
- add the competition in the competition list `ALL_COMPETITIONS_LIST`