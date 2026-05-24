# Environment to address Datascience problems

## Agent K


The `Agent` task associated to the [environment](env.py) is in [data_science.py](/src/agent/tasks/data_science.py)

- The yaml configs:
    - Base yaml: [data_science_interact.yaml](../configs/task/data_science_interact.yaml)
    - Flow: [agent-k-solve.yaml](../configs/method/agent-k-solve.yaml)
- In the flow, each action is associated to a custom command defined
  in [ds_custom_commands.py](../../src/agent/commands/ds_custom_commands.py)
- Template
  path: [/src/agent/prompts/templates/data_science/ds_main_pipeline](../../src/agent/prompts/templates/data_science/ds_main_pipeline/)

## Structure

- The graph of dependencies of the different stages (`Add submission`, `Train`, etc.) is defined
  in [env_stages.py](./env_stages.py)
