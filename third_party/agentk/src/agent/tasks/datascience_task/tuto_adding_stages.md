1. Add the jinja and the code with blanks in [code_blanks](src/agent/tasks/datascience_task/code_blanks). Code should output something otherwise an error will occur in the Python interpreter tool.
1. Update template_code_reset method in data_science.py to make sure you copy code templates in the workspace
2. Add the custom command in ds_custom_commands.py.
3. Update the plan of agent-k-solve.yaml.
4. Update the env and notably the update_obs method. Update some files (handle the blank filling)
5. Update the human test file (create new one if necessary).
