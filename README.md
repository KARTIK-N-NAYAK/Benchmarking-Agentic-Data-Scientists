# Benchmarking Agentic Data Scientists

This repository contains a small orchestration layer plus several bundled agent implementations. This guide explains how to create a working environment and how to run each agent from the current checkout.

## Repository map

### Root-level runners
- `src/autogluon_runner_cla.py` — AutoGluon classification runner
- `src/autogluon_runner_reg.py` — AutoGluon regression runner
- `src/download_data.py` — Kaggle downloader for the competition list in `config/agentk.txt`
- `src/run_experiment.py` — current stub, not wired up for end-to-end execution

### Bundled agent projects
- `third_party/agentk` — AgentK
- `third_party/mle_star` — MLE-STAR
- `third_party/mlzero` — AutoGluon Assistant / MLZero
- `third_party/aideml` — AIDE ML

> Important: the top-level `src/agentk_runner.py` is a host-specific wrapper and is not the recommended general entry point. Use the AgentK subproject directly for portable runs.

## Prerequisites

### Required tools
- Python 3.11
- `uv`
- Git
- PowerShell or Command Prompt

### Install `uv`

```powershell
winget install --id=astral-sh.uv -e
uv --version
```

## Recommended environment setup

### 1) Create the root environment

```powershell
cd Benchmarking-Agentic-Data-Scientists
uv sync
```

This creates `.venv` and installs the packages declared in `pyproject.toml`.

### 2) Activate the root environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If you prefer Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## Optional: Kaggle data download

If you want to fetch Kaggle competition data, create `kaggle.json` and accept the competition rules.

```powershell
cd d:/HiWi/Agent/Benchmarking-Agentic-Data-Scientists
uv run python src/download_data.py
```

The downloader reads `config/agentk.txt` and writes data under `data/`.

## Running each agent

### 1) AutoGluon classification runner

Use this runner for classification tasks.

```powershell
uv run python src/autogluon_runner_cla.py \
  --train_path path/to/train.csv \
  --test_path path/to/test.csv \
  --target target_column \
  --output_path outputs/predictions.csv \
  --output_column_name prediction \
  --time_limit 1800 \
  --model_dir models/autogluon_cla \
  --num_gpus 0 \
  --log_path logs/autogluon_cla.log
```

### 2) AutoGluon regression runner

Use this runner for regression tasks.

```powershell
uv run python src/autogluon_runner_reg.py \
  --train_path path/to/train.csv \
  --test_path path/to/test.csv \
  --target target_column \
  --output_path outputs/predictions.csv \
  --output_column_name prediction \
  --time_limit 1800 \
  --model_dir models/autogluon_reg \
  --num_gpus 0 \
  --log_path logs/autogluon_reg.log
```

### 3) AgentK

AgentK is the recommended option for the bundled AgentK workflow. Run it from inside its own subproject.

```powershell

uv run python third_party/agentk/run_complete_pipeline.py \
  --task_id fctp \
  --is_local_task \
  --tabular_task \
  --alt_raw_data_root ./data/raw_local_tasks \
  --llm openai/gpt-4o-mini \
  --code_llm openai/gpt-4o-mini \
  --total_time 1800 \
  --max_time_per_submission 600 \
  --attempt 0 \
  --workspace_name ./workspace/agentk
```

#### AgentK notes
- Replace `fctp` with the task ID you want to run.
- `--alt_raw_data_root` should point at a directory containing the raw task files.
- If you are using a hosted provider, set the environment variables expected by the model wrapper before launching.

### 4) MLE-Star

MLE-Star is a separate project under `third_party/mle_star`. It uses an OpenAI-compatible endpoint and reads `.env` values.

Create a `.env` at the repository root or in `third_party/mle_star`:

```env
OPENAI_BASE_URL=https://your-provider/v1
OPENAI_API_KEY=your-key
OPENROUTER_API_KEY=your-key
ROOT_AGENT_MODEL=openai/gpt-oss-20b:free
```

Then run:

```powershell
uv run python third_party/mle_star/scripts/run_pipeline.py \
  --task-name california-housing-prices \
  --task-type "Tabular Regression"
```


### 5) MLZero

MLZero needs the input and output directories.

```powershell
uv run python -m third_party.ml_zero.run_and_cleanup -i "$DATA_DIR" -o "$OUTPUT_DIR"
```

#### MLZero notes
- The bundled README says Linux-only support at present.
- On Windows, use WSL or a Linux container if you want the most reliable experience.

### 6) AIDE ML

To run AIDE, update the data_dir and desc_file in third_party/aideml/aide/utils/config.yaml.

```powershell
uv run python -m third_party.aideml.aide.run
uv run python -m third_party.aideml.finalize_submission
```

## Common troubleshooting

### 1) AgentK runner fails immediately
- Use the AgentK subproject directly instead of `src/agentk_runner.py`.
- Confirm that `uv sync` has been run inside `third_party/agentk`.

### 2) MLE-Star import errors
- Install `google-adk` in the active environment.
- Verify that `OPENAI_BASE_URL` and `OPENAI_API_KEY` are set.

### 3) Kaggle downloads fail
- Make sure `kaggle.json` exists in your home directory.
- Visit the competition rules page and accept the rules for each competition.

### 4) `mlzero` is not found
- Re-run `uv sync` from the repository root.
- Confirm the current shell has the root `.venv` activated.

## Recommended workflow

1. Run `uv sync` at the repository root.
2. Use the agent-specific commands above.
3. Use the AutoGluon runners for direct tabular experiments, and the bundled agent projects for full agentic workflows.

## Notes on the current repository

- `README.md` is a placeholder and does not document the actual execution flow.
- `src/run_experiment.py` is currently a stub.
- `src/download_description.py` references `src/datasets.txt`, which is not present in the current checkout.
- The AgentK host wrapper in `src/agentk_runner.py` contains hard-coded paths and should be treated as environment-specific.
