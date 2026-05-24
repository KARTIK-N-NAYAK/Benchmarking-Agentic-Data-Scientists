from pathlib import Path

from agent import PROJECT_ROOT
from ds_agent.competition_struct import DataType


def get_task_scripted_answer_path(input_modalities: set[DataType], output_modalities: set[DataType]) -> Path:
    input_modality = "-".join(sorted(input_modalities))
    target_modality = "-".join(sorted(output_modalities))

    scripted_answer_dir = PROJECT_ROOT / "third_party" / "data_science" / "benchmark_test"
    return scripted_answer_dir / f"input_{input_modality}_target_{target_modality}.txt"
