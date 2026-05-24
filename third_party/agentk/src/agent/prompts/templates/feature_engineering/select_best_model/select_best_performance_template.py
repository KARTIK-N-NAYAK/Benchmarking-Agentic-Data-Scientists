# <｜fim▁begin｜>
import os
import shutil

# <｜fim▁hole｜>
# best_performance_trial = -1  # to be filled by the LLM
# <｜fim▁end｜>

# @NO_MEMORY_START@
def prepare_best_dataset(workspace_path: str, best_performance_trial: int) -> None:
    train_data_source_path = os.path.join(workspace_path, "trials", f"train_tab_input_map_{best_performance_trial}.csv")
    target_data_source_path = os.path.join(workspace_path, "trials", f"train_tab_target_map_{best_performance_trial}.csv")
    target_data_source_path = os.path.join(workspace_path, "trials", f"test_tab_input_map_{best_performance_trial}.csv")

    train_data_target_path = os.path.join(workspace_path, "data", "train_tab_input_map.csv")
    target_data_target_path = os.path.join(workspace_path, "data", "train_tab_target_map.csv")
    target_data_target_path = os.path.join(workspace_path, "data", "test_tab_input_map.csv")

    shutil.move(train_data_source_path, train_data_target_path)
    shutil.move(target_data_source_path, target_data_target_path)
    shutil.move(target_data_source_path, target_data_target_path)
if __name__ == "__main__":
    workspace_path = "@WORKSPACE@"
    prepare_best_dataset(workspace_path, best_performance_trial)
    print(f"Successfully prepared best dataset")
# @NO_MEMORY_END@
