# <｜fim▁begin｜>
import os
import pandas as pd

# class ModelTrainer():
#    self.model
#     """Class to handle model training, evaluation, and saving the trained model"""
#     def train_model(self, train: pd.DataFrame, train_target: pd.DataFrame):
#         ...
#         return model
#     def evaluate_model(self, train: pd.DataFrame, train_target: pd.DataFrame) -> float:
#         ...
#         return accuracy
# <｜fim▁hole｜>
#
# <｜fim▁end｜>

# @NO_MEMORY_START@
if __name__ == "__main__":
    root_path = "@ROOT_DS_DATA_PATH@"
    workspace_path = "@WORKSPACE@"
    train_data_path = os.path.join(workspace_path, 'data', "train_tab_input_map.csv")
    target_data_path = os.path.join(workspace_path, 'data', "train_tab_target_map.csv")
    train = pd.read_csv(train_data_path)
    train_target = pd.read_csv(target_data_path)

    trainer = ModelTrainer()
    trainer.train_model(train, train_target)
    performance = trainer.evaluate_model(train, train_target)
    perf_file_name = os.path.join(workspace_path, "performance.txt")
    perf_txt = f"Model performance : {performance}"
    with open(perf_file_name, "w") as f:
        f.write(perf_txt)
    print(perf_txt)
    print()
# @NO_MEMORY_END@
