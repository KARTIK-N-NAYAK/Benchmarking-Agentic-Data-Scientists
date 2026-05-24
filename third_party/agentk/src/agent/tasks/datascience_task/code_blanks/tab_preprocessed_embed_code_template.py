# <｜fim▁begin｜>
import os
import pandas as pd
import torch
from torch import nn

# --- Design torch model and define variable `TAB_EMBED_DIM`
# 1. Create class TabularEmbedder(nn.Module) and implementing at least methods:
#   - __init__(input_dim: int, embed_dim: int) defines the model architecture
#   - forward:(x: torch.Tensor) -> torch.Tensor, embeds a batch of inputs of shape (batch_size, input_dim).
# 2. Define `TAB_EMBED_DIM`:
# TAB_EMBED_DIM = ...

# <｜fim▁hole｜>
#
# <｜fim▁end｜>

# @NO_MEMORY_START@
if __name__ == "__main__":
    # Load tabular features
    from submissions.new_submission.tab_fe import DataPreprocessor
    root_path = "@ROOT_DS_DATA_PATH@"
    train_data_path = os.path.join(root_path, "train_tab_input_map.csv")
    test_data_path = os.path.join(root_path, "test_tab_input_map.csv")
    train_target_data_path = os.path.join(root_path, "train_tab_target_map.csv")
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    target = pd.read_csv(train_target_data_path)

    preprocessor = DataPreprocessor()
    train, test = preprocessor.preprocess(train, test, target)
    if "id" in train.columns:
        train = train.drop("id", axis=1)


    # Convert to tensor
    train = torch.tensor(train.values)

    # --- We test that the model works:
    input_dim = train.shape[-1]
    model = TabularEmbedder(input_dim=input_dim, embed_dim=TAB_EMBED_DIM).to(torch.float)
    X_embed = model(train.to(torch.float))

    print(f"Embedding size: {X_embed.shape[-1]}")
# @NO_MEMORY_END@
