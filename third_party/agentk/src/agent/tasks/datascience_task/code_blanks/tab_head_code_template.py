# <｜fim▁begin｜>
import os.path

import pandas as pd
import torch
from torch import nn

# --- Design torch model and implement a differentiable torch loss functions
# Create class TabularHead(nn.Module) and implementing methods:
#   - __init__(embed_dim: int, output_dim: int) defining the model architecture
#   - forward:(x: torch.Tensor) -> torch.Tensor, taking an embedding as input of shape (batch_size, embed_dim).
# Create functions regression_loss and classification_loss for regression targets and classification targets
# def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     ...

# def classification_loss(pred_logits: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
#     ...

# <｜fim▁hole｜>
#
# <｜fim▁end｜>

if __name__ == "__main__":
    # Test that the model works for a set of arbitrary dimensions
    embed_dim = 32
    output_dim = 10
    batch_size = 5
    model = TabularHead(embed_dim=embed_dim, output_dim=output_dim).to(torch.float)
    output = model(torch.rand(batch_size, embed_dim, dtype=torch.float))

    # --- Test losses

    # For regression targets:
    target = torch.rand(batch_size, output_dim, dtype=torch.float)
    reg_loss = regression_loss(output, target).mean()

    # For classification targets:
    target = torch.rand(batch_size, output_dim, dtype=torch.float)
    class_loss = classification_loss(output, target).mean()
    # @NO_MEMORY_START@
    print(f"Could compute tabular outputs and losses without error.")
    print(f"Output size: {output.shape[-1]}")
    # @NO_MEMORY_END@
