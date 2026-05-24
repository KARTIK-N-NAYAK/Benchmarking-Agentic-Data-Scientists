# <｜fim▁begin｜>
import os.path

import pandas as pd

# --- Design torch model using a pre-trained encoder model
# Create class TextEmbedder(nn.Module) and implementing the methods:
#   - __init__() defines the model architecture, using a pretrained model such as BERT
#   - forward:(x: str) -> torch.Tensor, embeds a batch of inputs and loads the text embeddings.
# <｜fim▁hole｜>
# --- [End]

if __name__ == "__main__":
    # Test that the model works
    # Load text features
    root_path = "@ROOT_DS_DATA_PATH@"
    train_data_path = os.path.join(root_path, "train_txt_input_map.csv")
    x = pd.read_csv(train_data_path, index_col="id")
    test = "This is a sample text."
    model = TextEmbedder()
    X_embed = model.forward(test)
    # <｜fim▁end｜>

    # @NO_MEMORY_START@
    print(f"Embedding size: {X_embed.shape}")
    # @NO_MEMORY_END@
