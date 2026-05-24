# <｜fim▁begin｜>
import os.path

import torch
import pandas as pd
import torchvision.models as models
import torchvision.transforms as T
from torch import nn
from PIL import Image

# --- Design torch model using a pre-trained torchvision model
# Create class ImageEmbedder(nn.Module) and implementing the methods:
#   - __init__() defines the model architecture, using a pretrained model such as resnet50
#   - forward:(x: torch.Tensor) -> torch.Tensor, embeds a batch of inputs and loads the image embeddings.
#   - unfreeze:(n_last_layers: int = 2) unfreezes the parameters of the last n layers of the image embedding model.

# <｜fim▁hole｜>
#
# <｜fim▁end｜>

if __name__ == "__main__":
    # Test that the model works
    from submissions.new_submission.img_transform import CustomTrainImageInputTransform

    # Load image features
    root_path = "@ROOT_DS_DATA_PATH@"
    train_data_path = os.path.join(root_path, "train_img_input_map.csv")
    x = pd.read_csv(train_data_path, index_col="id")

    path = x.iloc[0, 0]
    image = Image.open(path)
    tensor_image = CustomTrainImageInputTransform(image)
    model = ImageEmbedder()
    X_embed = model(tensor_image.unsqueeze(0))

    # @NO_MEMORY_START@
    def test_unfreeze_functionality() -> None:
        model = ImageEmbedder()
        assert hasattr(model, 'unfreeze'), "The model does not have an 'unfreeze' method."

        try:
            model.unfreeze(n_last_layers=3)
        except Exception as e:
            assert False, f"Unfreeze method raised an exception: {e}"

        print("Unfreeze method exists and runs successfully.")


    test_unfreeze_functionality()

    print(f"Embedding size: {X_embed.shape}")
    # @NO_MEMORY_END@
