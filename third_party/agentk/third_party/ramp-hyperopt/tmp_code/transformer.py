from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ramphy import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

num_layers = Hyperparameter(dtype="int", default=1, values=[1, 2, 3, 5])
num_heads = Hyperparameter(dtype="int", default=1, values=[1, 2, 5, 10])
ff_size = Hyperparameter(dtype="int", default=128, values=[128, 256, 512, 1024, 2056])
activation = Hyperparameter(dtype="str", default="gelu", values=["gelu", "relu"])
num_epochs = Hyperparameter(dtype="int", default=100, values=[50, 100, 300])  # Maybe not necessary
loss_type = Hyperparameter(dtype="str", default="regr", values=["class", "regr"])
input_scaling = Hyperparameter(dtype="str", default="minmax", values=["standard", "minmax"])
if str(loss_type) == "class":
    target_bins = Hyperparameter(dtype="int", default=10, values=[5, 10, 20])
    binning_strategy = Hyperparameter(dtype="str", default="quantile", values=["uniform", "quantile", "kmeans"])
    debinning_strategy = Hyperparameter(dtype="str", default="sampling", values=["sampling", "mean"])

INPUT_SCALING = str(input_scaling)
NUM_LAYERS = int(num_layers)
NUM_HEADS = int(num_heads)
FF_SIZE = int(ff_size)
ACTIVATION = str(activation)
NUM_EPOCHS = int(num_epochs)
LOSS_TYPE = str(loss_type)
LEARNING_RATE = 0.001
BATCH_SIZE = 10
if str(loss_type) == "class":
    TARGET_BINS = int(target_bins)
    BINNING_STRATEGY = str(binning_strategy)
    DEBINNING_STRATEGY = str(debinning_strategy)

# TODO add early stopping
# TODO add sam


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# class DecoderLayer(nn.Module):
#     def __init__(
#         self, feat_size: int, num_heads: int, ff_size: int, activation: str = "gelu", dropout: float = 0.1
#     ) -> None:
#         super().__init__()

#         # Attention
#         self._self_attention = nn.MultiheadAttention(embed_dim=feat_size, num_heads=num_heads, dropout=dropout)
#         self.dropout_att = nn.Dropout(dropout)
#         self.norm_att = nn.LayerNorm(feat_size)

#         # FF block
#         self.norm_ff = nn.LayerNorm(feat_size)
#         self.linear1 = nn.Linear(feat_size, ff_size)
#         self.linear2 = nn.Linear(ff_size, feat_size)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         if activation == "gelu":
#             self.activation = nn.GELU
#         elif activation == "relu":
#             self.activation = nn.ReLU
#         else:
#             raise ValueError("Activation has to be either ReLU or GeLU")

#     def forward(self, x: torch.Tensor, att_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         att = self.self_attention(self.norm_att(x), att_mask=att_mask)
#         x = x + att
#         x = x + self.ff_block(self.norm_ff(x))
#         return x

#     def self_attention(self, x: torch.Tensor, att_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         if att_mask is not None:
#             x = x * att_mask
#         x = self._self_attention(x, x, x, is_causal=False)[0]
#         return self.dropout_att(x)

#     def ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
#         return self.dropout2(x)


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_size: int,
        num_heads: int,
        ff_size: int,
        output_size: int,
        softmax_out: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in ["relu", "gelu"]:
            raise ValueError("Only use Relu or Gelu")

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            batch_first=True,
            norm_first=True,  # Apparently this is better
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.softmax_out = softmax_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_embeddings = self.transformer(x)
        output = self.output_layer(out_embeddings)
        if self.softmax_out:
            output = F.softmax(output)
        return output


class Regressor(BaseEstimator):
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def unbin(self, y: np.ndarray) -> np.ndarray:
        """This function takes the binned y and samples the unbinned one"""
        bin_edges = self.y_binner.bin_edges_[0]
        unbinned_y = []
        for binned_y in y:
            lower_edge = bin_edges[binned_y]
            upper_edge = bin_edges[binned_y + 1]
            if DEBINNING_STRATEGY == "sample":
                unbinned_y.append(np.random.uniform(low=lower_edge, high=upper_edge))
            elif DEBINNING_STRATEGY == "mean":
                # This is probably also done by the inverse_transform of the sklearn binner
                unbinned_y.append(lower_edge + upper_edge / 2.0)
            else:
                raise ValueError(f"Debinning strategy {DEBINNING_STRATEGY} not implemented")
        return np.array(unbinned_y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Prepare everything
        # ---------------------------
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        feat_size = X.shape[1]

        # Prepare loss and bin the target
        if LOSS_TYPE == "class":
            self.criterion = nn.CrossEntropyLoss()
            output_size = TARGET_BINS
            softmax_out = False  # No softmax as from here https://jaykmody.com/blog/gpt-from-scratch
            self.y_binner = KBinsDiscretizer(n_bins=TARGET_BINS, encode="ordinal", strategy=BINNING_STRATEGY)
            y = y.reshape(-1, 1)  # Reshape y to be a 2D array
            y = self.y_binner.fit_transform(y).astype(int).flatten()
        elif LOSS_TYPE == "regr":
            self.criterion = nn.MSELoss()
            output_size = 1
            softmax_out = False
        else:
            raise ValueError("Only regr or class as loss types.")

        # Scale the features
        if INPUT_SCALING == "standard":
            self.feature_scaler = StandardScaler()
        elif INPUT_SCALING == "minmax":
            self.feature_scaler = MinMaxScaler()
        else:
            ValueError(f"Only minmax or standard scaling for features. {INPUT_SCALING} is not implemented")
        X = self.feature_scaler.fit_transform(X)

        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)
        if LOSS_TYPE == "class":
            y = y.to(torch.int64)
        elif LOSS_TYPE == "regr":
            y = y.unsqueeze(-1)
        writer = SummaryWriter(log_dir="./tensorboard")

        self.transformer = Transformer(
            num_layers=NUM_LAYERS,
            input_size=feat_size,
            num_heads=NUM_HEADS,
            ff_size=FF_SIZE,
            activation=ACTIVATION,
            output_size=output_size,
            softmax_out=softmax_out,
        ).to(self.device)

        optimizer = optim.Adam(self.transformer.parameters(), lr=LEARNING_RATE)
        # ---------------------------

        # Train
        # ---------------------------
        training_steps = 0
        for epoch in tqdm(range(NUM_EPOCHS), desc="Training Epoch"):
            epoch_loss = 0
            input_ids = np.arange(len(X))
            np.random.shuffle(input_ids)
            batch_count = 0
            for batch_idx in range(0, len(X), BATCH_SIZE):
                batch_X = X[batch_idx : batch_idx + BATCH_SIZE]
                batch_y = y[batch_idx : batch_idx + BATCH_SIZE]

                optimizer.zero_grad()

                output = self.transformer(batch_X)
                loss = self.criterion(input=output, target=batch_y)
                loss.backward()
                optimizer.step()

                # Log
                epoch_loss += loss.detach().cpu().numpy()
                writer.add_scalar("Loss/Batch_loss", loss.item(), training_steps)
                training_steps += 1
                batch_count += 1
            epoch_loss = epoch_loss / batch_count
            writer.add_scalar("Loss/Training_loss", epoch_loss, epoch)
            print(f"Epoch Loss: {epoch_loss}")
        print()
        # ---------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.feature_scaler.transform(X)
        X = torch.Tensor(X).to(self.device)
        y_pred = self.transformer(X)
        if LOSS_TYPE == "class":
            y_pred = self.unbin(y=y_pred)
        return y_pred


if __name__ == "__main__":
    feats = 20
    datapoints = 500
    X = np.zeros((datapoints, feats))
    for i in range(feats):
        X[:, i] = np.random.randn(datapoints) * 5 + np.random.randint(-100, 100)
    y = np.random.randint(0, 5, datapoints)

    reg = Regressor(metadata={"data_description": {"feature_types": ["num"] * feats}})

    reg.fit(X=X, y=y)
