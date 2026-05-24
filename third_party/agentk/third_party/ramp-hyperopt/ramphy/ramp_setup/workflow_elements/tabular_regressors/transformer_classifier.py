import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ramphy import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

# RAMP START HYPERPARAMETERS
num_layers = Hyperparameter(dtype="int", default=2, values=[2, 3, 5])
num_heads = Hyperparameter(dtype="int", default=5, values=[1, 2, 5, 10])
ff_size = Hyperparameter(dtype="int", default=256, values=[128, 256, 512, 1024, 2056])
input_scaling = Hyperparameter(dtype="str", default="quantile", values=["standard", "minmax", "quantile", "none"])
target_bins = Hyperparameter(dtype="int", default=10, values=[5, 10, 20])
binning_strategy = Hyperparameter(dtype="str", default="quantile", values=["uniform", "quantile", "kmeans"])
unbinning_strategy = Hyperparameter(dtype="str", default="sampling", values=["sampling", "mean"])
optimizer = Hyperparameter(dtype="str", default="sam", values=["sam", "adam"])
# RAMP END HYPERPARAMETERS


INPUT_SCALING = str(input_scaling)
NUM_LAYERS = int(num_layers)
NUM_HEADS = int(num_heads)
FF_SIZE = int(ff_size)
ACTIVATION = "gelu"
NUM_EPOCHS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 512
TARGET_BINS = int(target_bins)
BINNING_STRATEGY = str(binning_strategy)
UNBINNING_STRATEGY = str(unbinning_strategy)
OPTIMIZER = str(optimizer)

% NOT WORKING YET

class SAM(optim.Optimizer):
    """
    SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization https://arxiv.org/abs/2010.01412
    https://github.com/davda54/sam
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {{rho}}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


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
    """This is a regressor that trains a transformer as a classifier.
    It does so by binning the target, training the transformer with CELoss and then at
    prediction time, it unbins the prediction of the transformer to get the target"""

    def __init__(self, metadata: dict):
        self.metadata = metadata

    def unbin(self, y: np.ndarray) -> np.ndarray:
        """This function takes the binned y and samples the unbinned one"""
        bin_edges = self.y_binner.bin_edges_[0]
        unbinned_y = []
        for binned_y in y:
            lower_edge = bin_edges[binned_y]
            upper_edge = bin_edges[binned_y + 1]
            if UNBINNING_STRATEGY == "sampling":
                unbinned_y.append(np.random.uniform(low=lower_edge, high=upper_edge))
            elif UNBINNING_STRATEGY == "mean":
                # This is probably also done by the inverse_transform of the sklearn binner
                unbinned_y.append(lower_edge + upper_edge / 2.0)
            else:
                raise ValueError(f"Debinning strategy {{UNBINNING_STRATEGY}} not implemented")
        return np.array(unbinned_y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Prepare everything
        # ---------------------------
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        feat_size = X.shape[1]
        self.criterion = nn.CrossEntropyLoss()
        softmax_out = False  # No softmax as from here https://jaykmody.com/blog/gpt-from-scratch
        output_size = TARGET_BINS
        writer = SummaryWriter(log_dir="./tensorboard")

        # Bin the target
        self.y_binner = KBinsDiscretizer(n_bins=TARGET_BINS, encode="ordinal", strategy=BINNING_STRATEGY)
        y = y.reshape(-1, 1)  # Reshape y to be a 2D array
        y = self.y_binner.fit_transform(y).astype(int).flatten()

        # Scale the features
        if INPUT_SCALING == "standard":
            self.feature_scaler = StandardScaler()
        elif INPUT_SCALING == "minmax":
            self.feature_scaler = MinMaxScaler()
        elif INPUT_SCALING == "quantile":
            self.feature_scaler = QuantileTransformer()
        elif INPUT_SCALING == "none":
            self.feature_scaler = None
        else:
            ValueError(f"Only minmax or standard scaling for features. {{INPUT_SCALING}} is not implemented")
        if self.feature_scaler is not None:
            X = self.feature_scaler.fit_transform(X)

        X = torch.Tensor(X).to(self.device)  # type: ignore
        y = torch.Tensor(y).to(torch.int64).to(self.device)  # type: ignore

        self.transformer = Transformer(
            num_layers=NUM_LAYERS,
            input_size=feat_size,
            num_heads=NUM_HEADS,
            ff_size=FF_SIZE,
            activation=ACTIVATION,
            output_size=output_size,
            softmax_out=softmax_out,
        ).to(self.device)
        self.transformer.train()

        if OPTIMIZER == "adam":
            optimizer = optim.Adam(self.transformer.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER == "sam":
            optimizer = SAM(
                self.transformer.parameters(),
                base_optimizer=optim.Adam,
                rho=0.5,
                lr=LEARNING_RATE,
                weight_decay=1e-5,
            )
        else:
            ValueError("Only adam or sam optimizers available")
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
                # Forward
                output = self.transformer(batch_X)
                loss = self.criterion(input=output.to(torch.float), target=batch_y.to(torch.float))

                if OPTIMIZER == "adam":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif OPTIMIZER == "sam":
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    output = self.transformer(batch_X)
                    loss = self.criterion(input=output.to(torch.float), target=batch_y.to(torch.float))
                    loss.backward()
                    optimizer.second_step(zero_grad=True)

                # Log
                epoch_loss += loss.detach().cpu().numpy()
                writer.add_scalar("Loss/Batch_loss", loss.item(), training_steps)
                training_steps += 1
                batch_count += 1
            epoch_loss = epoch_loss / batch_count
            writer.add_scalar("Loss/Training_loss", epoch_loss, epoch)
        # ---------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.transformer.eval()
        y_pred = self.get_logits(X=X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def get_logits(self, X: np.ndarray) -> np.ndarray:
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)  # type: ignore
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)  # type: ignore
            y_pred = self.transformer(X).detach().cpu().numpy()
        return y_pred
