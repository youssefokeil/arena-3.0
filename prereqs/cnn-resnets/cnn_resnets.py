# %% [markdown]
# # CNNs & Resnets

import sys
import math
import json
import time
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

sys.path.insert(0, './arena_helpers')
sys.path.insert(0, '..')
from plotly_utils import line
import arena_helpers.tests as tests
import arena_helpers.utils as utils

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# ── Model Classes ────────────────────────────────────────────────────────────

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(t.zeros((1), device=x.device), x)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.bound = 1 / math.sqrt(in_features)
        self.is_bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(self.bound * (2 * t.rand(out_features, in_features) - 1))
        if self.is_bias:
            self.bias = nn.Parameter(self.bound * (2 * t.rand(out_features) - 1))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.T
        if self.is_bias:
            y += self.bias
        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, is_biased={self.is_bias}"


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1:]
        shape_middle = t.prod(t.tensor(shape[start_dim: end_dim + 1])).item()
        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(784, 100, True)
        self.linear2 = Linear(100, 10, True)
        self.flatten = Flatten(1, -1)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bound = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        weight = self.bound * (2 * t.rand(out_channels, in_channels, kernel_size, kernel_size) - 1)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        for mod in self._modules.values():
            x = mod(x)
        return x


class BatchNorm2d(nn.Module):
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean_x = x.mean(dim=(0, 2, 3), keepdim=True)
            var_x = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.num_batches_tracked += 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x.squeeze()
            normalized_x = (x - mean_x) / t.sqrt(var_x + self.eps)
        else:
            normalized_x = (x - self.running_mean.reshape((self.num_features, 1, 1))) / t.sqrt(
                self.running_var.reshape((self.num_features, 1, 1)) + self.eps)
        return normalized_x * self.weight.reshape((self.num_features, 1, 1)) + self.bias.reshape((self.num_features, 1, 1))

    def extra_repr(self) -> str:
        return f"Using Momentum={self.momentum}, epsilon={self.eps}, number of features are {self.num_features}"


class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return einops.reduce(x, "b c h w -> b c", "mean")


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
        self.relu = ReLU()
        self.left_branch = Sequential(
            Conv2d(in_feats, out_feats, 3, first_stride, 1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, 3, 1, 1),
            BatchNorm2d(out_feats)
        )
        if not is_shape_preserving:
            self.right_branch = Sequential(
                Conv2d(in_feats, out_feats, 1, first_stride, 0),
                BatchNorm2d(out_feats)
            )
        else:
            self.right_branch = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.right_branch(x) + self.left_branch(x)
        return self.relu(x)


class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        self.blocks = []
        self.blocks.append(ResidualBlock(in_feats, out_feats, first_stride))
        for _ in range(n_blocks - 1):
            self.blocks.append(ResidualBlock(out_feats, out_feats))
        self.res_block = Sequential(*self.blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.res_block(x)


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes
        self.n_groups = len(n_blocks_per_group)
        self.conv2d = Conv2d(3, out_feats0, 7, 2, 3)
        self.batch_norm = BatchNorm2d(out_feats0)
        self.relu = ReLU()
        self.max_pool = MaxPool2d(3, 2, 1)
        self.groups = []
        in_feats = out_feats0
        for group_i in range(self.n_groups):
            self.groups.append(BlockGroup(n_blocks_per_group[group_i], in_feats, out_features_per_group[group_i], first_strides_per_group[group_i]))
            in_feats = out_features_per_group[group_i]
        self.core_block = nn.Sequential(*self.groups)
        self.avg_pool = AveragePool()
        self.linear = Linear(out_features_per_group[-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.core_block(x)
        x = self.avg_pool(x)
        return self.linear(x)


# ── Functions ────────────────────────────────────────────────────────────────

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081),
])


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""
    mnist_trainset = datasets.MNIST("data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST("data", train=False, download=True, transform=MNIST_TRANSFORM)
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))
    return mnist_trainset, mnist_testset


@dataclass
class SimpleMLPTrainingArgs:
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], list[float], SimpleMLP]:
    """Trains & returns the model, using training parameters from the `args` object."""
    model = SimpleMLP().to(device)
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_val_loader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)
    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []
    accuracies = []

    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}")

        val_pbar = tqdm(mnist_val_loader)
        accuracy, total_size = 0, 0
        for val_imgs, val_labels in val_pbar:
            val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
            val_logits = model(val_imgs)
            val_preds = F.softmax(val_logits, dim=-1)
            _, pred_class = t.max(val_preds, dim=-1)
            accuracy += (pred_class == val_labels).int().sum()
            total_size += val_imgs.shape[0]
        accuracies.append(accuracy.item() / total_size)

    return loss_list, accuracies, model


def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }
    my_resnet.load_state_dict(state_dict_to_load)
    return my_resnet


@t.inference_mode()
def predict(model: nn.Module, images: Float[Tensor, "batch rgb h w"]) -> tuple[Float[Tensor, " batch"], Int[Tensor, " batch"]]:
    """Returns the maximum probability and predicted class for each image."""
    model.eval()
    logits = model(images)
    probs = t.softmax(logits, dim=-1)
    vals, indices = t.max(probs, dim=-1)
    return vals, indices


# ── Entry point (won't run on import) ────────────────────────────────────────

if __name__ == "__main__":
    print(Path.cwd())

    # DataLoader smoke test
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

    for img_batch, label_batch in mnist_testloader:
        print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
        break
    for img, label in mnist_testset:
        print(f"{img.shape=}\n{label=}\n")
        break

    t.testing.assert_close(img, img_batch[0])
    assert label == label_batch[0].item()

    # tqdm demo
    word = "hello!"
    pbar = tqdm(enumerate(word), total=len(word))
    t0 = time.time()
    for i, letter in pbar:
        time.sleep(1.0)
        pbar.set_postfix(i=i, letter=letter, time=f"{time.time() - t0:.3f}")

    # Training
    args = SimpleMLPTrainingArgs()
    loss_list, accuracies, model = train(args)

    # Conv2d test
    m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

    # ResNet34 test
    my_resnet = ResNet34()
    target_resnet = models.resnet34()
    utils.print_param_count(my_resnet, target_resnet)

    # Copy weights & predict
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)

    IMAGE_FILENAMES = [
        "chimpanzee.jpg", "golden_retriever.jpg", "platypus.jpg", "frogs.jpg",
        "fireworks.jpg", "astronaut.jpg", "iguana.jpg", "volcano.jpg", "goofy.jpg", "dragonfly.jpg",
    ]
    IMAGE_FOLDER = Path("resnet_inputs")
    images = [Image.open(IMAGE_FOLDER / f) for f in IMAGE_FILENAMES]

    IMAGENET_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0).to(device)

    with open("imagenet_labels.json") as f:
        imagenet_labels = list(json.load(f).values())

    my_probs, my_predictions = predict(my_resnet, prepared_images)
    pretrained_probs, pretrained_predictions = predict(pretrained_resnet, prepared_images)
    assert (my_predictions == pretrained_predictions).all()
    t.testing.assert_close(my_probs, pretrained_probs, atol=5e-4, rtol=0)
    print("All predictions match!")

    for i, img in enumerate(images):
        table = Table("Model", "Prediction", "Probability")
        table.add_row("My ResNet", imagenet_labels[my_predictions[i]], f"{my_probs[i]:.3%}")
        table.add_row("Reference Model", imagenet_labels[pretrained_predictions[i]], f"{pretrained_probs[i]:.3%}")
        rprint(table)
        display(img)
