import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


DEFAULT_VGG19_PTH_PATH = "/dev/shm/vgg19-dcbb9e9d.pth"
DEFAULT_LAYERS = ("relu1_2", "relu2_2", "relu3_4", "relu4_4")
LAYER_TO_FEATURE_IDX = {
    "conv1_1": 0,
    "relu1_1": 1,
    "conv1_2": 2,
    "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5,
    "relu2_1": 6,
    "conv2_2": 7,
    "relu2_2": 8,
    "pool2": 9,
    "conv3_1": 10,
    "relu3_1": 11,
    "conv3_2": 12,
    "relu3_2": 13,
    "conv3_3": 14,
    "relu3_3": 15,
    "conv3_4": 16,
    "relu3_4": 17,
    "pool3": 18,
    "conv4_1": 19,
    "relu4_1": 20,
    "conv4_2": 21,
    "relu4_2": 22,
    "conv4_3": 23,
    "relu4_3": 24,
    "conv4_4": 25,
    "relu4_4": 26,
    "pool4": 27,
    "conv5_1": 28,
    "relu5_1": 29,
    "conv5_2": 30,
    "relu5_2": 31,
    "conv5_3": 32,
    "relu5_3": 33,
    "conv5_4": 34,
    "relu5_4": 35,
    "pool5": 36,
}


class VGG19FeaturesPT(nn.Module):
    def __init__(self, weight_path=DEFAULT_VGG19_PTH_PATH, return_layers=None):
        super().__init__()
        self.return_layers = tuple(return_layers or DEFAULT_LAYERS)
        unknown_layers = set(self.return_layers) - set(LAYER_TO_FEATURE_IDX)
        if unknown_layers:
            raise ValueError(f"Unknown VGG return layers: {sorted(unknown_layers)}")

        model = vgg19(weights=None)
        state_dict = torch.load(weight_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        self.features = model.features
        self.return_indices = {LAYER_TO_FEATURE_IDX[name]: name for name in self.return_layers}
        self.max_return_idx = max(self.return_indices)
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        outputs = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.return_indices:
                outputs[self.return_indices[idx]] = x
            if idx >= self.max_return_idx:
                break
        return outputs


class VGGLossPT(nn.Module):
    def __init__(
        self,
        weight_path=DEFAULT_VGG19_PTH_PATH,
        layers=None,
        layer_weights=None,
        loss_type="l1",
        input_range="0_1",
        resize_to=None,
    ):
        super().__init__()
        self.vgg = VGG19FeaturesPT(weight_path=weight_path, return_layers=layers)
        self.layers = self.vgg.return_layers
        self.layer_weights = layer_weights or {name: 1.0 for name in self.layers}
        self.loss_type = loss_type
        self.input_range = input_range
        self.resize_to = resize_to
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _preprocess(self, x):
        if self.input_range == "-1_1":
            x = (x + 1.0) * 0.5
        elif self.input_range != "0_1":
            raise ValueError("input_range must be '0_1' or '-1_1'")
        if self.resize_to is not None:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def _distance(self, pred, target):
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)
        raise ValueError("loss_type must be 'l1' or 'l2'")

    def forward(self, pred, target):
        pred_features = self.vgg(self._preprocess(pred))
        with torch.no_grad():
            target_features = self.vgg(self._preprocess(target))

        total = pred.new_zeros(())
        losses = {}
        for name in self.layers:
            loss = self._distance(pred_features[name], target_features[name])
            weighted_loss = loss * self.layer_weights.get(name, 1.0)
            losses[name] = weighted_loss
            total = total + weighted_loss
        return total, losses


if __name__ == "__main__":
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = VGGLossPT(
        weight_path=DEFAULT_VGG19_PTH_PATH,
        layers=("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
        layer_weights={"relu1_2": 1.0, "relu2_2": 1.0, "relu3_4": 1.0, "relu4_4": 1.0},
        loss_type="l1",
        input_range="0_1",
    ).to(device)

    pred_np = rng.random([2, 3, 128, 128], dtype=np.float32)
    target_np = rng.random([2, 3, 128, 128], dtype=np.float32)
    pred = torch.from_numpy(pred_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    total_loss, per_layer_losses = loss_fn(pred, target)

    print(f"device: {device}")
    print(f"total_loss: {float(total_loss.detach().cpu().numpy()):.9f}")
    for name, value in per_layer_losses.items():
        print(f"{name}: {float(value.detach().cpu().numpy()):.9f}")
