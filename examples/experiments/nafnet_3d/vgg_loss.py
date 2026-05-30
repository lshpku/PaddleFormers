import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


DEFAULT_VGG19_NPZ_PATH = "/root/autodl-tmp/vgg19-dcbb9e9d-converted.npz"


class VGG19Features(nn.Layer):
    """Standalone VGG19 feature extractor for perceptual loss."""

    # Torchvision VGG19 feature indices:
    # conv1_1=0, conv1_2=2, conv2_1=5, conv2_2=7, conv3_1=10,
    # conv3_2=12, conv3_3=14, conv3_4=16, conv4_1=19, conv4_2=21,
    # conv4_3=23, conv4_4=25, conv5_1=28, conv5_2=30, conv5_3=32, conv5_4=34.
    CFG = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M",
    ]

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

    def __init__(self, weight_path=DEFAULT_VGG19_NPZ_PATH, return_layers=None):
        super().__init__()
        self.return_layers = tuple(return_layers or self.DEFAULT_LAYERS)
        unknown_layers = set(self.return_layers) - set(self.LAYER_TO_FEATURE_IDX)
        if unknown_layers:
            raise ValueError(f"Unknown VGG return layers: {sorted(unknown_layers)}")

        self.layers = nn.LayerList(self._make_layers())
        self.return_indices = {
            self.LAYER_TO_FEATURE_IDX[name]: name for name in self.return_layers
        }
        self.max_return_idx = max(self.return_indices)
        self._load_npz_weights(weight_path)
        self.eval()
        for param in self.parameters():
            param.stop_gradient = True

    def _make_layers(self):
        layers = []
        in_channels = 3
        for item in self.CFG:
            if item == "M":
                layers.append(nn.MaxPool2D(kernel_size=2, stride=2))
                continue
            conv = nn.Conv2D(in_channels, item, kernel_size=3, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU())
            in_channels = item
        return layers

    def _load_npz_weights(self, weight_path):
        weights = np.load(weight_path)
        missing = []
        for feature_idx, layer in enumerate(self.layers):
            if not isinstance(layer, nn.Conv2D):
                continue
            weight_key = f"features.{feature_idx}.weight"
            bias_key = f"features.{feature_idx}.bias"
            if weight_key not in weights or bias_key not in weights:
                missing.extend([k for k in (weight_key, bias_key) if k not in weights])
                continue
            layer.weight.set_value(weights[weight_key])
            layer.bias.set_value(weights[bias_key])
        if missing:
            raise KeyError(f"Missing VGG weights in {weight_path}: {missing}")

    def forward(self, x):
        outputs = {}
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.return_indices:
                outputs[self.return_indices[idx]] = x
            if idx >= self.max_return_idx:
                break
        return outputs


class VGGLoss(nn.Layer):
    """Standard VGG perceptual loss with ImageNet normalization."""

    def __init__(
        self,
        weight_path=DEFAULT_VGG19_NPZ_PATH,
        layers=None,
        layer_weights=None,
        loss_type="l1",
        input_range="0_1",
        resize_to=None,
    ):
        super().__init__()
        self.vgg = VGG19Features(weight_path=weight_path, return_layers=layers)
        self.layers = self.vgg.return_layers
        self.layer_weights = layer_weights or {name: 1.0 for name in self.layers}
        self.loss_type = loss_type
        self.input_range = input_range
        self.resize_to = resize_to
        self.register_buffer(
            "mean",
            paddle.to_tensor([0.485, 0.456, 0.406], dtype="float32").reshape([1, 3, 1, 1]),
            persistable=False,
        )
        self.register_buffer(
            "std",
            paddle.to_tensor([0.229, 0.224, 0.225], dtype="float32").reshape([1, 3, 1, 1]),
            persistable=False,
        )

    def _preprocess(self, x):
        if self.input_range == "-1_1":
            x = (x + 1.0) * 0.5
        elif self.input_range != "0_1":
            raise ValueError("input_range must be '0_1' or '-1_1'")
        if self.resize_to is not None:
            x = F.interpolate(
                x,
                size=self.resize_to,
                mode="bilinear",
                align_corners=False,
            )
        return (x - self.mean) / self.std

    def _distance(self, pred, target):
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)
        raise ValueError("loss_type must be 'l1' or 'l2'")

    def forward(self, pred, target):
        pred_features = self.vgg(self._preprocess(pred))
        with paddle.no_grad():
            target_features = self.vgg(self._preprocess(target))

        total = paddle.zeros([], dtype=pred.dtype)
        losses = {}
        for name in self.layers:
            loss = self._distance(pred_features[name], target_features[name])
            weighted_loss = loss * self.layer_weights.get(name, 1.0)
            losses[name] = weighted_loss
            total = total + weighted_loss
        return total, losses


if __name__ == "__main__":
    paddle.seed(2026)
    rng = np.random.default_rng(2026)

    loss_fn = VGGLoss()

    pred_np = rng.random([32, 3, 256, 256], dtype=np.float32)
    target_np = rng.random([32, 3, 256, 256], dtype=np.float32)
    pred = paddle.to_tensor(pred_np)
    target = paddle.to_tensor(target_np)
    pred.stop_gradient = False

    print(
        f"before forward: use={paddle.device.memory_allocated()/2**30:.3f} "
        f"max={paddle.device.max_memory_allocated()/2**30:.3f}"
    )

    total_loss, per_layer_losses = loss_fn(pred, target)

    print(
        f"after forward: use={paddle.device.memory_allocated()/2**30:.3f} "
        f"max={paddle.device.max_memory_allocated()/2**30:.3f}"
    )

    print(f"total_loss: {total_loss.item():.9f}")
    for name, value in per_layer_losses.items():
        print(f"{name}: {value.item():.9f}")

    del per_layer_losses
    total_loss.backward()

    print(
        f"after backward: use={paddle.device.memory_allocated()/2**30:.3f} "
        f"max={paddle.device.max_memory_allocated()/2**30:.3f}"
    )

    print("pred.grad:", pred.grad.shape, pred.grad.dtype)
