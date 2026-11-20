"""
Custom CNN trained completely from scratch (no pretrained weights anywhere).

Building block ("conv block"):
    Conv2D 3x3 (padding keeps size) -> [BatchNorm] -> ReLU -> MaxPool 2x2 (halves H and W)

Architecture for the final configuration (input 128 x 128 x 3):

    block 1:  32 filters   -> 64 x 64 x 32
    block 2:  64 filters   -> 32 x 32 x 64
    block 3: 128 filters   -> 16 x 16 x 128
   [block 4: 256 filters   ->  8 x  8 x 256]      (only in the "deeper" experiment)
    head:  Global Average Pooling (one number per filter)
           -> Dense(64) -> ReLU -> Dropout -> Dense(1)
    output: one logit; sigmoid(logit) = model score for "counterfeit"

With head="flatten" the feature map is flattened instead of averaged. That is
the original proposal's design; it has millions of weights in one layer and is
kept as Experiment 1 to show why it overfits.
"""
import torch
import torch.nn as nn


class PackagingCNN(nn.Module):
    def __init__(self, filters=(32, 64, 128), use_bn=True, head="gap",
                 dense_units=64, dropout=0.0, img_size=128):
        super().__init__()
        layers, in_ch = [], 3
        for f in filters:
            layers.append(nn.Conv2d(in_ch, f, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(f))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_ch = f
        self.features = nn.Sequential(*layers)

        if head == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            flat_dim = in_ch
        elif head == "flatten":
            self.pool = nn.Identity()
            side = img_size // (2 ** len(filters))
            flat_dim = in_ch * side * side
        else:
            raise ValueError(head)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x))).squeeze(1)   # logits, shape (B,)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: dict, img_size=128) -> PackagingCNN:
    return PackagingCNN(filters=tuple(cfg["filters"]), use_bn=cfg["use_bn"], head=cfg["head"],
                        dense_units=cfg["dense_units"], dropout=cfg["dropout"], img_size=img_size)
