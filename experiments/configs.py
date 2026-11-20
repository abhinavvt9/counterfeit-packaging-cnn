"""
CNN experiment configurations.

Each experiment changes ONE thing relative to the previous one (an "ablation"),
so the effect of each design decision can be measured on its own:

  E1  proposal CNN     3 blocks, no BatchNorm, Flatten -> Dense(128)   (millions of weights)
  E2  + BN + GAP       add BatchNorm, replace Flatten with Global Average Pooling
  E3  + Dropout        dropout 0.5 before the output layer
  E4  + Augmentation   small rotations / crops / brightness-contrast jitter
  E5  deeper           4th conv block (256 filters)
  E6  lower LR         E5 with learning rate 3e-4 instead of 1e-3
  E7  narrower         E4 with half the filters (16/32/64)

All experiments run under the controlled protocol + grouped split, on the same
5 folds. The final configuration is the one with the best mean VALIDATION
ROC-AUC. The test set is never used to choose.
"""

BASE = dict(filters=[32, 64, 128], use_bn=True, head="gap", dense_units=64,
            dropout=0.0, augment=False, lr=1e-3, weight_decay=0.0)

EXPERIMENTS = {
    "E1_proposal_cnn": dict(BASE, use_bn=False, head="flatten", dense_units=128),
    "E2_bn_gap": dict(BASE),
    "E3_dropout": dict(BASE, dropout=0.5),
    "E4_augmentation": dict(BASE, dropout=0.5, augment=True),
    "E5_deeper": dict(BASE, dropout=0.5, augment=True, filters=[32, 64, 128, 256]),
    "E6_lower_lr": dict(BASE, dropout=0.5, augment=True, filters=[32, 64, 128, 256], lr=3e-4),
    "E7_narrower": dict(BASE, dropout=0.5, augment=True, filters=[16, 32, 64]),
}

DESCRIPTIONS = {
    "E1_proposal_cnn": "Proposal CNN (Flatten, no BN)",
    "E2_bn_gap": "+ BatchNorm + GAP",
    "E3_dropout": "+ Dropout 0.5",
    "E4_augmentation": "+ Augmentation",
    "E5_deeper": "+ 4th block (256)",
    "E6_lower_lr": "E5 with LR 3e-4",
    "E7_narrower": "E4 with 16/32/64 filters",
}
