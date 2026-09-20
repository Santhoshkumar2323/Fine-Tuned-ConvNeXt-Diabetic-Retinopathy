import os
import gc
import cv2
import copy
import random
import time
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

class CFG:
    epochs = 10
    run_folds = 5
    n_folds = 5
    model_name = "convnext_tiny.fb_in22k_ft_in1k"
    image_size = 448
    num_classes = 5
    batch_size = 16
    num_workers = 2

    lr_head = 1e-4
    lr_backbone = 2e-5

    weight_decay = 1e-4
    warmup_epochs = 1

    ce_weight = 0.70
    ordinal_weight = 0.30

    label_smoothing = 0.05

    dropout = 0.20
    grad_clip = 1.0

    use_ema = True
    ema_decay = 0.999

    max_eyepacs_per_class = 2000

    crop_tol = 7
    circle_mask_ratio = 0.98

    horizontal_flip_p = 0.5
    affine_p = 0.5
    brightness_p = 0.25

    use_class_weights = True
    use_tta = True

    patience = 5

    seed = 42

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    aptos_dir = (
        "/kaggle/input/datasets/"
        "shiva8227/aptos2019-blindness-detection"
    )

    eyepacs_dir = (
        "/kaggle/input/datasets/"
        "sovitrath/"
        "diabetic-retinopathy-2015-data-colored-resized"
    )

    output_dir = "/kaggle/working/dr_v3"


os.makedirs(CFG.output_dir, exist_ok=True)


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)

print("=" * 70)
print("TRAINING")
print("=" * 70)

print(f"Device       : {CFG.device}")
print(f"Image size   : {CFG.image_size}")
print(f"Batch size   : {CFG.batch_size}")
print(f"Epochs       : {CFG.epochs}")
print(f"Folds        : {CFG.run_folds} / {CFG.n_folds}")
print(f"Model        : {CFG.model_name}")

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    mem = (
        torch.cuda.get_device_properties(0)
        .total_memory / 1e9
    )

    print(f"GPU memory   : {mem:.2f} GB")

else:

    print("[WARNING] CUDA unavailable.")



def crop_image_from_gray(image, tol=7):

    if image.ndim == 2:

        mask = image > tol

        if mask.sum() == 0:
            return image

        return image[
            np.ix_(
                mask.any(axis=1),
                mask.any(axis=0)
            )
        ]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    mask = gray > tol

    if mask.sum() == 0:
        return image

    rows = mask.any(axis=1)
    cols = mask.any(axis=0)

    if rows.sum() == 0 or cols.sum() == 0:
        return image

    return image[
        np.ix_(rows, cols)
    ]


def apply_circle_mask(
    image,
    size,
    ratio=0.98
):

    mask = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    center = (
        size // 2,
        size // 2
    )

    radius = int(
        size * 0.5 * ratio
    )

    cv2.circle(
        mask,
        center,
        radius,
        1,
        thickness=-1
    )

    return (
        image *
        mask[..., None]
    ).astype(np.uint8)


def ben_graham_processing(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = crop_image_from_gray(
        image,
        CFG.crop_tol
    )

    image = cv2.resize(
        image,
        (
            CFG.image_size,
            CFG.image_size
        ),
        interpolation=cv2.INTER_AREA
    )

    ksize = int(
        CFG.image_size / 10
    )

    if ksize % 2 == 0:
        ksize += 1

    blur = cv2.GaussianBlur(
        image,
        (ksize, ksize),
        0
    )

    image = cv2.addWeighted(
        image,
        4,
        blur,
        -4,
        128
    )

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    image = apply_circle_mask(
        image,
        CFG.image_size,
        CFG.circle_mask_ratio
    )

    return image


def load_aptos():

    csv_path = os.path.join(
        CFG.aptos_dir,
        "train.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            csv_path
        )

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "id_code": "image_id",
            "diagnosis": "label"
        }
    )

    df["image_path"] = df[
        "image_id"
    ].apply(
        lambda x:
        os.path.join(
            CFG.aptos_dir,
            "train_images",
            f"{x}.png"
        )
    )

    df["source"] = "aptos"

    return df[
        [
            "image_path",
            "label",
            "source"
        ]
    ]


def load_eyepacs():

    csv_path = os.path.join(
        CFG.eyepacs_dir,
        "trainLabels.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            csv_path
        )

    df = pd.read_csv(csv_path)

    df = df.rename(
        columns={
            "image": "image_id",
            "level": "label"
        }
    )

    print(
        f"EyePACS original: {len(df)}"
    )

    files = []

    for root, _, filenames in os.walk(
        CFG.eyepacs_dir
    ):

        for f in filenames:

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                files.append(
                    os.path.join(
                        root,
                        f
                    )
                )

    file_map = {
        os.path.splitext(
            os.path.basename(f)
        )[0]: f
        for f in files
    }

    df["image_path"] = df[
        "image_id"
    ].map(file_map)

    missing = df[
        "image_path"
    ].isna().sum()

    print(
        f"EyePACS missing: {missing}"
    )

    df = df.dropna(
        subset=["image_path"]
    ).copy()

    df = (
        df
        .groupby(
            "label",
            group_keys=False
        )
        .apply(
            lambda x:
            x.sample(
                min(
                    len(x),
                    CFG.max_eyepacs_per_class
                ),
                random_state=CFG.seed
            )
        )
        .reset_index(drop=True)
    )

    df["source"] = "eyepacs"

    print(
        f"EyePACS used: {len(df)}"
    )

    return df[
        [
            "image_path",
            "label",
            "source"
        ]
    ]


print("\nLoading datasets...")

aptos = load_aptos()
eyepacs = load_eyepacs()

df = pd.concat(
    [
        aptos,
        eyepacs
    ],
    ignore_index=True
)


exists_mask = df[
    "image_path"
].apply(os.path.isfile)

print(
    f"Missing paths removed: "
    f"{(~exists_mask).sum()}"
)

df = df[
    exists_mask
].reset_index(drop=True)


df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df[
    df["label"].between(0, 4)
].copy()

df["label"] = df[
    "label"
].astype(int)


print(
    f"\nTotal images: {len(df)}"
)

print(
    "\nClass distribution:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)


print(
    "\nSource distribution:"
)

print(
    df.groupby(
        ["source", "label"]
    ).size()
)


df["stratify_key"] = (
    df["source"].astype(str)
    + "_"
    + df["label"].astype(str)
)


train_tf = A.Compose([

    A.HorizontalFlip(
        p=CFG.horizontal_flip_p
    ),

    A.Affine(
        scale=(0.95, 1.05),
        translate_percent=(
            -0.02,
            0.02
        ),
        rotate=(-15, 15),
        shear=(-3, 3),
        p=CFG.affine_p
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=CFG.brightness_p
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.10
    ),

    A.Normalize(),

    ToTensorV2()
])


val_tf = A.Compose([
    A.Normalize(),
    ToTensorV2()
])


class DRDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["image_path"]

        label = int(row["label"])

        try:

            image = cv2.imread(path)

            if image is None:

                raise RuntimeError(
                    "cv2.imread returned None"
                )

            image = ben_graham_processing(
                image
            )

            image = self.transform(
                image=image
            )["image"]

            return image, torch.tensor(
                label,
                dtype=torch.long
            )

        except Exception as e:
            print(
                f"\n[ERROR] Failed image:"
                f"\n{path}"
                f"\nReason: {e}"
            )

            dummy = np.zeros(
                (
                    CFG.image_size,
                    CFG.image_size,
                    3
                ),
                dtype=np.uint8
            )

            dummy = self.transform(
                image=dummy
            )["image"]

            return dummy, torch.tensor(
                label,
                dtype=torch.long
            )


class DRModel(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        self.backbone = timm.create_model(
            CFG.model_name,
            pretrained=pretrained,
            num_classes=0
        )

        features = self.backbone.num_features

        self.dropout = nn.Dropout(
            CFG.dropout
        )

        self.classifier = nn.Linear(
            features,
            CFG.num_classes
        )

        self.ordinal_head = nn.Linear(
            features,
            CFG.num_classes - 1
        )

    def forward(self, x):

        features = self.backbone(x)

        features = self.dropout(
            features
        )

        class_logits = self.classifier(
            features
        )

        ordinal_logits = self.ordinal_head(
            features
        )

        return (
            class_logits,
            ordinal_logits
        )


def make_ordinal_targets(
    labels
):


    thresholds = torch.arange(
        CFG.num_classes - 1,
        device=labels.device
    )

    return (
        labels.unsqueeze(1)
        > thresholds
    ).float()


class DRLoss(nn.Module):

    def __init__(
        self,
        class_weights=None
    ):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=
                CFG.label_smoothing
        )

        self.ordinal = nn.BCEWithLogitsLoss()

    def forward(
        self,
        class_logits,
        ordinal_logits,
        targets
    ):

        ce_loss = self.ce(
            class_logits,
            targets
        )

        ordinal_targets = (
            make_ordinal_targets(
                targets
            )
        )

        ordinal_loss = self.ordinal(
            ordinal_logits,
            ordinal_targets
        )

        total = (
            CFG.ce_weight * ce_loss
            +
            CFG.ordinal_weight * ordinal_loss
        )

        return (
            total,
            ce_loss.detach(),
            ordinal_loss.detach()
        )


class ModelEMA:

    def __init__(
        self,
        model,
        decay=0.999
    ):

        self.decay = decay
        self.ema = copy.deepcopy(model).to(
            CFG.device
        )

        self.ema.eval()

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):

        ema_state = (
            self.ema.state_dict()
        )

        model_state = (
            model.state_dict()
        )

        for key in ema_state.keys():

            if ema_state[key].dtype.is_floating_point:

                ema_state[key].mul_(
                    self.decay
                ).add_(
                    model_state[key],
                    alpha=1 - self.decay
                )

            else:

                ema_state[key].copy_(
                    model_state[key]
                )


def get_class_weights(train_df):

    if not CFG.use_class_weights:
        return None

    classes = np.arange(
        CFG.num_classes
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"]
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=CFG.device
    )

    print(
        "Class weights:",
        weights.cpu().numpy()
    )

    return weights


def metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic"
    )

    return (
        accuracy,
        macro_f1,
        qwk
    )


@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0
    batches = 0

    for images, labels in loader:

        images = images.to(
            CFG.device,
            non_blocking=True
        )

        labels = labels.to(
            CFG.device,
            non_blocking=True
        )

        class_logits, ordinal_logits = (
            model(images)
        )

        loss, _, _ = criterion(
            class_logits,
            ordinal_logits,
            labels
        )

        preds = class_logits.argmax(
            dim=1
        )

        total_loss += loss.item()
        batches += 1

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    acc, f1, qwk = metrics(
        all_labels,
        all_preds
    )

    return (
        total_loss / max(batches, 1),
        acc,
        f1,
        qwk,
        np.array(all_labels),
        np.array(all_preds)
    )


@torch.no_grad()
def predict_tta(
    model,
    images
):

    outputs = []

    # Original
    logits, _ = model(images)
    outputs.append(
        torch.softmax(
            logits,
            dim=1
        )
    )

    if CFG.use_tta:

        flipped = torch.flip(
            images,
            dims=[3]
        )

        logits, _ = model(flipped)

        outputs.append(
            torch.softmax(
                logits,
                dim=1
            )
        )

        flipped = torch.flip(
            images,
            dims=[2]
        )

        logits, _ = model(flipped)

        outputs.append(
            torch.softmax(
                logits,
                dim=1
            )
        )

        flipped = torch.flip(
            images,
            dims=[2, 3]
        )

        logits, _ = model(flipped)

        outputs.append(
            torch.softmax(
                logits,
                dim=1
            )
        )

    return torch.stack(
        outputs
    ).mean(dim=0)


def train_fold(
    fold,
    train_df,
    val_df
):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold + 1}")
    print("=" * 70)

    train_dataset = DRDataset(
        train_df,
        train_tf
    )

    val_dataset = DRDataset(
        val_df,
        val_tf
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=(
            CFG.num_workers > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=(
            CFG.num_workers > 0
        )
    )

    model = DRModel().to(
        CFG.device
    )

    backbone_params = list(
        model.backbone.parameters()
    )

    head_params = list(
        model.classifier.parameters()
    ) + list(
        model.ordinal_head.parameters()
    )

    print(
        "Backbone params:",
        sum(
            p.numel()
            for p in backbone_params
        )
    )

    print(
        "Head params:",
        sum(
            p.numel()
            for p in head_params
        )
    )


    class_weights = (
        get_class_weights(
            train_df
        )
    )

    criterion = DRLoss(
        class_weights
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_params,
                "lr": CFG.lr_head
            },
            {
                "params": backbone_params,
                "lr": CFG.lr_backbone
            }
        ],
        weight_decay=CFG.weight_decay
    )

    def lr_lambda(epoch):

        if epoch < CFG.warmup_epochs:

            return (
                (epoch + 1)
                / CFG.warmup_epochs
            )

        progress = (
            epoch - CFG.warmup_epochs
        ) / max(
            CFG.epochs - CFG.warmup_epochs,
            1
        )

        return (
            0.5 *
            (
                1 +
                np.cos(
                    np.pi * progress
                )
            )
        )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda
        )
    )

    use_amp = (
        CFG.device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    ema = None

    if CFG.use_ema:

        ema = ModelEMA(
            model,
            CFG.ema_decay
        )

    best_qwk = -999

    patience_counter = 0

    checkpoint_path = os.path.join(
        CFG.output_dir,
        f"fold_{fold}.pth"
    )

    y_true, y_pred = np.array([]), np.array([])

    for epoch in range(
        CFG.epochs
    ):

        model.train()

        start = time.time()

        running_loss = 0
        valid_steps = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Fold {fold + 1} "
                f"Epoch {epoch + 1}"
            )
        )

        for images, labels in pbar:

            images = images.to(
                CFG.device,
                non_blocking=True
            )

            labels = labels.to(
                CFG.device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            try:

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp
                ):

                    class_logits, ordinal_logits = (
                        model(images)
                    )

                    loss, ce_loss, ordinal_loss = (
                        criterion(
                            class_logits,
                            ordinal_logits,
                            labels
                        )
                    )

                if not torch.isfinite(loss):

                    print(
                        "\n[WARNING] "
                        "Non-finite loss skipped."
                    )

                    continue

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CFG.grad_clip
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                if ema is not None:

                    ema.update(
                        model
                    )

                running_loss += (
                    loss.item()
                )

                valid_steps += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            except RuntimeError as e:

                if "out of memory" in str(
                    e
                ).lower():

                    print(
                        "\n[ERROR] "
                        "CUDA OOM."
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    continue

                raise e

        scheduler.step()

        train_loss = (
            running_loss /
            max(valid_steps, 1)
        )

        eval_model = (
            ema.ema
            if ema is not None
            else model
        )

        val_loss, val_acc, val_f1, val_qwk, y_true, y_pred = (
            validate(
                eval_model,
                val_loader,
                criterion
            )
        )

        elapsed = time.time() - start

        current_lr_head = (
            optimizer.param_groups[0]["lr"]
        )

        current_lr_backbone = (
            optimizer.param_groups[1]["lr"]
        )

        print(
            f"\nEpoch {epoch + 1}/{CFG.epochs}"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val Acc    : {val_acc:.4f}"
        )

        print(
            f"Val MacroF1: {val_f1:.4f}"
        )

        print(
            f"Val QWK    : {val_qwk:.4f}"
        )

        print(
            f"LR Head    : {current_lr_head:.2e}"
        )

        print(
            f"LR Backbone: {current_lr_backbone:.2e}"
        )

        print(
            f"Time       : {elapsed:.1f}s"
        )


        if val_qwk > best_qwk:

            best_qwk = val_qwk

            patience_counter = 0

            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict":
                        eval_model.state_dict(),
                    "best_qwk":
                        best_qwk,
                    "config":
                        {
                            k: v
                            for k, v
                            in CFG.__dict__.items()
                            if not k.startswith("__")
                        }
                },
                checkpoint_path
            )

            print(
                "[CHECKPOINT] "
                "Best QWK saved."
            )

        else:

            patience_counter += 1

            print(
                f"[EARLY STOP] "
                f"{patience_counter}/"
                f"{CFG.patience}"
            )

            if (
                patience_counter
                >= CFG.patience
            ):

                print(
                    "Early stopping."
                )

                break


    if len(y_true) > 0:

        print("\nFold confusion matrix:")

        print(
            confusion_matrix(
                y_true,
                y_pred
            )
        )

        print(
            "\nClassification report:"
        )

        print(
            classification_report(
                y_true,
                y_pred,
                digits=4,
                zero_division=0
            )
        )

    del model
    del ema

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return checkpoint_path, best_qwk


skf = StratifiedKFold(
    n_splits=CFG.n_folds,
    shuffle=True,
    random_state=CFG.seed
)

df["fold"] = -1

for fold, (_, val_idx) in enumerate(
    skf.split(
        df,
        df["stratify_key"]
    )
):

    df.loc[
        val_idx,
        "fold"
    ] = fold



fold_results = []

for fold in range(
    CFG.run_folds
):

    train_df = df[
        df["fold"] != fold
    ].reset_index(
        drop=True
    )

    val_df = df[
        df["fold"] == fold
    ].reset_index(
        drop=True
    )

    print(
        f"\nFold {fold + 1}"
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Valid: {len(val_df)}"
    )

    checkpoint, qwk = train_fold(
        fold,
        train_df,
        val_df
    )

    fold_results.append(
        {
            "fold": fold,
            "checkpoint": checkpoint,
            "qwk": qwk
        }
    )


results_df = pd.DataFrame(
    fold_results
)

results_df.to_csv(
    os.path.join(
        CFG.output_dir,
        "training_summary.csv"
    ),
    index=False
)

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(results_df)


actual_folds = min(CFG.run_folds, CFG.n_folds)

if actual_folds == CFG.n_folds:

    print("\n" + "=" * 70)
    print("FINAL OUT-OF-FOLD EVALUATION (EMA weights + TTA)")
    print("=" * 70)

    oof_preds = np.full(len(df), -1, dtype=np.int32)
    oof_labels = df["label"].values.astype(np.int32)

    for fold in range(CFG.n_folds):

        checkpoint_path = os.path.join(
            CFG.output_dir,
            f"fold_{fold}.pth"
        )

        if not os.path.exists(checkpoint_path):

            print(f"[WARNING] Missing checkpoint for fold {fold}.")

            continue

        print(f"\n[INFO] OOF fold {fold}")

        try:

            val_df = df[df["fold"] == fold].reset_index(drop=True)

            val_dataset = DRDataset(val_df, val_tf)

            val_loader = DataLoader(
                val_dataset,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            ckpt = torch.load(
                checkpoint_path,
                map_location=CFG.device
            )

            model = DRModel(pretrained=False).to(CFG.device)

            model.load_state_dict(ckpt["model_state_dict"])

            model.eval()

            fold_preds = []

            with torch.no_grad():

                for images, _ in tqdm(
                    val_loader,
                    desc=f"OOF Fold {fold}"
                ):

                    images = images.to(
                        CFG.device,
                        non_blocking=True
                    )

                    with torch.amp.autocast(
                        device_type=CFG.device.type,
                        enabled=CFG.device.type == "cuda"
                    ):

                        probs = predict_tta(model, images)

                    fold_preds.extend(
                        probs.argmax(dim=1).cpu().numpy()
                    )

            oof_preds[
                df["fold"].values == fold
            ] = np.array(fold_preds)

            del model
            del val_loader
            del val_dataset

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:

            print(f"[ERROR] OOF fold {fold} failed:")
            print(str(e))

    valid_mask = oof_preds >= 0

    if valid_mask.sum() > 0:

        final_labels = oof_labels[valid_mask]
        final_preds = oof_preds[valid_mask]

        final_accuracy = accuracy_score(final_labels, final_preds)

        final_f1 = f1_score(
            final_labels,
            final_preds,
            average="macro",
            zero_division=0
        )

        final_qwk = cohen_kappa_score(
            final_labels,
            final_preds,
            weights="quadratic"
        )

        print("\n" + "=" * 70)
        print("FINAL OOF RESULTS")
        print("=" * 70)

        print(f"Accuracy : {final_accuracy:.4f}")
        print(f"Macro-F1 : {final_f1:.4f}")
        print(f"QWK      : {final_qwk:.4f}")

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                final_labels,
                final_preds,
                labels=[0, 1, 2, 3, 4]
            )
        )

        print("\nClassification Report:")

        print(
            classification_report(
                final_labels,
                final_preds,
                labels=[0, 1, 2, 3, 4],
                zero_division=0
            )
        )

    else:

        print("[ERROR] No valid OOF predictions.")

else:

    print("\n" + "=" * 70)
    print("BENCHMARK MODE - OOF evaluation skipped")
    print("(you did not train all n_folds; this run is for")
    print("timing / correctness checking only)")
    print("=" * 70)