import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report
)

class CFG:
    image_size = 448
    num_classes = 5
    batch_size = 16
    num_workers = 2
    dropout = 0.20
    crop_tol = 7
    circle_mask_ratio = 0.98
    use_tta = True
    model_name = "convnext_tiny.fb_in22k_ft_in1k"
    output_dir = "/kaggle/input/notebooks/santhosh2323/fine-tune-vision/dr_v3"
    n_folds = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aptos_dir = "/kaggle/input/datasets/shiva8227/aptos2019-blindness-detection"
    eyepacs_dir = "/kaggle/input/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized"
    max_eyepacs_per_class = 2000
    seed = 42


def crop_image_from_gray(image, tol=7):
    if image.ndim == 2:
        mask = image > tol
        if mask.sum() == 0:
            return image
        return image[np.ix_(mask.any(1), mask.any(0))]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return image
    rows, cols = mask.any(1), mask.any(0)
    if rows.sum() == 0 or cols.sum() == 0:
        return image
    return image[np.ix_(rows, cols)]


def apply_circle_mask(image, size, ratio=0.98):
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.5 * ratio)
    cv2.circle(mask, center, radius, 1, thickness=-1)
    return (image * mask[..., None]).astype(np.uint8)


def ben_graham_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image, CFG.crop_tol)
    image = cv2.resize(image, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_AREA)
    ksize = int(CFG.image_size / 10)
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(image, (ksize, ksize), 0)
    image = cv2.addWeighted(image, 4, blur, -4, 128)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = apply_circle_mask(image, CFG.image_size, CFG.circle_mask_ratio)
    return image


val_tf = A.Compose([A.Normalize(), ToTensorV2()])


class DRDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path, label = row["image_path"], int(row["label"])
        try:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError("cv2.imread returned None")
            image = ben_graham_processing(image)
            image = self.transform(image=image)["image"]
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            dummy = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
            dummy = self.transform(image=dummy)["image"]
            return dummy, torch.tensor(label, dtype=torch.long)


class DRModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(CFG.model_name, pretrained=pretrained, num_classes=0)
        features = self.backbone.num_features
        self.dropout = nn.Dropout(CFG.dropout)
        self.classifier = nn.Linear(features, CFG.num_classes)
        self.ordinal_head = nn.Linear(features, CFG.num_classes - 1)

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        return self.classifier(f), self.ordinal_head(f)


@torch.no_grad()
def predict_tta(model, images):
    outputs = []
    logits, _ = model(images)
    outputs.append(torch.softmax(logits, dim=1))
    if CFG.use_tta:
        for dims in ([3], [2], [2, 3]):
            logits, _ = model(torch.flip(images, dims=dims))
            outputs.append(torch.softmax(logits, dim=1))
    return torch.stack(outputs).mean(dim=0)


def load_aptos():
    df = pd.read_csv(os.path.join(CFG.aptos_dir, "train.csv"))
    df = df.rename(columns={"id_code": "image_id", "diagnosis": "label"})
    df["image_path"] = df["image_id"].apply(
        lambda x: os.path.join(CFG.aptos_dir, "train_images", f"{x}.png"))
    df["source"] = "aptos"
    return df[["image_path", "label", "source"]]


def load_eyepacs():
    df = pd.read_csv(os.path.join(CFG.eyepacs_dir, "trainLabels.csv"))
    df = df.rename(columns={"image": "image_id", "level": "label"})
    files = []
    for root, _, filenames in os.walk(CFG.eyepacs_dir):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(root, f))
    file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    df["image_path"] = df["image_id"].map(file_map)
    df = df.dropna(subset=["image_path"]).copy()
    df = (df.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), CFG.max_eyepacs_per_class), random_state=CFG.seed))
            .reset_index(drop=True))
    df["source"] = "eyepacs"
    return df[["image_path", "label", "source"]]


print("Rebuilding dataframe + fold split...")

aptos = load_aptos()
eyepacs = load_eyepacs()
df = pd.concat([aptos, eyepacs], ignore_index=True)
df = df[df["image_path"].apply(os.path.isfile)].reset_index(drop=True)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].between(0, 4)].copy()
df["label"] = df["label"].astype(int)
df["stratify_key"] = df["source"].astype(str) + "_" + df["label"].astype(str)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df["stratify_key"])):
    df.loc[val_idx, "fold"] = fold

print(f"Total images: {len(df)}")


oof_preds = np.full(len(df), -1, dtype=np.int32)
oof_labels = df["label"].values.astype(np.int32)

for fold in range(CFG.n_folds):

    checkpoint_path = os.path.join(CFG.output_dir, f"fold_{fold}.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] Missing checkpoint for fold {fold}.")
        continue

    print(f"\n[INFO] OOF fold {fold}")

    val_df = df[df["fold"] == fold].reset_index(drop=True)
    val_dataset = DRDataset(val_df, val_tf)
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available()
    )

    ckpt = torch.load(checkpoint_path, map_location=CFG.device, weights_only=False)

    model = DRModel(pretrained=False).to(CFG.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  checkpoint best_qwk (from training): {ckpt.get('best_qwk')}")

    fold_preds = []
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc=f"OOF Fold {fold}"):
            images = images.to(CFG.device, non_blocking=True)
            with torch.amp.autocast(device_type=CFG.device.type, enabled=CFG.device.type == "cuda"):
                probs = predict_tta(model, images)
            fold_preds.extend(probs.argmax(dim=1).cpu().numpy())

    oof_preds[df["fold"].values == fold] = np.array(fold_preds)

    del model, val_loader, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

valid_mask = oof_preds >= 0

if valid_mask.sum() > 0:
    final_labels = oof_labels[valid_mask]
    final_preds = oof_preds[valid_mask]

    acc = accuracy_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds, average="macro", zero_division=0)
    qwk = cohen_kappa_score(final_labels, final_preds, weights="quadratic")

    print("\n" + "=" * 70)
    print("FINAL OOF RESULTS")
    print("=" * 70)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"QWK      : {qwk:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(final_labels, final_preds, labels=[0, 1, 2, 3, 4]))
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, labels=[0, 1, 2, 3, 4], zero_division=0))
else:
    print("[ERROR] No valid OOF predictions - check checkpoint paths / dataset paths.")