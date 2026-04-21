import os, cv2, time
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from datetime import datetime

class CFG:
    weights_dir   = "weights"          
    images_dir    = "images"           
    results_dir   = "results"          
    image_size    = 384
    num_classes   = 5
    model_name    = "convnext_tiny.fb_in22k_ft_in1k"

    PRINT_TO_TERMINAL = True           
    SAVE_CSV          = True           
    USE_TTA           = True           

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

def ben_graham_processing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CFG.image_size, CFG.image_size))
    ksize = int(CFG.image_size / 10)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    image = cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (ksize, ksize), 0),
        -4, 128
    )
    return image

val_tf = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

def load_all_models():
    """Automatically finds and loads all .pth files from weights_dir."""
    weight_files = sorted([
        f for f in os.listdir(CFG.weights_dir) if f.endswith(".pth")
    ])

    if len(weight_files) == 0:
        raise FileNotFoundError(f"[ERROR] No .pth files found in '{CFG.weights_dir}/'")

    print(f"\n{'='*55}")
    print(f"  LOADING {len(weight_files)} MODEL WEIGHTS")
    print(f"{'='*55}")

    models = []
    for wf in weight_files:
        path = os.path.join(CFG.weights_dir, wf)
        model = timm.create_model(
            CFG.model_name, pretrained=False, num_classes=CFG.num_classes
        )
        model.load_state_dict(torch.load(path, map_location=CFG.device))
        model.to(CFG.device)
        model.eval()
        models.append(model)
        print(f"  ✅ Loaded: {wf}")

    print(f"{'='*55}\n")
    return models

def predict_single(image_path, models):
    """Run ensemble + TTA inference on one image."""

    img = cv2.imread(image_path)
    if img is None:
        return None, None, f"[WARN] Could not read image: {image_path}"

    img = ben_graham_processing(img)
    tensor = val_tf(image=img)["image"]
    x = tensor.unsqueeze(0).to(CFG.device)   # shape: [1, 3, H, W]

    all_probs = []

    with torch.no_grad():
        for model in models:
            if CFG.USE_TTA:
                # 4 TTA passes
                x_hflip  = torch.flip(x, [3])
                x_vflip  = torch.flip(x, [2])
                x_hvflip = torch.flip(x, [2, 3])

                out1 = torch.softmax(model(x),        dim=1)
                out2 = torch.softmax(model(x_hflip),  dim=1)
                out3 = torch.softmax(model(x_vflip),  dim=1)
                out4 = torch.softmax(model(x_hvflip), dim=1)

                avg = (out1 + out2 + out3 + out4) / 4.0
            else:
                avg = torch.softmax(model(x), dim=1)

            all_probs.append(avg.cpu().numpy())

    ensemble_probs = np.mean(all_probs, axis=0)[0]   # shape: [5]
    predicted_grade = int(np.argmax(ensemble_probs))
    confidence = float(ensemble_probs[predicted_grade]) * 100

    return predicted_grade, confidence, None

def main():
    run_start = time.time()

    print(f"\n{'='*55}")
    print(f"  DIABETIC RETINOPATHY INFERENCE")
    print(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device   : {CFG.device}")
    print(f"  TTA      : {'ON (4-step)' if CFG.USE_TTA else 'OFF'}")
    print(f"  Images   : {CFG.images_dir}/")
    print(f"  Weights  : {CFG.weights_dir}/")
    print(f"{'='*55}")

    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = sorted([
        f for f in os.listdir(CFG.images_dir)
        if f.lower().endswith(valid_exts)
    ])

    if len(image_files) == 0:
        print(f"[ERROR] No images found in '{CFG.images_dir}/'")
        return

    print(f"\n  Found {len(image_files)} images to process.\n")
   
    models = load_all_models()
    results = []
    skipped = 0

    if CFG.PRINT_TO_TERMINAL:
        print(f"{'='*55}")
        print(f"  {'IMAGE':<35} {'GRADE':<5} {'LABEL':<20} {'CONF':>6}")
        print(f"{'='*55}")

    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(CFG.images_dir, fname)
        grade, conf, err = predict_single(img_path, models)

        if err:
            print(f"  [{idx:>4}] {err}")
            skipped += 1
            continue

        label = GRADE_LABELS[grade]

        results.append({
            "image":      fname,
            "grade":      grade,
            "label":      label,
            "confidence": round(conf, 2)
        })

        if CFG.PRINT_TO_TERMINAL:
            print(f"  [{idx:>4}] {fname:<35} {grade:<5} {label:<20} {conf:>5.1f}%")

    elapsed = time.time() - run_start
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Total images   : {len(image_files)}")
    print(f"  Processed      : {len(results)}")
    print(f"  Skipped/errors : {skipped}")
    print(f"  Time elapsed   : {elapsed:.1f}s")
    print(f"  Avg per image  : {elapsed/max(len(results),1):.2f}s")

    if results:
        df = pd.DataFrame(results)
        grade_counts = df["grade"].value_counts().sort_index()
        print(f"\n  Grade Distribution:")
        for g, count in grade_counts.items():
            print(f"    Grade {g} ({GRADE_LABELS[g]}): {count} images")
            
    if CFG.SAVE_CSV and results:
        os.makedirs(CFG.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        csv_name = f"inference_{timestamp}.csv"
        csv_path = os.path.join(CFG.results_dir, csv_name)
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n CSV saved → {csv_path}")

    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()