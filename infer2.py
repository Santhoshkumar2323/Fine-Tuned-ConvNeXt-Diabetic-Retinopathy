"""
=============================================================
  DIABETIC RETINOPATHY INFERENCE SCRIPT
  Model  : ConvNeXt-Tiny (5-Fold Ensemble + 4-Step TTA)
  Device : CPU
  Extras : Uncertainty Gating + Grad-CAM heatmaps
  Output : One folder per run — no overwriting ever
=============================================================
"""

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
    SAVE_GRADCAM      = True        
    USE_TTA           = True        
    UNCERTAINTY_THRESHOLD = 60      
    device = torch.device("cpu")

GRADE_INFO = {
    0: {
        "label"  : "No DR",
        "stage"  : "Stage 0",
        "meaning": "No signs of diabetic retinopathy",
        "action" : "Routine annual screening recommended",
    },
    1: {
        "label"  : "Mild DR",
        "stage"  : "Stage 1",
        "meaning": "Small microaneurysms only",
        "action" : "Monitor — follow up in 12 months",
    },
    2: {
        "label"  : "Moderate DR",
        "stage"  : "Stage 2",
        "meaning": "More than microaneurysms, less than severe",
        "action" : "Refer to ophthalmologist — follow up in 6 months",
    },
    3: {
        "label"  : "Severe DR",
        "stage"  : "Stage 3",
        "meaning": "Extensive hemorrhages, venous beading present",
        "action" : "Urgent ophthalmology referral required",
    },
    4: {
        "label"  : "Proliferative DR",
        "stage"  : "Stage 4",
        "meaning": "Neovascularization — advanced stage",
        "action" : "Immediate intervention needed",
    },
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

val_tf = A.Compose([A.Normalize(), ToTensorV2()])

def confidence_tag(conf):
    if conf >= 75:   return "✅ High confidence"
    elif conf >= 60: return "⚠️  Moderate confidence"
    else:            return "❌ Low confidence"

def load_all_models():
    weight_files = sorted([
        f for f in os.listdir(CFG.weights_dir) if f.endswith(".pth")
    ])
    if not weight_files:
        raise FileNotFoundError(
            f"[ERROR] No .pth files in '{CFG.weights_dir}/'"
        )

    print(f"\n{'='*54}")
    print(f"  LOADING {len(weight_files)} MODEL WEIGHTS")
    print(f"{'='*54}")
    models = []
    for wf in weight_files:
        path  = os.path.join(CFG.weights_dir, wf)
        model = timm.create_model(
            CFG.model_name, pretrained=False, num_classes=CFG.num_classes
        )
        model.load_state_dict(torch.load(path, map_location=CFG.device))
        model.to(CFG.device)
        model.eval()
        models.append(model)
        print(f"  ✅ Loaded : {wf}")
    print(f"{'='*54}\n")
    return models


def generate_gradcam(model, x, target_class, orig_img_rgb):
    gradients  = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.stages[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(x)
    output[0, target_class].backward()

    fh.remove()
    bh.remove()

    grads   = gradients[0].detach().cpu().numpy()[0]
    acts    = activations[0].detach().cpu().numpy()[0]
    weights = grads.mean(axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_resized = cv2.resize(cam, (CFG.image_size, CFG.image_size))
    heatmap     = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    orig_bgr = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR)
    orig_bgr = cv2.resize(orig_bgr, (CFG.image_size, CFG.image_size))
    blended  = cv2.addWeighted(orig_bgr, 0.55, heatmap, 0.45, 0)
    return blended


def predict_single(image_path, models):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None, None, None, None, f"Cannot read: {image_path}"

    img_processed = ben_graham_processing(img_bgr)
    tensor        = val_tf(image=img_processed)["image"]
    x             = tensor.unsqueeze(0).to(CFG.device)

    all_probs = []
    with torch.no_grad():
        for model in models:
            if CFG.USE_TTA:
                x_hf = torch.flip(x, [3])
                x_vf = torch.flip(x, [2])
                x_hv = torch.flip(x, [2, 3])
                o1   = torch.softmax(model(x),    dim=1)
                o2   = torch.softmax(model(x_hf), dim=1)
                o3   = torch.softmax(model(x_vf), dim=1)
                o4   = torch.softmax(model(x_hv), dim=1)
                avg  = (o1 + o2 + o3 + o4) / 4.0
            else:
                avg  = torch.softmax(model(x), dim=1)
            all_probs.append(avg.cpu().numpy())

    ensemble_probs  = np.mean(all_probs, axis=0)[0]
    predicted_grade = int(np.argmax(ensemble_probs))
    confidence      = float(ensemble_probs[predicted_grade]) * 100
    is_uncertain    = confidence < CFG.UNCERTAINTY_THRESHOLD
    probs_dict      = {
        i: round(float(ensemble_probs[i]) * 100, 1)
        for i in range(CFG.num_classes)
    }

    gradcam_img = None
    if CFG.SAVE_GRADCAM:
        try:
            x_grad      = tensor.unsqueeze(0).to(CFG.device).requires_grad_(True)
            gradcam_img = generate_gradcam(
                models[0], x_grad, predicted_grade, img_processed
            )
        except Exception:
            gradcam_img = None

    return predicted_grade, confidence, probs_dict, is_uncertain, gradcam_img, None

def print_result(idx, fname, grade, conf, probs, is_uncertain):
    W    = 54
    info = GRADE_INFO[grade]
    tag  = confidence_tag(conf)

    print(f"\n  {'─'*W}")
    print(f"  IMAGE [{idx}] : {fname}")
    print(f"  {'─'*W}")

    if is_uncertain:
        print(f"  ⚠️  STATUS    : UNCERTAIN — REFER FOR MANUAL REVIEW")
        print(f"  ► Best guess : {info['stage']} — {info['label']}")
        print(f"  ► CONF       : {conf:.1f}%  {tag}")
        print(f"  ► NOTE       : Below {CFG.UNCERTAINTY_THRESHOLD}% threshold")
        print(f"                 Do not rely on this prediction alone")
    else:
        print(f"  ✅ STATUS    : CONFIDENT PREDICTION")
        print(f"  ► RESULT    : {info['stage']} — {info['label']}")
        print(f"  ► MEANING   : {info['meaning']}")
        print(f"  ► ACTION    : {info['action']}")
        print(f"  ► CONF      : {conf:.1f}%  {tag}")

    print(f"  {'─'*W}")
    print(f"  PROBABILITY BREAKDOWN:")
    for i in range(CFG.num_classes):
        g    = GRADE_INFO[i]
        p    = probs[i]
        bar  = "█" * int(p / 5)
        mark = "  ◄ PREDICTED" if i == grade else ""
        print(f"    {g['stage']} {g['label']:<18}: {p:>5.1f}%  {bar}{mark}")
    print(f"  {'─'*W}")

def main():
    run_start = time.time()

    # --- Create unique run folder ---
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_folder = os.path.join(CFG.results_dir, f"run_{timestamp}")
    gradcam_folder = os.path.join(run_folder, "gradcam")

    os.makedirs(run_folder, exist_ok=True)
    if CFG.SAVE_GRADCAM:
        os.makedirs(gradcam_folder, exist_ok=True)

    print(f"\n{'='*54}")
    print(f"  DIABETIC RETINOPATHY INFERENCE")
    print(f"  Date       : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Device     : CPU")
    print(f"  TTA        : {'ON — 4-step' if CFG.USE_TTA else 'OFF'}")
    print(f"  Uncertain  : flag if confidence < {CFG.UNCERTAINTY_THRESHOLD}%")
    print(f"  Grad-CAM   : {'ON' if CFG.SAVE_GRADCAM else 'OFF'}")
    print(f"  Images     : {CFG.images_dir}/")
    print(f"  Weights    : {CFG.weights_dir}/")
    print(f"  Run folder : {run_folder}/")
    print(f"{'='*54}")

    valid_exts  = (".jpg", ".jpeg", ".png")
    image_files = sorted([
        f for f in os.listdir(CFG.images_dir)
        if f.lower().endswith(valid_exts)
    ])
    if not image_files:
        print(f"\n  [ERROR] No images in '{CFG.images_dir}/'")
        return

    print(f"\n  Found {len(image_files)} image(s) to process.")
    models  = load_all_models()
    results = []
    skipped = 0
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(CFG.images_dir, fname)

        grade, conf, probs, is_uncertain, gradcam_img, err = predict_single(
            img_path, models
        )

        if err:
            print(f"\n  [{idx}] ❌ ERROR — {err}")
            skipped += 1
            continue
        if CFG.SAVE_GRADCAM and gradcam_img is not None:
            base      = os.path.splitext(fname)[0]
            gcam_path = os.path.join(gradcam_folder, f"{base}_gradcam.jpg")
            cv2.imwrite(gcam_path, gradcam_img)

        if CFG.PRINT_TO_TERMINAL:
            print_result(idx, fname, grade, conf, probs, is_uncertain)

        info = GRADE_INFO[grade]
        results.append({
            "image"                       : fname,
            "grade"                       : grade,
            "stage"                       : info["stage"],
            "label"                       : info["label"],
            "confidence_%"                : round(conf, 2),
            "confidence_level"            : confidence_tag(conf).replace("✅ ", "").replace("⚠️  ", "").replace("❌ ", ""),
            "uncertain_flag"              : "YES" if is_uncertain else "NO",
            "prob_Stage0_NoDR_%"          : probs[0],
            "prob_Stage1_Mild_%"          : probs[1],
            "prob_Stage2_Moderate_%"      : probs[2],
            "prob_Stage3_Severe_%"        : probs[3],
            "prob_Stage4_Proliferative_%" : probs[4],
        })
    elapsed         = time.time() - run_start
    uncertain_count = sum(1 for r in results if r["uncertain_flag"] == "YES")

    print(f"\n  {'='*54}")
    print(f"  {'INFERENCE COMPLETE':^54}")
    print(f"  {'='*54}")
    print(f"  Total images   : {len(image_files)}")
    print(f"  Processed      : {len(results)}")
    print(f"  Skipped        : {skipped}")
    print(f"  Uncertain      : {uncertain_count}  (flagged for review)")
    print(f"  Time elapsed   : {elapsed:.1f}s  "
          f"({elapsed/max(len(results),1):.1f}s per image)")

    if results:
        df_out = pd.DataFrame(results)
        print(f"\n  GRADE DISTRIBUTION:")
        print(f"  {'─'*44}")
        for g in range(CFG.num_classes):
            count = len(df_out[df_out["grade"] == g])
            if count > 0:
                bar  = "█" * count
                info = GRADE_INFO[g]
                print(f"  {info['stage']} — {info['label']:<22}: {count:>2}  {bar}")
        print(f"  {'─'*44}")

        if uncertain_count > 0:
            print(f"\n  ⚠️  UNCERTAIN — NEEDS MANUAL REVIEW:")
            print(f"  {'─'*44}")
            for r in results:
                if r["uncertain_flag"] == "YES":
                    print(f"    → {r['image']}")
                    print(f"       Best guess : {r['stage']} — {r['label']}")
                    print(f"       Confidence : {r['confidence_%']}%")
            print(f"  {'─'*44}")

    # --- Save CSV inside run folder ---
    if CFG.SAVE_CSV and results:
        csv_path = os.path.join(run_folder, f"inference_{timestamp}.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n  ✅ CSV     saved → {csv_path}")

    if CFG.SAVE_GRADCAM:
        print(f"  ✅ Grad-CAM saved → {gradcam_folder}/")

    print(f"\n  📁 Full run saved → {run_folder}/")
    print(f"  {'='*54}\n")

if __name__ == "__main__":
    main()