# Retinopathy-AI

Diabetic retinopathy grading using a ConvNeXt-Tiny ensemble trained on the APTOS dataset via Kaggle (free GPU).


## Output


<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; width: 100%; max-width: 800px; margin: 20px auto; font-family: sans-serif;">
  

  <div style="width: 100%; height: 320px; text-align: center; border: 1px solid #e1e4e8; padding: 10px; border-radius: 6px; background-color: #f6f8fa; display: flex; flex-direction: column; justify-content: space-between; align-items: center;">
    <img src="outputs/tta-comparison/confidence_stability_plot.png" alt="TTA Stability Plot" style="max-width: 100%; max-height: 80%; object-fit: contain; border-radius: 4px;" />
    <p style="margin: 8px 0 0 0; font-weight: bold; color: #24292e; font-size: 14px;">Caption 1: TTA Pipeline Confidence vs Baseline</p>
  </div>


  <div style="width: 100%; height: 320px; text-align: center; border: 1px solid #e1e4e8; padding: 10px; border-radius: 6px; background-color: #f6f8fa; display: flex; flex-direction: column; justify-content: space-between; align-items: center;">
    <img src="outputs/evaluation-unseen-data/dr_accuracy_breakdown.png" alt="DR Accuracy Breakdown" style="max-width: 100%; max-height: 80%; object-fit: contain; border-radius: 4px;" />
    <p style="margin: 8px 0 0 0; font-weight: bold; color: #24292e; font-size: 14px;">Caption 2: DR Accuracy Breakdown on Unseen Data</p>
  </div>


  <div style="width: 100%; height: 320px; text-align: center; border: 1px solid #e1e4e8; padding: 10px; border-radius: 6px; background-color: #f6f8fa; display: flex; flex-direction: column; justify-content: space-between; align-items: center;">
    <img src="outputs/weights_inspection/cross_fold_performance_summary.png" alt="Cross Validation QWK" style="max-width: 100%; max-height: 80%; object-fit: contain; border-radius: 4px;" />
    <p style="margin: 8px 0 0 0; font-weight: bold; color: #24292e; font-size: 14px;">Caption 3: 5-Fold Best Val QWK Comparison</p>
  </div>


  <div style="width: 100%; height: 320px; text-align: center; border: 1px solid #e1e4e8; padding: 10px; border-radius: 6px; background-color: #f6f8fa; display: flex; flex-direction: column; justify-content: space-between; align-items: center;">
    <img src="outputs/weights_inspection/weight_distribution_fold_0.png" alt="Fold 0 Weight Distribution" style="max-width: 100%; max-height: 80%; object-fit: contain; border-radius: 4px;" />
    <p style="margin: 8px 0 0 0; font-weight: bold; color: #24292e; font-size: 14px;">Caption 4: backbone.stem.0 Weight Distribution (Fold 0)</p>
  </div>

</div>


## Architecture
![Architecture Diagram](./architecture/how_it_works.svg)


# What it does:

Takes retinal fundus images as input and predicts the DR grade (0–4) using a 5-fold model ensemble with Test-Time Augmentation (TTA). Also generates Grad-CAM heatmaps to visualize which regions of the image influenced the prediction.

# Grades:

0 — No DR

1 — Mild DR

2 — Moderate DR

3 — Severe DR

4 — Proliferative DR

# Files:

infer2.py = Main inference script with Grad-CAM, uncertainty flagging, and detailed terminal output

inference.py = Simpler inference script — just grades + confidence

inspect_weights.py = Inspect loaded .pth weight files (layer count, param count, size)


# Usage

Full inference with Grad-CAM and uncertainty flagging:

python infer2.py

Simple inference (grade + confidence only):

python inference.py

Inspect weight files:

python inspect_weights.py



# Key features:

5-fold ensemble — averages predictions across multiple model weights for more reliable output

4-step TTA — runs each image through horizontal flip, vertical flip, and both, then averages

Uncertainty flagging — marks predictions below 60% confidence for manual review

Grad-CAM — saves heatmap overlays showing which retinal regions drove the prediction

Ben Graham preprocessing — standard fundus image enhancement before inference


# Notes:


Weights were trained on Kaggle using free GPU (T4)

Model: convnext_tiny.fb_in22k_ft_in1k  via the timm library

CPU inference supported; GPU used automatically if available

Each run saves to a timestamped folder inside results/ — no overwriting
