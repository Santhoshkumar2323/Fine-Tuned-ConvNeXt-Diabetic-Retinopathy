import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report
)

PREDICTIONS_CSV = r"results\run_19-09-2026_2\inference_19-09-2026_2.csv"
GROUND_TRUTH_CSV = "unseen-IDRID-data.csv"


def get_next_run_folder(base_dir="evaluation_results"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    run_num = 1
    while os.path.exists(os.path.join(base_dir, f"evaluation_run_{run_num}")):
        run_num += 1
        
    run_folder = os.path.join(base_dir, f"evaluation_run_{run_num}")
    os.makedirs(run_folder)
    return run_folder


def main():
    output_dir = get_next_run_folder()
    print(f"Saving all evaluation assets to: {output_dir}\n")

    preds = pd.read_csv(PREDICTIONS_CSV)
    truth = pd.read_csv(GROUND_TRUTH_CSV)

    merged = truth.merge(preds, on="image", how="left")

    missing = merged["grade"].isna().sum()

    if missing > 0:
        print(f"[WARNING] {missing} image(s) from ground truth were not "
              f"found in your predictions CSV - check filenames match "
              f"and that you tested exactly these 100 images.")
        print(merged[merged["grade"].isna()]["image"].tolist())

    merged = merged.dropna(subset=["grade"]).copy()
    merged["grade"] = merged["grade"].astype(int)

    y_true = merged["true_grade"].values
    y_pred = merged["grade"].values

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    print("=" * 60)
    print(f"REAL ACCURACY ON {len(merged)} IDRiD IMAGES (unseen data)")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"QWK      : {qwk:.4f}")

    print("\nConfusion Matrix (rows = true grade, cols = predicted grade):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]))

    target_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0,
        target_names=target_names
    ))

    merged["correct"] = merged["true_grade"] == merged["grade"]
    merged["off_by"] = (merged["true_grade"] - merged["grade"]).abs()

    def categorize_error(row):
        if row["correct"]:
            return "Correct"
        elif row["off_by"] == 1:
            return "Off by 1 Grade"
        else:
            return "Off by 2+ Grades"

    merged["Error Category"] = merged.apply(categorize_error, axis=1)

    plt.figure(figsize=(10, 6))
    
    grade_map = {i: name for i, name in enumerate(target_names)}
    merged["True Grade Name"] = merged["true_grade"].map(grade_map)

    sns.countplot(
        data=merged, 
        x="True Grade Name", 
        hue="Error Category", 
        order=target_names,
        palette={"Correct": "#2ecc71", "Off by 1 Grade": "#f1c40f", "Off by 2+ Grades": "#e74c3c"}
    )

    plt.title("Model Prediction Accuracy Breakdown by DR Severity")
    plt.xlabel("True Severity Grade")
    plt.ylabel("Number of Images")
    plt.legend(title="Prediction Status")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    chart_path = os.path.join(output_dir, "dr_accuracy_breakdown.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nClear accuracy bar chart saved -> {chart_path}")

    out_cols = ["image", "true_grade", "grade", "confidence_%", "correct", "off_by"]
    out_path = os.path.join(output_dir, "idrid_eval_detailed.csv")

    merged[out_cols].to_csv(out_path, index=False)
    print(f"Per-image detail saved -> {out_path}")

    wrong = merged[~merged["correct"]].sort_values("off_by", ascending=False)

    if len(wrong) > 0:
        print(f"\nBiggest misses (predicted far from true grade):")
        print(wrong[out_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
