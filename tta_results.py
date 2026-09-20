import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

CSV_A_PATH = r"results\run_19-09-2026_1\inference_19-09-2026_1.csv"   # e.g. no-TTA run
CSV_A_LABEL = "no_tta"

CSV_B_PATH = r"results\run_19-09-2026_2\inference_19-09-2026_2.csv"   # e.g. TTA run
CSV_B_LABEL = "tta"

RESULTS_DIR = "results"
TTA_RESULTS_DIR = os.path.join(RESULTS_DIR, "tta_results")


def main():
    df_a = pd.read_csv(CSV_A_PATH)
    df_b = pd.read_csv(CSV_B_PATH)

    if "uncertain_flag_no_tta" in df_a.columns or "uncertain_flag" in df_a.columns:
        merged = df_a.copy() if "grade_changed" in df_a.columns else df_a.merge(
            df_b, on="image", suffixes=(f"_{CSV_A_LABEL}", f"_{CSV_B_LABEL}")
        )
    else:
        merged = df_a.merge(
            df_b, on="image", suffixes=(f"_{CSV_A_LABEL}", f"_{CSV_B_LABEL}")
        )

    grade_a = f"grade_{CSV_A_LABEL}"
    grade_b = f"grade_{CSV_B_LABEL}"
    conf_a  = f"confidence_%_{CSV_A_LABEL}"
    conf_b  = f"confidence_%_{CSV_B_LABEL}"

    if "grade_changed" not in merged.columns:
        merged["grade_changed"] = merged[grade_a] != merged[grade_b]
    if "confidence_shift" not in merged.columns:
        merged["confidence_shift"] = merged[conf_b] - merged[conf_a]

    total = len(merged)
    num_changed = int(merged["grade_changed"].sum())

    print(f"Predictions that changed ({CSV_A_LABEL} -> {CSV_B_LABEL}): {num_changed} / {total}")

    changed_columns = ["image", grade_a, grade_b, conf_a, conf_b, "confidence_shift"]
    available_changed_cols = [c for c in changed_columns if c in merged.columns]
    changed_only = merged[merged["grade_changed"]][available_changed_cols]

    os.makedirs(TTA_RESULTS_DIR, exist_ok=True)
    existing_runs = [
        d for d in os.listdir(TTA_RESULTS_DIR)
        if os.path.isdir(os.path.join(TTA_RESULTS_DIR, d)) and d.startswith("comparison_run_")
    ]
    run_number = len(existing_runs) + 1
    run_folder_name = f"comparison_run_{run_number}"
    current_run_dir = os.path.join(TTA_RESULTS_DIR, run_folder_name)
    os.makedirs(current_run_dir, exist_ok=True)

    date_str = datetime.now().strftime("%d-%m-%Y")
    out_path = os.path.join(current_run_dir, f"comparison_{date_str}.csv")
    merged.to_csv(out_path, index=False)
    print(f"Full comparison saved -> {out_path}")

    if num_changed > 0:
        changed_path = os.path.join(current_run_dir, f"comparison_{date_str}_changed_only.csv")
        changed_only.to_csv(changed_path, index=False)
        print(f"Changed-only subset saved -> {changed_path}")

    sns.set_theme(style="whitegrid")

    flag_a = f"uncertain_flag_{CSV_A_LABEL}"
    flag_b = f"uncertain_flag_{CSV_B_LABEL}"
    
    if flag_a in merged.columns and flag_b in merged.columns:
        plt.figure(figsize=(6, 5))
        
        uncertain_before = (merged[flag_a] == "YES").sum()
        uncertain_after = (merged[flag_b] == "YES").sum()
        
        bars = plt.bar(
            ["Baseline (No TTA)", "With TTA Pipeline"],
            [uncertain_before, uncertain_after],
            color=["#e74c3c", "#2ecc71"],
            width=0.5
        )
        
        plt.title("Uncertainty Resolution Analysis (Lower is Better)", fontsize=12, fontweight="bold", pad=15)
        plt.ylabel("Number of Low-Confidence Predictions (Uncertain = YES)", fontsize=10)
        
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, fontweight="bold")
            
        plt.tight_layout()
        g1_path = os.path.join(current_run_dir, "uncertainty_resolution_chart.png")
        plt.savefig(g1_path, dpi=300)
        plt.close()
        print(f"Graph 1 Saved: Executive Uncertainty Chart -> {g1_path}")
    else:
        print("\nSkipping Graph 1: 'uncertain_flag' columns not found in the input data structure.")

    if conf_a in merged.columns and conf_b in merged.columns:
        plt.figure(figsize=(7, 6))
        

        sns.scatterplot(
            data=merged,
            x=conf_a,
            y=conf_b,
            hue="grade_changed",
            palette={True: "#e74c3c", False: "#34495e"},
            alpha=0.8,
            s=60
        )
        
        plt.plot([0, 100], [0, 100], color="gray", linestyle="--", alpha=0.7, label="No Change Identity Line")
        
        plt.title("TTA Reliability & Decision Stability Plot", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel(f"Baseline Confidence (% {CSV_A_LABEL.upper()})", fontsize=10)
        plt.ylabel(f"TTA Pipeline Confidence (% {CSV_B_LABEL.upper()})", fontsize=10)
        plt.xlim(0, 105)
        plt.ylim(0, 105)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        business_labels = ["Grade Remained Stable", "Grade Swapped/Corrected", "Identity Line"]
        plt.legend(handles, business_labels, loc="upper left", frameon=True)
        
        plt.tight_layout()
        g2_path = os.path.join(current_run_dir, "confidence_stability_plot.png")
        plt.savefig(g2_path, dpi=300)
        plt.close()
        print(f"Graph 2 Saved: Engineering Stability Plot -> {g2_path}")
    else:
        print("Skipping Graph 2: Confidence score columns not found.")


if __name__ == "__main__":
    main()
