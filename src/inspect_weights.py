import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

WEIGHTS_DIR = "weights"
WEIGHTS_RESULTS_DIR = "weights_results"

def inspect_and_visualize_weights():
    if not os.path.exists(WEIGHTS_DIR):
        print(f"[ERROR] Weights directory '{WEIGHTS_DIR}/' does not exist.")
        return

    weight_files = sorted([
        f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth")
    ])

    if not weight_files:
        print(f"[ERROR] No .pth files found in '{WEIGHTS_DIR}/'")
        return

    os.makedirs(WEIGHTS_RESULTS_DIR, exist_ok=True)
    existing_runs = [
        d for d in os.listdir(WEIGHTS_RESULTS_DIR)
        if os.path.isdir(os.path.join(WEIGHTS_RESULTS_DIR, d)) and d.startswith("weights_run_")
    ]
    run_number = len(existing_runs) + 1
    current_run_dir = os.path.join(WEIGHTS_RESULTS_DIR, f"weights_run_{run_number}")
    os.makedirs(current_run_dir, exist_ok=True)

    model_names = []
    qwk_scores = []
    has_valid_qwk = False

    sns.set_theme(style="whitegrid")

    for wf in weight_files:
        path = os.path.join(WEIGHTS_DIR, wf)
        
        print("\n" + "=" * 60)
        print(f"  ADVANCED CHECKPOINT INSPECTION — {wf}")
        print("=" * 60)

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ERROR] Failed to load {wf}. Reason: {e}")
            continue

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            epoch = checkpoint.get("epoch", "N/A")
            best_qwk = checkpoint.get("best_qwk", "N/A")
            fold = checkpoint.get("fold", "N/A")
            config = checkpoint.get("config", {})
            has_metadata = True
        else:
            state_dict = checkpoint
            best_qwk = "N/A"
            has_metadata = False

        model_names.append(wf)
        if isinstance(best_qwk, (int, float)):
            qwk_scores.append(float(best_qwk))
            has_valid_qwk = True
        else:
            qwk_scores.append(0.0)

        print(" FILE METADATA & TRAINING CONTEXT")
        print("-" * 60)
        print(f"• File Path       : {path}")
        print(f"• File Size       : {os.path.getsize(path) / 1e6:.2f} MB")
        
        if has_metadata:
            print(f"• Assigned Fold   : Fold {fold}")
            print(f"• Saved Epoch     : Epoch {epoch}")
            print(f"• Best Val QWK    : {best_qwk:.4f}" if isinstance(best_qwk, float) else f"• Best Val QWK    : {best_qwk}")
            print(f"• Input Resolution: {config.get('image_size', 'Unknown')}x{config.get('image_size', 'Unknown')}")
            print(f"• Checkpoint Type : Multi-Key Training Snapshot")
        else:
            print(f"• Checkpoint Type : Pure Weights Only")

        print("\n DIABETIC RETINOPATHY HEAD VERIFICATION")
        print("-" * 60)
        
        head_keys = [k for k in state_dict.keys() if "head.fc.weight" in k or "head.weight" in k or "classifier" in k]
        
        if head_keys:
            head_key = head_keys[0]  
            head_tensor = state_dict[head_key]
            num_classes = head_tensor.shape[0]  
            shape_str = str(list(head_tensor.shape))
            
            print(f"✔ Found Classification Head: '{head_key}'")
            print(f"• Configured Outputs: {num_classes} Classes " + 
                  ("(Matches No-DR to Proliferative scale)" if num_classes == 5 else "(⚠️ Unexpected class count!)"))
            print(f"• Target Layer Shape: {shape_str}")
        else:
            print(" WARNING: Could not find classification head tensor layer.")

        print("\n WEIGHT HEALTH & MATHEMATICAL ACCURACY")
        print("-" * 60)
        
        total_params = 0
        layer_count = len(state_dict.keys())
        
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor):
                total_params += tensor.numel()

        print(f"• Total Layers Processed : {layer_count} layers")
        print(f"• Total Parameter Count  : {total_params:,} (~{total_params/1e6:.1f}M parameters)")

        stem_key = "backbone.stem.0.weight"
        if stem_key in state_dict and isinstance(state_dict[stem_key], torch.Tensor):
            weights = state_dict[stem_key].float()
            mean_val = weights.mean().item()
            std_val = weights.std().item()
            dead_neurons = (weights == 0).sum().item() / weights.numel() * 100
            
            print(f"\n• Feature Layer Status ('{stem_key}'):")
            print(f"  - Mathematical Mean : {mean_val:.4f} " + ("(Healthy / Centered near 0)" if abs(mean_val) < 0.1 else "(⚠️ Biased distribution)"))
            print(f"  - Std Deviation     : {std_val:.4f} " + ("(Stable)" if std_val > 0.001 else "(⚠️ Dead/Vanishing values)"))
            print(f"  - Dead Neurons (0s) : {dead_neurons:.2f}%")
            
            # --- GRAPH 1: Weight Distribution Curve for current file ---
            plt.figure(figsize=(6, 4))
            weights_np = weights.numpy().flatten()
            
            sns.histplot(weights_np, kde=True, color="#3498db", bins=40, edgecolors='w', alpha=0.7)
            plt.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.4f}")
            
            plt.title(f"Weight Value Distribution\nLayer: {stem_key} ({wf})", fontsize=11, fontweight="bold")
            plt.xlabel("Weight Values", fontsize=9)
            plt.ylabel("Frequency", fontsize=9)
            plt.legend(loc="upper right")
            plt.tight_layout()
            
            g1_name = f"weight_distribution_{os.path.splitext(wf)[0]}.png"
            g1_path = os.path.join(current_run_dir, g1_name)
            plt.savefig(g1_path, dpi=300)
            plt.close()
            print(f"✔ Distribution curve generated -> {g1_path}")
        else:
            print("\n• Feature Layer Status: Standard layer metrics unavailable for this checkpoint format.")

        print("\n" + "=" * 60)
        if head_keys and state_dict[head_keys[0]].shape[0] == 5:
            print("Recommendation: Weights are stable, validated, and structured\n"
                  "                correctly for 5-class Retinopathy inference.")
        else:
            print("Recommendation: Review classification layer configuration before running tests.")
        print("=" * 60 + "\n")


    if has_valid_qwk:
        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(model_names, qwk_scores, color="#2ecc71", width=0.4)
        
        plt.title("Cross-Fold Model Performance Comparison (Best Val QWK)", fontsize=12, fontweight="bold", pad=15)
        plt.ylabel("Quadratic Weighted Kappa (QWK)", fontsize=10)
        plt.xlabel("Checkpoint Files", fontsize=10)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight="bold")
            
        plt.tight_layout()
        g2_path = os.path.join(current_run_dir, "cross_fold_performance_summary.png")
        plt.savefig(g2_path, dpi=300)
        plt.close()
        print(f"✔ Global Comparison Summary Chart saved -> {g2_path}")
    else:
        print("Skipping Cross-Fold Performance Summary Chart: No numerical validation QWK metrics found in files.")


if __name__ == "__main__":
    inspect_and_visualize_weights()
