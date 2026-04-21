
import os
import torch

WEIGHTS_DIR = "weights"

def inspect_weights():
    weight_files = sorted([
        f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth")
    ])

    if not weight_files:
        print(f"[ERROR] No .pth files found in '{WEIGHTS_DIR}/'")
        return

    print(f"\n{'='*60}")
    print(f"  WEIGHT INSPECTOR — {len(weight_files)} file(s) found")
    print(f"{'='*60}\n")

    for wf in weight_files:
        path = os.path.join(WEIGHTS_DIR, wf)
        print(f"{'─'*60}")
        print(f"  FILE: {wf}")
        print(f"  PATH: {path}")
        print(f"  SIZE: {os.path.getsize(path) / 1e6:.2f} MB")
        print(f"{'─'*60}")

        state_dict = torch.load(path, map_location="cpu")

        total_params = 0
        layer_count  = 0

        print(f"  {'LAYER NAME':<55} {'SHAPE'}")
        print(f"  {'─'*55} {'─'*15}")

        for i, (key, tensor) in enumerate(state_dict.items()):
            total_params += tensor.numel()
            layer_count  += 1
            if i < 10:
                shape_str = str(list(tensor.shape))
                print(f"  {key:<55} {shape_str}")

        if layer_count > 10:
            print(f"  ... ({layer_count - 10} more layers not shown)")

        print(f"\n  Total Layers     : {layer_count}")
        print(f"  Total Parameters : {total_params:,}")
        print(f"  (~{total_params/1e6:.1f} million parameters)\n")

    print(f"{'='*60}")
    print(f"All weights inspected successfully.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    inspect_weights()