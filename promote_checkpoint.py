#!/usr/bin/env python3
import shutil
import yaml
import re
import glob
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

# The directory where experiment 1 lives
EXP_DIR = Path("/work/tvoigtlaender/thesis/ml/mlruns/1")
FIXED_EXPERIMENT_ID = "1"

# Define the runs and which epoch to promote
# Format: (original_run_name, epoch_number)
PROMOTE_TASKS = [
    ("2025_apr_full_dataset_1_1", 7),
    ("2025_apr_full_dataset_1_1", 21),
    ("2025_apr_full_dataset_1_1", 47),
]

# =========================
# HELPERS
# =========================

def sanitize_run_id(run_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", run_name)[:64]

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, obj):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, default_flow_style=False)

def write_text_file(path: Path, content: str):
    """Write plain text to file without YAML formatting."""
    with open(path, "w") as f:
        f.write(content)

def find_checkpoint_dir(base_path: Path, epoch: int):
    """
    Finds a directory starting with 'epoch_0N' or 'epoch_N'
    inside the checkpoints folder.
    """
    checkpoints_root = base_path / "artifacts" / "checkpoints"
    if not checkpoints_root.exists():
        return None

    # Format epoch with leading zero for matching (e.g., 07)
    epoch_str = f"{epoch:02d}"
    pattern = str(checkpoints_root / f"epoch_{epoch_str}*")

    matches = glob.glob(pattern)
    return Path(matches[0]) if matches else None

# =========================
# MAIN PROMOTION LOGIC
# =========================

def main():
    for base_run_name, epoch in PROMOTE_TASKS:
        new_run_name = f"{base_run_name}_at{epoch}"
        print(f"\nPromoting: {base_run_name} (Epoch {epoch}) -> {new_run_name}")

        src_run_path = EXP_DIR / sanitize_run_id(base_run_name)
        dst_run_path = EXP_DIR / sanitize_run_id(new_run_name)

        if not src_run_path.exists():
            raise FileNotFoundError(f"  [!] Source run not found: {src_run_path}")

        if dst_run_path.exists():
            raise ValueError(f"  [!] Destination run already exists, skipping: {dst_run_path}")

        # 1. Create directory structure (selective copy)
        print(f"  [>] Creating run structure and copying metadata...")
        dst_run_path.mkdir(parents=True)

        # Copy critical metadata files only
        for item in ["meta.yaml", "params", "metrics", "tags"]:
            src_item = src_run_path / item
            if src_item.is_dir():
                shutil.copytree(src_item, dst_run_path / item)
            else:
                shutil.copy2(src_item, dst_run_path / item)

        # 2. Fix Metadata for the new run
        meta_path = dst_run_path / "meta.yaml"
        meta = load_yaml(meta_path)
        meta["run_id"] = dst_run_path.name
        meta["run_name"] = new_run_name
        meta["artifact_uri"] = str(dst_run_path)
        save_yaml(meta_path, meta)

        # 3. Fix Tags (mlflow.runName)
        tags_dir = dst_run_path / "tags"
        tags_dir.mkdir(exist_ok=True)
        write_text_file(tags_dir / "mlflow.runName", new_run_name)

        # 4. Selective Artifact Promotion
        dst_artifacts = dst_run_path / "artifacts"
        dst_artifacts.mkdir(exist_ok=True)

        # A. Find and Copy the Checkpoint Directory
        checkpoint_dir = find_checkpoint_dir(src_run_path, epoch)
        if checkpoint_dir and checkpoint_dir.is_dir():
            print(f"  [✓] Found checkpoint: {checkpoint_dir.name}")
            target_model_dir = dst_artifacts / "model"
            shutil.copytree(checkpoint_dir, target_model_dir)
            print(f"      Promoted to {target_model_dir}")
        else:
            raise ValueError(f"  [!] Checkpoint directory for epoch {epoch} not found matching pattern.")

        # B. Copy Scaler files if they exist
        src_artifacts = src_run_path / "artifacts"
        if src_artifacts.exists():
            for pkl_file in src_artifacts.glob("*.pkl"):
                print(f"  [✓] Copying scaler: {pkl_file.name}")
                shutil.copy2(pkl_file, dst_artifacts / pkl_file.name)
        else:
            print(f"  No scaler to copy at {pkl_file.name}")

    print("\nPromotion complete.")

if __name__ == "__main__":
    main()
