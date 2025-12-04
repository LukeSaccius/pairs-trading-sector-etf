"""
Script to reorganize results folder structure
"""
import shutil
from pathlib import Path

RESULTS_DIR = Path("I:/Winter-Break-Research/results")

# Define the new structure
STRUCTURE = {
    "experiments/v14_vidyamurthy": [
        "2025-12-03_12-46_v14_vidyamurthy_full",
    ],
    "experiments/v15_kalman": [
        "2025-12-03_13-08_v15_full_features",
        "2025-12-03_13-12_v15b_vix_volsizing",
    ],
    "experiments/v16_optimized": [
        "2025-12-03_15-56_v16_optimized",
        "2025-12-03_15-59_v16_optimized",
        "2025-12-03_16-06_v16b_best",
    ],
    "experiments/v17_final": [
        "2025-12-03_20-34_v17a_vol_filter",  # BEST
        "2025-12-03_20-35_v17b_dynamic_exit",
        "2025-12-03_20-35_v17c_combined",
        "2025-12-03_20-45_v17d_slow_conv",
        "2025-12-03_20-48_v17e_slow_conv_60",
    ],
    "archive/duplicates/v15c_kalman": [
        "2025-12-03_13-27_v15c_kalman_momentum",
        "2025-12-03_13-29_v15c_kalman_momentum",
        "2025-12-03_13-32_v15c_kalman_momentum",
        "2025-12-03_13-39_v15c_kalman_momentum",
        "2025-12-03_13-40_v15c_kalman_momentum",
        "2025-12-03_13-41_v15b_vix_volsizing",
    ],
    "archive/duplicates/v16b_runs": [
        "2025-12-03_16-10_v16b_best",
        "2025-12-03_16-11_v16b_best",
        "2025-12-03_16-46_v16b_best",
        "2025-12-03_17-14_v16b_best",
        "2025-12-03_20-09_v16b_best",
    ],
    "archive/duplicates/v17_early": [
        "2025-12-03_16-45_v17_dynamic_holding",
        "2025-12-03_16-46_v17b_dynamic_balanced",
    ],
}

def main():
    print("=" * 60)
    print("Reorganizing Results Folder")
    print("=" * 60)
    
    # Create target directories
    for target_dir in STRUCTURE.keys():
        target_path = RESULTS_DIR / target_dir
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {target_path}")
    
    # Move folders
    moved = 0
    skipped = 0
    errors = 0
    
    for target_dir, sources in STRUCTURE.items():
        target_path = RESULTS_DIR / target_dir
        
        for source_name in sources:
            source_path = RESULTS_DIR / source_name
            dest_path = target_path / source_name
            
            if source_path.exists():
                if dest_path.exists():
                    print(f"  [SKIP] Already exists: {dest_path}")
                    skipped += 1
                else:
                    try:
                        shutil.move(str(source_path), str(dest_path))
                        print(f"  [MOVE] {source_name} -> {target_dir}/")
                        moved += 1
                    except Exception as e:
                        print(f"  [ERROR] {source_name}: {e}")
                        errors += 1
            else:
                print(f"  [SKIP] Not found: {source_name}")
                skipped += 1
    
    print()
    print("=" * 60)
    print(f"Summary: {moved} moved, {skipped} skipped, {errors} errors")
    print("=" * 60)
    
    # Show final structure
    print("\nFinal structure:")
    for p in sorted(RESULTS_DIR.glob("**/*")):
        if p.is_dir():
            rel = p.relative_to(RESULTS_DIR)
            depth = len(rel.parts)
            indent = "  " * (depth - 1)
            print(f"{indent}📁 {p.name}/")

if __name__ == "__main__":
    main()
