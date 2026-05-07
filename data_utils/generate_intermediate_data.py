#!/usr/bin/env python3
"""
Generate intermediate training data for sequential MRI reconstruction experts.

This script helps chain multiple expert models together by:
1. Creating evaluation JSON files for inference
2. Processing reconstructed outputs to create training data for the next expert

CONFIGURATION: Edit the parameters in the main() function at the bottom
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import re


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_case_id(filepath: str) -> Tuple[int, str]:
    """
    Extract case ID from filepath for sorting.

    Args:
        filepath: Path to the file

    Returns:
        Tuple of (case_number, full_case_id) for sorting
    """
    filename = Path(filepath).name

    # Extract case number from path (e.g., /415/, /524/, etc.)
    match = re.search(r'/(\d+)/', filepath.replace('\\', '/'))
    if match:
        case_num = int(match.group(1))
        return (case_num, match.group(1))

    # Fallback: try to find any number in filename
    match = re.search(r'(\d+)', filename)
    if match:
        case_num = int(match.group(1))
        return (case_num, match.group(1))

    # Last resort: use filename as-is
    return (0, filename)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_eval_json(input_json: Path, output_json: Path, use_all_splits: bool = True):
    """
    Create evaluation JSON where all data is moved to the testing section.

    Args:
        input_json: Path to original training JSON
        output_json: Path to output evaluation JSON
        use_all_splits: If True, combine all splits into testing. If False, only use training split.
    """
    print(f"📖 Reading {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Collect all data
    all_data = []

    if use_all_splits:
        # Combine all splits (training + validation + testing)
        for split in ['training', 'validation', 'test']:
            if split in data and data[split]:
                print(f"✅ Found {len(data[split])} cases in '{split}' split")
                all_data.extend(data[split])
    else:
        # Only use training split
        if 'training' in data:
            all_data = data['training']
            print(f"✅ Found {len(all_data)} cases in 'training' split")

    print(f"📊 Total cases collected: {len(all_data)}")

    # Sort by case ID to ensure proper ordering
    sorted_data = sorted(all_data, key=lambda x: extract_case_id(x['input']))
    print(f"🔢 Sorted {len(sorted_data)} cases by case ID")

    # Create new JSON with everything in testing
    # Keep one dummy entry in training and validation to prevent bugs from empty splits
    dummy_entry = sorted_data[0] if sorted_data else {'input': '', 'target': ''}
    eval_data = {
        'training': [dummy_entry],
        'validation': [dummy_entry],
        'test': sorted_data
    }

    # Save output
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print(f"\n✅ Created evaluation JSON: {output_json}")
    print(f"   - Training: {len(eval_data['training'])} (dummy entry)")
    print(f"   - Validation: {len(eval_data['validation'])} (dummy entry)")
    print(f"   - Testing: {len(eval_data['test'])}")
    print(f"\n📌 Next step: Run inference using this JSON file")
    print(f"   Example: python main.py +experiments=... dataset_test.args.jsonData={output_json}")


def create_next_expert_json(
    original_json: Path,
    reconstructed_dir: Path,
    output_json: Path,
    next_expert: str,
    target_dir: Path = None
):
    """
    Create training JSON for the next expert using reconstructed outputs.

    Args:
        original_json: Original training JSON (for metadata)
        reconstructed_dir: Directory containing reconstructed volumes from previous expert
        output_json: Path to output JSON for next expert
        next_expert: Name of next expert (e.g., 'axial', 'coronal')
        target_dir: Optional directory containing target files (if different from original)
    """
    print(f"📖 Reading original JSON: {original_json}...")
    with open(original_json, 'r') as f:
        original_data = json.load(f)

    reconstructed_dir = Path(reconstructed_dir)
    if not reconstructed_dir.exists():
        raise ValueError(f"Reconstructed directory not found: {reconstructed_dir}")

    print(f"🔍 Scanning reconstructed directory: {reconstructed_dir}...")
    reconstructed_files = list(reconstructed_dir.glob("*.nii.gz")) + \
                         list(reconstructed_dir.glob("*.nii"))
    print(f"✅ Found {len(reconstructed_files)} reconstructed files")

    # Create mapping from case ID to reconstructed file
    case_to_reconstructed = {}
    for recon_file in reconstructed_files:
        case_num, case_id = extract_case_id(str(recon_file))
        case_to_reconstructed[case_id] = str(recon_file)

    print(f"🗂️  Mapped {len(case_to_reconstructed)} unique case IDs")

    # Process each split
    new_data = {}
    for split in ['training', 'validation', 'test']:
        if split not in original_data:
            new_data[split] = []
            continue

        new_split = []
        missing_cases = []

        for item in original_data[split]:
            # Extract case ID from original input
            case_num, case_id = extract_case_id(item['input'])

            # Find corresponding reconstructed file
            if case_id in case_to_reconstructed:
                new_item = {
                    'input': case_to_reconstructed[case_id],
                    'target': item['target'] if target_dir is None else \
                             str(target_dir / Path(item['target']).name)
                }
                new_split.append(new_item)
            else:
                missing_cases.append(case_id)

        new_data[split] = new_split

        print(f"\n{split.upper()}:")
        print(f"  - Original cases: {len(original_data[split])}")
        print(f"  - Matched cases: {len(new_split)}")
        if missing_cases:
            print(f"  ⚠️  Missing cases: {len(missing_cases)}")
            print(f"      {', '.join(missing_cases[:5])}" +
                  (f" ... and {len(missing_cases)-5} more" if len(missing_cases) > 5 else ""))

    # Save output
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"\n✅ Created next expert JSON: {output_json}")
    print(f"   - Training cases: {len(new_data['training'])}")
    print(f"   - Validation cases: {len(new_data['validation'])}")
    print(f"   - Testing cases: {len(new_data.get('test', []))}")
    print(f"\n📌 Next step: Train {next_expert} expert using this JSON file")


def verify_json(json_file: Path):
    """Verify JSON file structure and report statistics."""
    print(f"🔍 Verifying JSON: {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    for split in ['training', 'validation', 'test']:
        if split not in data:
            print(f"  ⚠️  Missing '{split}' split")
            continue

        cases = data[split]
        print(f"\n{split.upper()} ({len(cases)} cases):")

        if len(cases) == 0:
            print("  (empty)")
            continue

        # Check first few cases
        for i, case in enumerate(cases[:3]):
            input_path = Path(case['input'])
            target_path = Path(case['target'])

            input_exists = "✅" if input_path.exists() else "❌"
            target_exists = "✅" if target_path.exists() else "❌"

            print(f"  Case {i+1}:")
            print(f"    Input:  {input_exists} {case['input']}")
            print(f"    Target: {target_exists} {case['target']}")

        if len(cases) > 3:
            print(f"  ... and {len(cases)-3} more cases")


# ============================================================================
# MAIN - CONFIGURE YOUR PARAMETERS HERE
# ============================================================================

def main():
    """
    Main function - Edit the MODE and parameters below to run different operations.
    """

    # ========================================================================
    # CONFIGURATION - EDIT THIS SECTION
    # ========================================================================

    # Choose operation mode:
    # - 'create_eval_json': Create evaluation JSON for inference
    # - 'create_next_expert_json': Create training JSON for next expert
    # - 'verify_json': Verify JSON file structure
    MODE = 'create_eval_json'

    # ------------------------------------------------------------------------
    # FOR MODE: 'create_eval_json'
    # ------------------------------------------------------------------------
    if MODE == 'create_eval_json':
        input_json = Path('json_datasets/fixed/SCA_2_8x.json')
        output_json = Path('json_datasets/fixed/SCA_2_8x_all.json')
        use_all_splits = True  # True: combine all splits, False: only training

        create_eval_json(input_json, output_json, use_all_splits)

    # ------------------------------------------------------------------------
    # FOR MODE: 'create_next_expert_json'
    # ------------------------------------------------------------------------
    elif MODE == 'create_next_expert_json':
        original_json = Path('json_datasets/Multi_view_IXI_sagittal.json')
        reconstructed_dir = Path('test_results/test_2d_SUNet_sagittal_8x_1')
        output_json = Path('json_datasets/Multi_view/SAC_8x/IXI_axial_from_sagittal_8x.json')
        next_expert = 'axial'
        target_dir = None  # Set to Path(...) if target files are in different location

        create_next_expert_json(
            original_json,
            reconstructed_dir,
            output_json,
            next_expert,
            target_dir
        )

    # ------------------------------------------------------------------------
    # FOR MODE: 'verify_json'
    # ------------------------------------------------------------------------
    elif MODE == 'verify_json':
        json_file = Path('json_datasets/eval_sagittal_8x.json')

        verify_json(json_file)

    else:
        print(f"❌ Unknown MODE: {MODE}")
        print("Valid modes: 'create_eval_json', 'create_next_expert_json', 'verify_json'")


if __name__ == '__main__':
    main()
