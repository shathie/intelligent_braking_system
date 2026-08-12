#!/usr/bin/env python3
"""
Main execution script for the Intelligent Multi-Modal Braking System.

This script provides a command-line interface for running the complete workflow:
1. Setup environment
2. Download datasets
3. Analyze datasets
4. Preprocess data
5. Train models
6. Evaluate models
7. Run simulations
8. Generate reports

Usage:
    python main.py --help
    python main.py --step analyze    # Run only analysis
    python main.py --step train      # Run only training
    python main.py --all             # Run complete workflow
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_script(script_name: str, args: List[str] = None) -> bool:
    """
    Run a script from the scripts directory.
    
    Args:
        script_name: Name of the script (without .py extension)
        args: Additional arguments to pass to the script
    
    Returns:
        True if successful, False otherwise
    """
    script_path = Path("scripts") / f"{script_name}.py"
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return False
    
    command = [sys.executable, str(script_path)]
    if args:
        command.extend(args)
    
    # Avoid printing characters that may not be encodable in the user's default console encoding
    # Print a plain-text indicator instead.
    print(f"\n[RUNNING] {' '.join(command)}\n")
    
    run_env = os.environ.copy()
    run_env.setdefault('IBS_FORCE_CPU', '1')
    run_env.setdefault('CUDA_VISIBLE_DEVICES', '-1')

    try:
        result = subprocess.run(command, check=True, env=run_env)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e}")
        return False


def run_all_steps() -> None:
    """Run all steps in sequence."""
    print("="*70)
    print("RUNNING COMPLETE WORKFLOW")
    print("="*70)
    
    steps = [
        ("download_datasets", "Downloading datasets"),
        ("analyze_datasets", "Analyzing datasets"),
        ("preprocess_data", "Preprocessing data"),
        ("train", "Training models"),
        ("evaluate", "Evaluating models"),
        ("simulate", "Running simulations"),
        ("report", "Generating reports")
    ]
    
    for script, description in steps:
        print(f"\n{'='*70}")
        print(f"STEP: {description.upper()}")
        print(f"{'='*70}\n")
        
        if not run_script(script):
            print(f"\nStep '{description}' failed. Continuing with next steps...")
    
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW FINISHED")
    print("="*70)


def run_specific_step(step: str) -> None:
    """Run a specific step."""
    step_scripts = {
        'setup': 'setup',
        'download': 'download_datasets',
        'analyze': 'analyze_datasets',
        'preprocess': 'preprocess_data',
        'train': 'train',
        'evaluate': 'evaluate',
        'simulate': 'simulate',
        'report': 'report',
        'all': None  # Special case
    }
    
    if step not in step_scripts:
        print(f" Unknown step: {step}")
        print(f"Available steps: {list(step_scripts.keys())}")
        return
    
    if step == 'all':
        run_all_steps()
    else:
        script = step_scripts[step]
        print(f"\n{'='*70}")
        print(f"STEP: {step.upper()}")
        print(f"{'='*70}\n")
        run_script(script)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Intelligent Multi-Modal Braking System - Main Execution Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --step analyze    # Run only dataset analysis
  python main.py --step train      # Run only training
  python main.py --all             # Run complete workflow
  python main.py --step simulate  # Run only simulations
  python main.py --step report     # Generate only reports
        """
    )
    
    parser.add_argument(
        '--step', '-s',
        type=str,
        choices=['setup', 'download', 'analyze', 'preprocess', 'train', 'evaluate', 'simulate', 'report', 'all'],
        help='Specific step to run'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run complete workflow (all steps in sequence)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available steps'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable steps:")
        print("  setup       - Setup environment")
        print("  download    - Download datasets")
        print("  analyze     - Analyze datasets")
        print("  preprocess  - Preprocess data")
        print("  train       - Train models")
        print("  evaluate    - Evaluate models")
        print("  simulate    - Run simulations")
        print("  report      - Generate reports")
        print("  all        - Run all steps in sequence")
        return
    
    if args.all:
        run_all_steps()
    elif args.step:
        run_specific_step(args.step)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()