#!/usr/bin/env python3
"""
Monitor pipeline training progress and extract metrics.
Polls log file and reports on key milestones.
"""
import re
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def get_latest_log():
    """Get the most recent pipeline log file."""
    logs_dir = Path("output/logs")
    pattern = "pipeline_all_*.log"
    logs = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None

def extract_epoch_metrics(log_path):
    """Extract training metrics from log."""
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Find all epoch completions
    epochs = re.findall(r"Epoch (\d+).*?Train Loss: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+)%", content)
    return epochs

def get_pipeline_status(log_path):
    """Get current pipeline phase."""
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
    
    if "STEP 2.*Temporal" in content or "STEP 2" in content and "Temporal" in content[-500:]:
        return "Temporal Training"
    elif "STEP 3.*Fusion" in content or "STEP 3" in content and "Fusion" in content[-500:]:
        return "Fusion Training"
    elif "STEP 4.*PINN" in content or "STEP 4" in content and "PINN" in content[-500:]:
        return "PINN Training"
    elif "STEP 5.*SAC" in content or "STEP 5" in content and "SAC" in content[-500:]:
        return "SAC Training"
    elif "PIPELINE COMPLETE" in content:
        return "COMPLETE"
    else:
        # Extract current ViT epoch
        matches = re.findall(r"Epoch (\d+)/30", content)
        if matches:
            return f"ViT Training - Epoch {matches[-1]}/30"
        return "Starting"

def format_metrics(epochs_data):
    """Format epoch metrics for display."""
    if not epochs_data:
        return "No epochs completed yet"
    
    lines = ["Epoch Results:"]
    for epoch_num, train_loss, val_loss, val_acc in epochs_data[-5:]:  # Show last 5
        lines.append(f"  Epoch {epoch_num}: Loss={val_loss} Acc={val_acc}%")
    
    if len(epochs_data) > 1:
        first_acc = float(epochs_data[0][3])
        last_acc = float(epochs_data[-1][3])
        improvement = last_acc - first_acc
        lines.append(f"\nImprovement: {first_acc:.2f}% → {last_acc:.2f}% (Δ={improvement:+.2f}%)")
    
    return "\n".join(lines)

def main():
    """Main monitoring loop."""
    print("Pipeline Monitoring Started")
    print("=" * 60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    last_epoch_count = 0
    last_status = ""
    
    # Monitor for up to 30 hours
    for check_num in range(360):  # 360 checks × 5 min = 30 hours
        time.sleep(300)  # Check every 5 minutes
        
        log_path = get_latest_log()
        if not log_path:
            continue
        
        try:
            status = get_pipeline_status(str(log_path))
            epochs = extract_epoch_metrics(str(log_path))
            current_epoch = len(epochs)
            
            # Report changes
            if status != last_status:
                print(f"[{datetime.now().strftime('%H:%M')}] Phase Change: {status}")
                last_status = status
            
            if current_epoch > last_epoch_count and current_epoch % 4 == 0:  # Every 4 epochs
                print(f"[{datetime.now().strftime('%H:%M')}] {format_metrics(epochs)}")
                last_epoch_count = current_epoch
            elif current_epoch > last_epoch_count:
                last_epoch_count = current_epoch
            
            # Check for completion
            if status == "COMPLETE":
                print("\n" + "=" * 60)
                print("✅ PIPELINE COMPLETE!")
                print("=" * 60)
                print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("\nRun: python scripts/analyze_results.py")
                break
        
        except Exception as e:
            pass  # Silently continue on error

if __name__ == "__main__":
    main()
