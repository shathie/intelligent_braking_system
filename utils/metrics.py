"""
Evaluation metrics for the intelligent braking system.
Includes metrics for classification, regression, control, and system performance.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)


class ClassificationMetrics:
    """Metrics for classification tasks (e.g., road surface classification)."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    def compute(self, y_true: Union[np.ndarray, torch.Tensor],
               y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_score(
                y_true, y_pred, average=None, labels=[i], zero_division=0
            )[0] if i in np.unique(y_true) else 0.0
            
            metrics[f'recall_{class_name}'] = recall_score(
                y_true, y_pred, average=None, labels=[i], zero_division=0
            )[0] if i in np.unique(y_true) else 0.0
        
        return metrics
    
    def print_report(self, y_true: Union[np.ndarray, torch.Tensor],
                    y_pred: Union[np.ndarray, torch.Tensor]) -> None:
        """Print classification report."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=self.class_names,
                                    labels=list(range(len(self.class_names))), zero_division=0))
        print("="*50 + "\n")


class RegressionMetrics:
    """Metrics for regression tasks (e.g., friction estimation)."""
    
    def compute(self, y_true: Union[np.ndarray, torch.Tensor],
               y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Compute metrics
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'median_error': np.median(np.abs(y_true - y_pred)),
            'std_error': np.std(y_true - y_pred),
        }
        
        # For friction estimation, we care about errors in different ranges
        # Error when true μ < 0.3 (low friction)
        low_friction_mask = y_true < 0.3
        if np.sum(low_friction_mask) > 0:
            metrics['mse_low_friction'] = mean_squared_error(
                y_true[low_friction_mask], y_pred[low_friction_mask]
            )
            metrics['mae_low_friction'] = mean_absolute_error(
                y_true[low_friction_mask], y_pred[low_friction_mask]
            )
        
        # Error when true μ > 0.7 (high friction)
        high_friction_mask = y_true > 0.7
        if np.sum(high_friction_mask) > 0:
            metrics['mse_high_friction'] = mean_squared_error(
                y_true[high_friction_mask], y_pred[high_friction_mask]
            )
            metrics['mae_high_friction'] = mean_absolute_error(
                y_true[high_friction_mask], y_pred[high_friction_mask]
            )
        
        return metrics
    
    def print_report(self, y_true: Union[np.ndarray, torch.Tensor],
                    y_pred: Union[np.ndarray, torch.Tensor]) -> None:
        """Print regression report."""
        metrics = self.compute(y_true, y_pred)
        
        print("\n" + "="*50)
        print("REGRESSION REPORT")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key:25s}: {value:.6f}")
        print("="*50 + "\n")


class ControlMetrics:
    """Metrics for control system performance (e.g., braking control)."""
    
    def compute(self, trajectories: Dict[str, np.ndarray],
               targets: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Compute control system metrics.
        
        Args:
            trajectories: Dictionary of trajectories (e.g., {'velocity': [...], 'braking_force': [...]})
            targets: Optional dictionary of target trajectories
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Stopping distance
        if 'velocity' in trajectories and 'time' in trajectories:
            v = trajectories['velocity']
            t = trajectories['time']
            stopping_distance = np.trapezoid(v, t)
            metrics['stopping_distance'] = stopping_distance
        
        # Stopping time
        if 'velocity' in trajectories and 'time' in trajectories:
            v = trajectories['velocity']
            t = trajectories['time']
            stopping_time = t[np.argmax(v <= 0.1)] if np.any(v <= 0.1) else t[-1]
            metrics['stopping_time'] = stopping_time
        
        # Jerk (rate of change of acceleration)
        if 'acceleration' in trajectories and 'time' in trajectories:
            a = trajectories['acceleration']
            t = trajectories['time']
            dt = np.diff(t)
            jerk = np.diff(a) / dt
            metrics['max_jerk'] = np.max(np.abs(jerk))
            metrics['mean_jerk'] = np.mean(np.abs(jerk))
            metrics['rms_jerk'] = np.sqrt(np.mean(jerk ** 2))
        
        # Braking efficiency
        if 'velocity' in trajectories and 'braking_force' in trajectories:
            initial_energy = 0.5 * 1500 * (trajectories['velocity'][0] ** 2)
            work_done = np.trapezoid(trajectories['braking_force'] * 1500 * 9.81, trajectories['time'])
            metrics['braking_efficiency'] = work_done / initial_energy if initial_energy > 0 else 0
        
        # Wheel lock detection
        if 'wheel_speeds' in trajectories and 'velocity' in trajectories:
            wheel_speeds = trajectories['wheel_speeds']
            v_x = trajectories['velocity']
            R_w = 0.3  # Wheel radius
            
            # Calculate slip ratios
            slip_ratios = []
            for i in range(wheel_speeds.shape[1]):
                s = (R_w * wheel_speeds[:, i] - v_x) / np.maximum(R_w * wheel_speeds[:, i], v_x, 1e-6)
                slip_ratios.append(s)
            
            slip_ratios = np.array(slip_ratios).T
            
            # Percentage of time with high slip (>0.2)
            high_slip = np.mean(np.abs(slip_ratios) > 0.2) * 100
            metrics['wheel_lock_percentage'] = high_slip
        
        # Compare with targets if provided
        if targets is not None:
            for key in targets:
                if key in trajectories:
                    error = np.abs(trajectories[key] - targets[key])
                    metrics[f'{key}_error'] = np.mean(error)
        
        return metrics
    
    def print_report(self, trajectories: Dict[str, np.ndarray],
                    targets: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Print control system report."""
        metrics = self.compute(trajectories, targets)
        
        print("\n" + "="*50)
        print("CONTROL SYSTEM REPORT")
        print("="*50)
        for key, value in metrics.items():
            if 'distance' in key or 'time' in key:
                print(f"{key:25s}: {value:.2f}")
            elif 'percentage' in key:
                print(f"{key:25s}: {value:.2f}%")
            else:
                print(f"{key:25s}: {value:.4f}")
        print("="*50 + "\n")


class SystemMetrics:
    """Overall system performance metrics."""
    
    def __init__(self):
        self.classification_metrics = ClassificationMetrics(num_classes=5)
        self.regression_metrics = RegressionMetrics()
        self.control_metrics = ControlMetrics()
    
    def compute_all(self, classification_data: Optional[Dict] = None,
                   regression_data: Optional[Dict] = None,
                   control_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute all metrics for the system.
        
        Args:
            classification_data: Dictionary with 'y_true' and 'y_pred' for classification
            regression_data: Dictionary with 'y_true' and 'y_pred' for regression
            control_data: Dictionary with trajectories for control
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        if classification_data is not None:
            class_metrics = self.classification_metrics.compute(
                classification_data['y_true'],
                classification_data['y_pred']
            )
            metrics.update({f'classification_{k}': v for k, v in class_metrics.items()})
        
        if regression_data is not None:
            reg_metrics = self.regression_metrics.compute(
                regression_data['y_true'],
                regression_data['y_pred']
            )
            metrics.update({f'regression_{k}': v for k, v in reg_metrics.items()})
        
        if control_data is not None:
            control_metrics = self.control_metrics.compute(control_data)
            metrics.update({f'control_{k}': v for k, v in control_metrics.items()})
        
        return metrics
    
    def print_full_report(self, classification_data: Optional[Dict] = None,
                          regression_data: Optional[Dict] = None,
                          control_data: Optional[Dict] = None) -> None:
        """Print full system report."""
        print("\n" + "="*70)
        print("FULL SYSTEM PERFORMANCE REPORT")
        print("="*70)
        
        if classification_data is not None:
            self.classification_metrics.print_report(
                classification_data['y_true'],
                classification_data['y_pred']
            )
        
        if regression_data is not None:
            self.regression_metrics.print_report(
                regression_data['y_true'],
                regression_data['y_pred']
            )
        
        if control_data is not None:
            self.control_metrics.print_report(control_data)
        
        print("="*70 + "\n")


# Utility functions
def compute_metrics(y_true: Union[np.ndarray, torch.Tensor],
                   y_pred: Union[np.ndarray, torch.Tensor],
                   task_type: str = 'regression') -> Dict[str, float]:
    """
    Convenience function to compute metrics based on task type.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        task_type: 'classification' or 'regression'
    
    Returns:
        Dictionary of metrics
    """
    if task_type == 'classification':
        metrics = ClassificationMetrics(num_classes=len(np.unique(y_true)))
        return metrics.compute(y_true, y_pred)
    else:
        metrics = RegressionMetrics()
        return metrics.compute(y_true, y_pred)


def save_metrics(metrics: Dict[str, float], filename: str, output_dir: str = "metrics") -> None:
    """Save metrics to JSON file."""
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {path}")


# Example usage
if __name__ == "__main__":
    # Example 1: Classification metrics
    y_true_class = np.random.randint(0, 5, 100)
    y_pred_class = np.random.randint(0, 5, 100)
    
    class_metrics = ClassificationMetrics(num_classes=5)
    metrics = class_metrics.compute(y_true_class, y_pred_class)
    class_metrics.print_report(y_true_class, y_pred_class)
    
    # Example 2: Regression metrics
    y_true_reg = np.random.rand(100) * 0.8 + 0.1
    y_pred_reg = y_true_reg + np.random.randn(100) * 0.05
    y_pred_reg = np.clip(y_pred_reg, 0, 1)
    
    reg_metrics = RegressionMetrics()
    metrics = reg_metrics.compute(y_true_reg, y_pred_reg)
    reg_metrics.print_report(y_true_reg, y_pred_reg)
    
    # Example 3: Control metrics
    trajectories = {
        'time': np.linspace(0, 5, 100),
        'velocity': np.maximum(20 - 4 * np.linspace(0, 5, 100), 0),
        'acceleration': np.full(100, -4.0),
        'braking_force': np.full(100, 0.8),
        'wheel_speeds': np.random.rand(100, 4) * 50
    }
    
    control_metrics = ControlMetrics()
    metrics = control_metrics.compute(trajectories)
    control_metrics.print_report(trajectories)
