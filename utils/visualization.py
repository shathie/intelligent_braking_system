"""
Visualization utilities for the intelligent braking system.
Includes plotting functions for training curves, prediction results, and system analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class TrainingPlotter:
    """Plot training curves and metrics."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_plot_path(self, base_name: str) -> Path:
        """Return a non-overwriting file path in the output directory."""
        from datetime import datetime
        stem = base_name.replace(' ', '_').lower()
        return self.output_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    def _smooth(self, values: List[float], window: int = 5) -> np.ndarray:
        """Simple moving average smoothing for clearer training curves."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < window:
            return arr
        kernel = np.ones(window, dtype=np.float64) / window
        return np.convolve(arr, kernel, mode='same')
    
    def plot_training_curve(self, history: Dict[str, List[float]],
                           title: str = "Training Curve") -> None:
        """
        Plot training and validation loss/accuracy.
        
        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot loss
        if 'train_loss' in history or 'val_loss' in history:
            ax1.set_title("Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            
            if 'train_loss' in history:
                train_loss = np.asarray(history['train_loss'], dtype=np.float64)
                ax1.plot(train_loss, label='Train Loss', alpha=0.3, linewidth=1)
                ax1.plot(self._smooth(train_loss.tolist()), label='Train Loss (smooth)', linewidth=2)
            if 'val_loss' in history:
                val_loss = np.asarray(history['val_loss'], dtype=np.float64)
                ax1.plot(val_loss, label='Val Loss', alpha=0.3, linewidth=1)
                ax1.plot(self._smooth(val_loss.tolist()), label='Val Loss (smooth)', linewidth=2)

            if 'train_loss' in history and len(history['train_loss']) > 1:
                loss_values = np.asarray(history['train_loss'], dtype=np.float64)
                non_zero = loss_values[np.abs(loss_values) > 1e-12]
                if non_zero.size > 1 and (non_zero.max() / non_zero.min()) > 100:
                    ax1.set_yscale('log')
            
            ax1.legend()
            ax1.grid(True)
        
        # Plot non-loss metrics (accuracy, MAE, R2, etc.)
        metric_keys = [k for k in history.keys() if 'loss' not in k.lower()]
        if metric_keys:
            ax2.set_title("Metrics")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Metric Value")
            for key in metric_keys:
                values = np.asarray(history[key], dtype=np.float64)
                ax2.plot(values, label=key, alpha=0.35, linewidth=1)
                ax2.plot(self._smooth(values.tolist()), label=f"{key} (smooth)", linewidth=2)
            ax2.legend()
            ax2.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plot_name = title if title else "training_curve"
        plt.savefig(self._build_plot_path(plot_name), dpi=220, bbox_inches='tight')
        plt.close(fig)
    
    def plot_loss_components(self, history: Dict[str, List[float]],
                            title: str = "Loss Components") -> None:
        """Plot different loss components (data loss, physics loss, etc.)."""
        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(16, 6))

        loss_keys = [k for k in history.keys() if 'loss' in k.lower()]
        metric_keys = [k for k in history.keys() if k not in loss_keys]

        for key in loss_keys:
            values = np.asarray(history[key], dtype=np.float64)
            ax_loss.plot(values, label=key, alpha=0.35, linewidth=1)
            ax_loss.plot(self._smooth(values.tolist()), label=f"{key} (smooth)", linewidth=2)

        if loss_keys:
            all_loss = np.concatenate([np.asarray(history[k], dtype=np.float64) for k in loss_keys if len(history[k]) > 0])
            non_zero = np.abs(all_loss[np.abs(all_loss) > 1e-12])
            if non_zero.size > 1 and (non_zero.max() / non_zero.min()) > 100:
                ax_loss.set_yscale('log')
        ax_loss.set_title(f"{title} - Losses")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True)

        for key in metric_keys:
            values = np.asarray(history[key], dtype=np.float64)
            ax_metric.plot(values, label=key, alpha=0.35, linewidth=1)
            ax_metric.plot(self._smooth(values.tolist()), label=f"{key} (smooth)", linewidth=2)
        ax_metric.set_title(f"{title} - Metrics")
        ax_metric.set_xlabel("Epoch")
        ax_metric.set_ylabel("Metric")
        ax_metric.legend()
        ax_metric.grid(True)

        plt.tight_layout()
        plot_name = title if title else "loss_components"
        plt.savefig(self._build_plot_path(plot_name), dpi=220, bbox_inches='tight')
        plt.close(fig)


class PredictionVisualizer:
    """Visualize model predictions."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_friction_prediction(self, true_mu: np.ndarray, pred_mu: np.ndarray,
                                title: str = "Friction Coefficient Prediction") -> None:
        """Plot true vs predicted friction coefficients."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        ax1.scatter(true_mu, pred_mu, alpha=0.5)
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        ax1.set_xlabel("True μ")
        ax1.set_ylabel("Predicted μ")
        ax1.set_title("Prediction vs Ground Truth")
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Error histogram
        errors = pred_mu - true_mu
        ax2.hist(errors, bins=50, alpha=0.7)
        ax2.set_xlabel("Prediction Error")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Prediction Error Distribution")
        ax2.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / "friction_prediction.png")
        plt.show()
    
    def plot_surface_classification(self, images: np.ndarray, 
                                   true_labels: np.ndarray, 
                                   pred_labels: np.ndarray,
                                   class_names: List[str],
                                   num_samples: int = 16) -> None:
        """Plot sample images with true and predicted labels."""
        # Denormalize images if needed
        if images.min() < 0:
            images = (images * np.array([0.229, 0.224, 0.225]) + 
                     np.array([0.485, 0.456, 0.406])) * 255
            images = images.astype(np.uint8)
        
        # Select random samples
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            axes[i].imshow(images[idx])
            true_name = class_names[true_labels[idx]]
            pred_name = class_names[pred_labels[idx]]
            color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
            axes[i].set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
            axes[i].axis('off')
        
        plt.suptitle("Surface Classification Samples")
        plt.tight_layout()
        plt.savefig(self.output_dir / "surface_classification.png")
        plt.show()
    
    def plot_braking_simulation(self, times: List[np.ndarray], velocities: List[np.ndarray],
                                braking_forces: List[np.ndarray], surface_types: List[str],
                                title: str = "Braking Simulation Results") -> None:
        """Plot braking simulation results for different surfaces."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Velocity plot
        for i, surface in enumerate(surface_types):
            ax1.plot(times[i], velocities[i], label=surface, linewidth=2)
        
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Velocity [m/s]")
        ax1.set_title("Velocity over Time")
        ax1.legend()
        ax1.grid(True)
        
        # Braking force plot
        for i, surface in enumerate(surface_types):
            ax2.plot(times[i], braking_forces[i], label=surface, linewidth=2)
        
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Braking Force")
        ax2.set_title("Braking Force over Time")
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / "braking_simulation.png")
        plt.show()


class SystemAnalyzer:
    """Analyze system performance and generate reports."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str],
                             title: str = "Confusion Matrix") -> None:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Force a fixed label space so matrix dimensions always match class_names.
        labels = list(range(len(class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.show()
    
    def plot_feature_importance(self, importance: np.ndarray, 
                               feature_names: List[str],
                               title: str = "Feature Importance") -> None:
        """Plot feature importance scores."""
        indices = np.argsort(importance)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.bar(range(len(importance)), importance[indices])
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
        ax.set_title(title)
        ax.set_ylabel("Importance Score")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png")
        plt.show()
    
    def plot_latency_analysis(self, latencies: Dict[str, List[float]],
                            title: str = "System Latency Analysis") -> None:
        """Plot latency distribution for different components."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (component, times) in enumerate(latencies.items()):
            if i >= 4:
                break
            
            # Histogram
            axes[i].hist(times, bins=50, alpha=0.7)
            axes[i].set_title(component)
            axes[i].set_xlabel("Latency [ms]")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True)
            
            # Add mean and std
            mean = np.mean(times)
            std = np.std(times)
            axes[i].axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f} ms')
            axes[i].axvline(mean + std, color='g', linestyle='--', label=f'Mean+Std: {mean+std:.2f} ms')
            axes[i].legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_analysis.png")
        plt.show()


class RealTimeVisualizer:
    """Real-time visualization for system monitoring."""
    
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.ion()
        
        # Initialize plots
        self.lines = {}
        self.data = {}
        
        # Camera view
        self.camera_ax = self.axes[0, 0]
        self.camera_ax.set_title("Camera View")
        self.camera_ax.axis('off')
        self.camera_img = None
        
        # Friction estimation
        self.friction_ax = self.axes[0, 1]
        self.friction_ax.set_title("Friction Estimation")
        self.friction_ax.set_xlabel("Time [s]")
        self.friction_ax.set_ylabel("μ")
        self.friction_ax.set_ylim([0, 1])
        self.friction_line, = self.friction_ax.plot([], [], 'b-')
        
        # Vehicle state
        self.state_ax = self.axes[1, 0]
        self.state_ax.set_title("Vehicle State")
        self.state_ax.set_xlabel("Time [s]")
        self.state_ax.set_ylabel("Value")
        self.state_lines = {}
        
        # Braking force
        self.braking_ax = self.axes[1, 1]
        self.braking_ax.set_title("Braking Force")
        self.braking_ax.set_xlabel("Time [s]")
        self.braking_ax.set_ylabel("Force")
        self.braking_ax.set_ylim([0, 1])
        self.braking_line, = self.braking_ax.plot([], [], 'r-')
        
        # Initialize data buffers
        self.time_buffer = []
        self.max_points = 100
    
    def update(self, frame: Dict) -> None:
        """
        Update visualization with new frame data.
        
        Args:
            frame: Dictionary containing:
                - image: Camera image
                - mu: Estimated friction
                - v_x: Vehicle velocity
                - a_x: Longitudinal acceleration
                - braking_force: Current braking force
                - timestamp: Frame timestamp
        """
        # Update time buffer
        if 'timestamp' in frame:
            self.time_buffer.append(frame['timestamp'])
            if len(self.time_buffer) > self.max_points:
                self.time_buffer.pop(0)
        
        # Update camera view
        if 'image' in frame:
            if self.camera_img is None:
                self.camera_img = self.camera_ax.imshow(frame['image'])
            else:
                self.camera_img.set_data(frame['image'])
        
        # Update friction plot
        if 'mu' in frame and len(self.time_buffer) > 0:
            times = np.array(self.time_buffer) - self.time_buffer[0]
            mu = frame['mu']
            self.friction_line.set_data(times, [mu] * len(times))
            self.friction_ax.relim()
            self.friction_ax.autoscale_view()
        
        # Update braking force plot
        if 'braking_force' in frame and len(self.time_buffer) > 0:
            times = np.array(self.time_buffer) - self.time_buffer[0]
            force = frame['braking_force']
            self.braking_line.set_data(times, [force] * len(times))
            self.braking_ax.relim()
            self.braking_ax.autoscale_view()
        
        # Redraw
        plt.draw()
        plt.pause(0.01)
    
    def close(self):
        """Close the visualization."""
        plt.ioff()
        plt.close()


# Utility functions
def save_figure(fig: Figure, filename: str, output_dir: str = "plots") -> None:
    """Save a figure to file."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gradient_flow(model: torch.nn.Module, output_dir: str = "plots") -> None:
    """Plot gradient flow through the network."""
    # Get gradients
    gradients = []
    layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.abs().mean().item())
            layers.append(name)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(layers, gradients)
    ax.set_xlabel("Gradient Magnitude")
    ax.set_title("Gradient Flow")
    
    save_figure(fig, "gradient_flow.png", output_dir)


def plot_weight_distribution(model: torch.nn.Module, output_dir: str = "plots") -> None:
    """Plot weight distribution for all layers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= 4:
            break
        
        weights = param.data.cpu().numpy().flatten()
        axes[i].hist(weights, bins=50, alpha=0.7)
        axes[i].set_title(name)
        axes[i].set_xlabel("Weight Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True)
    
    plt.suptitle("Weight Distributions")
    plt.tight_layout()
    save_figure(fig, "weight_distribution.png", output_dir)


# Example usage
if __name__ == "__main__":
    # Example 1: Plot training curve
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.2, 0.15],
        'val_loss': [0.55, 0.45, 0.35, 0.25, 0.2],
        'train_acc': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_acc': [0.55, 0.65, 0.75, 0.8, 0.85]
    }
    plotter = TrainingPlotter()
    plotter.plot_training_curve(history)
    
    # Example 2: Plot friction prediction
    true_mu = np.random.rand(100) * 0.8 + 0.1
    pred_mu = true_mu + np.random.randn(100) * 0.05
    pred_mu = np.clip(pred_mu, 0, 1)
    visualizer = PredictionVisualizer()
    visualizer.plot_friction_prediction(true_mu, pred_mu)
    
    # Example 3: Plot braking simulation
    times = [np.linspace(0, 5, 100) for _ in range(4)]
    velocities = [
        np.maximum(20 - 4 * t, 0) for t in times
    ]
    braking_forces = [
        np.clip(0.8 * (1 - t/5), 0, 1) for t in times
    ]
    surface_types = ['dry', 'wet', 'icy', 'rough']
    visualizer.plot_braking_simulation(times, velocities, braking_forces, surface_types)

