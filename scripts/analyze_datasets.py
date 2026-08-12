"""
Comprehensive dataset analysis script.
Generates statistics, visualizations, and reports for all datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import MultiModalBrakingDataset
from data.external_datasets import (
    THURoadSurfaceDataset,
    MendeleyRoadSurfaceDataset,
    DAWNWeatherDataset,
    BDD100KDataset,
    KITTIRawDataset,
)
from utils.visualization import SystemAnalyzer, PredictionVisualizer


class DatasetAnalyzer:
    """Analyze dataset statistics and generate reports."""
    
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = SystemAnalyzer(output_dir)
    
    def analyze_thu_dataset(self, data_dir: str) -> Dict:
        """Analyze Tsinghua Road Surface Dataset."""
        print("\n" + "="*70)
        print("ANALYZING TSINGHUA ROAD SURFACE DATASET")
        print("="*70)
        
        dataset = THURoadSurfaceDataset(data_dir)
        
        # Basic statistics
        stats = {
            'total_samples': len(dataset),
            'classes': {},
            'mu_distribution': {}
        }
        
        # Class distribution
        for sample in dataset.samples:
            surface = sample['surface_type']
            mu = sample['mu']
            
            if surface not in stats['classes']:
                stats['classes'][surface] = 0
            stats['classes'][surface] += 1
            
            if surface not in stats['mu_distribution']:
                stats['mu_distribution'][surface] = []
            stats['mu_distribution'][surface].append(mu)
        
        # Print statistics
        print(f"\nTotal Samples: {stats['total_samples']}")
        print("\nClass Distribution:")
        for surface, count in sorted(stats['classes'].items()):
            pct = 100 * count / stats['total_samples']
            print(f"  {surface:10s}: {count:6d} ({pct:5.1f}%)")
        
        print("\nFriction Coefficient Statistics by Class:")
        for surface in sorted(stats['mu_distribution'].keys()):
            mu_values = np.array(stats['mu_distribution'][surface])
            print(f"  {surface:10s}: min={mu_values.min():.3f}, max={mu_values.max():.3f}, "
                  f"mean={mu_values.mean():.3f}, std={mu_values.std():.3f}")
        
        # Generate visualizations
        self._plot_class_distribution(stats['classes'], "thu_class_distribution")
        self._plot_mu_distribution(stats['mu_distribution'], "thu_mu_distribution")
        
        return stats
    
    def analyze_mendeley_dataset(self, data_dir: str) -> Dict:
        """Analyze Mendeley road-surface dataset (train/test split folders)."""
        print("\n" + "="*70)
        print("ANALYZING MENDELEY VEHICLE DATASET")
        print("="*70)

        dataset = MendeleyRoadSurfaceDataset(data_dir, split='train-set')

        stats = {
            'total_samples': len(dataset),
            'classes': {},
            'mu_distribution': {},
            'can_data': pd.DataFrame(),
            'signal_stats': {}
        }

        for sample in dataset.samples:
            surface = sample['surface_type']
            mu = sample['mu']
            stats['classes'][surface] = stats['classes'].get(surface, 0) + 1
            stats['mu_distribution'].setdefault(surface, []).append(mu)

        print(f"\nTotal Mendeley Train Samples: {stats['total_samples']}")
        print("\nClass Distribution:")
        for surface, count in sorted(stats['classes'].items()):
            pct = 100 * count / max(1, stats['total_samples'])
            print(f"  {surface:25s}: {count:8d} ({pct:5.2f}%)")

        self._plot_class_distribution(stats['classes'], "mendeley_class_distribution")
        self._plot_mu_distribution(stats['mu_distribution'], "mendeley_mu_distribution")

        can_path = os.path.join(data_dir, "can_data.csv")
        if not os.path.exists(can_path):
            return stats

        can_data = pd.read_csv(can_path)
        stats['can_data'] = can_data
        if can_data.empty:
            return stats

        print(f"\nTotal CAN Messages: {len(can_data)}")
        if 'timestamp' in can_data.columns:
            print(f"Time Range: {can_data['timestamp'].min()} to {can_data['timestamp'].max()}")

        important_signals = ['a_x', 'a_y', 'v_x', 'omega_FL', 'omega_FR', 
                            'omega_RL', 'omega_RR', 'steering_angle', 
                            'brake_pressure', 'yaw_rate', 'mu']
        
        print("\nSignal Statistics:")
        for signal in important_signals:
            if signal in can_data.columns:
                values = can_data[signal].dropna()
                if len(values) > 0:
                    stats['signal_stats'][signal] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'missing': int(can_data[signal].isna().sum())
                    }
                    print(f"  {signal:15s}: min={values.min():8.3f}, max={values.max():8.3f}, "
                          f"mean={values.mean():8.3f}, std={values.std():8.3f}, "
                          f"missing={can_data[signal].isna().sum()}")
        
        if 'mu' in can_data.columns:
            mu_values = can_data['mu'].dropna().values
            if len(mu_values) > 0:
                stats['mu_values'] = mu_values

        self._plot_signal_correlations(can_data, important_signals)

        return stats

    def analyze_bdd100k_dataset(self, data_dir: str) -> Dict:
        """Analyse BDD100K image dataset."""
        print("\n" + "="*70)
        print("ANALYZING BDD100K DATASET")
        print("="*70)

        dataset = BDD100KDataset(data_dir)

        stats: Dict = {'total_samples': len(dataset), 'classes': {}, 'mu_distribution': {}}
        for sample in dataset.samples:
            surface = sample['surface_type']
            mu = sample['mu']
            stats['classes'][surface] = stats['classes'].get(surface, 0) + 1
            stats['mu_distribution'].setdefault(surface, []).append(mu)

        print(f"\nTotal Samples: {stats['total_samples']}")
        print("\nCondition Distribution:")
        for surface, count in sorted(stats['classes'].items()):
            pct = 100 * count / max(1, stats['total_samples'])
            print(f"  {surface:30s}: {count:8d} ({pct:5.2f}%)")

        self._plot_class_distribution(stats['classes'], "bdd100k_class_distribution")
        self._plot_mu_distribution(stats['mu_distribution'], "bdd100k_mu_distribution")
        return stats

    def analyze_kitti_dataset(self, data_dir: str) -> Dict:
        """Analyse KITTI raw image dataset."""
        print("\n" + "="*70)
        print("ANALYZING KITTI RAW DATASET")
        print("="*70)

        dataset = KITTIRawDataset(data_dir)

        stats: Dict = {'total_samples': len(dataset), 'classes': {}, 'mu_distribution': {}}
        for sample in dataset.samples:
            surface = sample['surface_type']
            mu = sample['mu']
            stats['classes'][surface] = stats['classes'].get(surface, 0) + 1
            stats['mu_distribution'].setdefault(surface, []).append(mu)

        print(f"\nTotal Frames: {stats['total_samples']}")
        print("\nSurface Distribution:")
        for surface, count in sorted(stats['classes'].items()):
            pct = 100 * count / max(1, stats['total_samples'])
            print(f"  {surface:30s}: {count:8d} ({pct:5.2f}%)")

        self._plot_class_distribution(stats['classes'], "kitti_class_distribution")
        self._plot_mu_distribution(stats['mu_distribution'], "kitti_mu_distribution")
        return stats
    
    def analyze_custom_dataset(self, data_dir: str) -> Dict:
        """Analyze custom dataset."""
        print("\n" + "="*70)
        print("ANALYZING CUSTOM DATASET")
        print("="*70)
        
        dataset = MultiModalBrakingDataset(data_dir)
        
        stats = {
            'total_samples': len(dataset),
            'images': len([f for f in os.listdir(os.path.join(data_dir, "images")) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))])
        }
        
        print(f"\nTotal Samples: {stats['total_samples']}")
        print(f"Total Images: {stats['images']}")
        
        return stats

    def analyze_dawn_dataset(self, data_dir: str) -> Dict:
        """Analyze DAWN adverse-weather dataset."""
        print("\n" + "="*70)
        print("ANALYZING DAWN WEATHER DATASET")
        print("="*70)

        dataset = DAWNWeatherDataset(data_dir)
        stats = {
            'total_samples': len(dataset),
            'classes': {},
            'mu_distribution': {}
        }

        for sample in dataset.samples:
            weather = sample['surface_type']
            mu = sample['mu']
            stats['classes'][weather] = stats['classes'].get(weather, 0) + 1
            stats['mu_distribution'].setdefault(weather, []).append(mu)

        print(f"\nTotal DAWN Samples: {stats['total_samples']}")
        print("\nWeather Distribution:")
        for weather, count in sorted(stats['classes'].items()):
            pct = 100 * count / max(1, stats['total_samples'])
            print(f"  {weather:10s}: {count:6d} ({pct:5.1f}%)")

        self._plot_class_distribution(stats['classes'], "dawn_class_distribution")
        self._plot_mu_distribution(stats['mu_distribution'], "dawn_mu_distribution")
        return stats
    
    def _plot_class_distribution(self, class_counts: Dict[str, int], filename: str) -> None:
        """Plot class distribution."""
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        n = len(classes)

        # Scale figure width so labels don't overlap (min 10, 0.6 per class)
        fig_w = max(10, n * 0.65)
        fig, ax = plt.subplots(figsize=(fig_w, 7))
        
        bars = ax.bar(range(n), counts, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = 100 * count / total
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)
        
        ax.set_xticks(range(n))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Road Surface Type")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Class Distribution")
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=120)
        plt.close()
    
    def _plot_mu_distribution(self, mu_dist: Dict[str, List[float]], filename: str) -> None:
        """Plot friction coefficient distribution by class."""
        labels = sorted(mu_dist.keys())
        data = [mu_dist[s] for s in labels]
        n = len(labels)

        fig_w = max(10, n * 0.65)
        fig, ax = plt.subplots(figsize=(fig_w, 7))
        
        bp = ax.boxplot(data, patch_artist=True, positions=range(1, n + 1))
        # Set x-tick labels explicitly for compatibility across matplotlib versions
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        # Customize box colors
        colors = ['lightblue', 'lightgreen', 'pink', 'lightyellow', 'lavender',
                  'wheat', 'lightcoral', 'palegreen']
        for patch, color in zip(bp.get('boxes', []), colors * (n // len(colors) + 1)):
            patch.set_facecolor(color)
        
        ax.set_xlabel("Road Surface Type")
        ax.set_ylabel("Friction Coefficient (μ)")
        ax.set_title("Friction Coefficient Distribution by Surface Type")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=120)
        plt.close()
    
    def _plot_signal_correlations(self, can_data: pd.DataFrame, signals: List[str]) -> None:
        """Plot correlation matrix for CAN signals."""
        # Select only available signals
        available_signals = [s for s in signals if s in can_data.columns]
        if len(available_signals) < 2:
            return
        
        # Compute correlation matrix
        corr_matrix = can_data[available_signals].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(corr_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        
        fig.colorbar(cax)
        
        # Add signal names
        ax.set_xticks(range(len(available_signals)))
        ax.set_yticks(range(len(available_signals)))
        ax.set_xticklabels(available_signals, rotation=45, ha='left')
        ax.set_yticklabels(available_signals)
        
        # Add correlation values
        for i in range(len(available_signals)):
            for j in range(len(available_signals)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.7 else 'white')
        
        ax.set_title("CAN Signal Correlation Matrix")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "signal_correlations.png")
        plt.close()
    
    def generate_report(self, thu_stats: Dict, mendeley_stats: Dict,
                       dawn_stats: Dict, custom_stats: Dict) -> None:
        """Generate HTML report."""
        report_path = self.output_dir / "dataset_analysis_report.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            report_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dataset Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>Dataset Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Tsinghua Road Surface Dataset</h2>
                <div class="stats">
                    <h3>Statistics</h3>
                    <p><strong>Total Samples:</strong> {thu_stats.get('total_samples', 0)}</p>
                    <h4>Class Distribution:</h4>
                    <table>
                        <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
                        {''.join(f'<tr><td>{k}</td><td>{v}</td><td>{100*v/thu_stats.get("total_samples",1):.1f}%</td></tr>' for k, v in sorted(thu_stats.get('classes', {}).items()))}
                    </table>
                    <h4>Friction Coefficient by Class:</h4>
                    <table>
                        <tr><th>Class</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr>
                        {''.join(f'<tr><td>{k}</td><td>{np.min(vs):.3f}</td><td>{np.max(vs):.3f}</td><td>{np.mean(vs):.3f}</td><td>{np.std(vs):.3f}</td></tr>' for k, vs in sorted(thu_stats.get('mu_distribution', {}).items()))}
                    </table>
                </div>
                <h3>Visualizations</h3>
                <img src="thu_class_distribution.png" alt="Class Distribution">
                <img src="thu_mu_distribution.png" alt="Friction Distribution">
                
                <h2>Mendeley Vehicle Dataset</h2>
                <div class="stats">
                    <h3>Statistics</h3>
                    <p><strong>Total CAN Messages:</strong> {len(mendeley_stats.get('can_data', pd.DataFrame()))}</p>
                    <p><strong>Duration:</strong> {self._safe_timestamp_duration(mendeley_stats.get('can_data', pd.DataFrame())):.2f} seconds</p>
                    <p><strong>Sample Rate:</strong> {len(mendeley_stats.get('can_data', pd.DataFrame())) / max(self._safe_timestamp_duration(mendeley_stats.get('can_data', pd.DataFrame())), 1):.1f} Hz</p>
                    <h4>Signal Statistics:</h4>
                    <table>
                        <tr><th>Signal</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th><th>Missing</th></tr>
                        {''.join(f'<tr><td>{k}</td><td>{v["min"]:.3f}</td><td>{v["max"]:.3f}</td><td>{v["mean"]:.3f}</td><td>{v["std"]:.3f}</td><td>{v["missing"]}</td></tr>' for k, v in mendeley_stats.get('signal_stats', {}).items())}
                    </table>
                </div>
                <h3>Visualizations</h3>
                <img src="mendeley_class_distribution.png" alt="Mendeley Class Distribution">
                <img src="mendeley_mu_distribution.png" alt="Friction Distribution">
                <img src="signal_correlations.png" alt="Signal Correlations">

                <h2>DAWN Weather Dataset</h2>
                <div class="stats">
                    <h3>Statistics</h3>
                    <p><strong>Total Samples:</strong> {dawn_stats.get('total_samples', 0)}</p>
                    <h4>Weather Distribution:</h4>
                    <table>
                        <tr><th>Weather</th><th>Count</th><th>Percentage</th></tr>
                        {''.join(f'<tr><td>{k}</td><td>{v}</td><td>{100*v/max(dawn_stats.get("total_samples",1),1):.1f}%</td></tr>' for k, v in sorted(dawn_stats.get('classes', {}).items()))}
                    </table>
                </div>
                <h3>Visualizations</h3>
                <img src="dawn_class_distribution.png" alt="DAWN Weather Distribution">
                <img src="dawn_mu_distribution.png" alt="DAWN Estimated Friction">
                
                <h2>Custom Dataset</h2>
                <div class="stats">
                    <p><strong>Total Samples:</strong> {custom_stats.get('total_samples', 0)}</p>
                    <p><strong>Total Images:</strong> {custom_stats.get('images', 0)}</p>
                </div>
                
                <h2>Recommendations</h2>
                <div class="stats">
                    <p>{self._generate_recommendations(thu_stats, mendeley_stats, dawn_stats, custom_stats)}</p>
                </div>
            </body>
            </html>
            """
            f.write(report_html)
        
        print(f"\n Report generated: {report_path}")
    
    def _safe_timestamp_duration(self, can_data: pd.DataFrame) -> float:
        """Safely compute the duration of CAN data when a timestamp column exists."""
        if can_data is None or can_data.empty:
            return 0.0
        if 'timestamp' not in can_data.columns:
            return 0.0
        if not pd.api.types.is_datetime64_any_dtype(can_data['timestamp']):
            try:
                parsed = pd.to_datetime(can_data['timestamp'])
                if parsed.notna().any():
                    return float((parsed.max() - parsed.min()).total_seconds())
            except Exception:
                return 0.0
            return 0.0
        return float((can_data['timestamp'].max() - can_data['timestamp'].min()).total_seconds())

    def _generate_recommendations(self, thu_stats: Dict, mendeley_stats: Dict,
                                 dawn_stats: Dict, custom_stats: Dict) -> str:
        """Generate recommendations based on dataset analysis."""
        recommendations = []
        
        # Check data balance
        if 'classes' in thu_stats:
            total = thu_stats['total_samples']
            for surface, count in thu_stats['classes'].items():
                pct = 100 * count / total
                if pct < 5:
                    recommendations.append(
                        f"[WARNING] Class '{surface}' has only {pct:.1f}% of samples. Consider data augmentation."
                    )
        
        # Check for missing data
        if 'signal_stats' in mendeley_stats:
            for signal, stats in mendeley_stats['signal_stats'].items():
                if stats['missing'] > 0:
                    pct_missing = 100 * stats['missing'] / len(mendeley_stats['can_data'])
                    if pct_missing > 10:
                        recommendations.append(
                            f"[WARNING] Signal '{signal}' has {pct_missing:.1f}% missing values. Consider imputation."
                        )
        
        # Check friction range
        if 'mu_values' in mendeley_stats:
            mu_min = np.min(mendeley_stats['mu_values'])
            mu_max = np.max(mendeley_stats['mu_values'])
            if mu_min < 0.05:
                recommendations.append(
                    "[WARNING] Some friction values are very low (<0.05). Verify these are correct."
                )
            if mu_max > 1.0:
                recommendations.append(
                    "[WARNING] Some friction values are >1.0. Verify these are correct (μ should be ≤1)."
                )

        if 'classes' in dawn_stats and dawn_stats.get('total_samples', 0) > 0:
            total = dawn_stats['total_samples']
            for weather, count in dawn_stats['classes'].items():
                pct = 100 * count / total
                if pct < 10:
                    recommendations.append(
                        f"[WARNING] DAWN class '{weather}' has only {pct:.1f}% of samples. Consider balancing."
                    )
        
        if not recommendations:
            recommendations.append("[OK] Datasets look good! No major issues detected.")
        
        return '<br>'.join(recommendations)


def main():
    """Main analysis function."""
    analyzer = DatasetAnalyzer()
    
    # Analyze all datasets
    thu_stats = {}
    mendeley_stats = {}
    dawn_stats = {}
    custom_stats = {}
    
    # Check if datasets exist
    thu_dir = "data/external/thu_road_surface"
    mendeley_dir = "data/external/mendeley_vehicle"
    dawn_dir = "data/external/dawn"
    custom_dir = "data/train"
    
    if os.path.exists(thu_dir):
        thu_stats = analyzer.analyze_thu_dataset(thu_dir)
    else:
        print(f"[WARNING] Tsinghua dataset not found at: {os.path.abspath(thu_dir)}")
    
    if os.path.exists(mendeley_dir):
        mendeley_stats = analyzer.analyze_mendeley_dataset(mendeley_dir)
    else:
        print(f"[WARNING] Mendeley dataset not found at: {os.path.abspath(mendeley_dir)}")

    if os.path.exists(dawn_dir):
        dawn_stats = analyzer.analyze_dawn_dataset(dawn_dir)
    else:
        print(f"[WARNING] DAWN dataset not found at: {os.path.abspath(dawn_dir)}")
    
    if os.path.exists(custom_dir):
        custom_stats = analyzer.analyze_custom_dataset(custom_dir)
    else:
        print(f"[INFO] Custom dataset not found at: {os.path.abspath(custom_dir)}")

    bdd_dir = "data/external/bdd100k"
    kitti_dir = "data/external/kitti_raw"
    if os.path.exists(bdd_dir):
        try:
            analyzer.analyze_bdd100k_dataset(bdd_dir)
        except Exception as e:
            print(f"[WARNING] BDD100K analysis failed: {e}")
    else:
        print(f"[INFO] BDD100K dataset not found at: {os.path.abspath(bdd_dir)}")

    if os.path.exists(kitti_dir):
        try:
            analyzer.analyze_kitti_dataset(kitti_dir)
        except Exception as e:
            print(f"[WARNING] KITTI analysis failed: {e}")
    else:
        print(f"[INFO] KITTI dataset not found at: {os.path.abspath(kitti_dir)}")
    
    # Generate report
    analyzer.generate_report(thu_stats, mendeley_stats, dawn_stats, custom_stats)
    
    print("\n" + "="*70)
    print("[OK] Dataset analysis complete!")
    print(f"Report saved to: {analyzer.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()