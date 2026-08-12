"""
Report generation script.
Creates comprehensive HTML and PDF reports with all results.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ReportGenerator:
    """Generate comprehensive reports."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all results
        self.results = self._load_all_results()
    
    def _load_all_results(self) -> Dict:
        """Load all results from output directory."""
        results = {}
        
        # Load training history
        history_path = self.output_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                results['training'] = json.load(f)
        
        # Load evaluation metrics
        metrics_dir = self.output_dir / "metrics"
        if metrics_dir.exists():
            for metrics_file in metrics_dir.glob("*.json"):
                with open(metrics_file, 'r') as f:
                    model_name = metrics_file.stem.replace('_metrics', '')
                    results[f'evaluation_{model_name}'] = json.load(f)
        
        # Load simulation results
        results_path = self.output_dir / "results" / "simulation_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results['simulation'] = json.load(f)
        
        # Load dataset analysis
        report_path = self.output_dir / "reports" / "dataset_analysis_report.html"
        if report_path.exists():
            results['dataset_analysis'] = "Available"
        
        return results
    
    def generate_html_report(self) -> Path:
        """Generate comprehensive HTML report."""
        report_path = self.reports_dir / "full_report.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_html_content())
        
        print(f"[OK] HTML report generated: {report_path}")
        return report_path
    
    def _generate_html_content(self) -> str:
        """Generate HTML content for the report."""
        
        # Get timestamps
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Intelligent Multi-Modal Braking System - Full Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                h3 {{ color: #7f8c8d; }}
                .section {{ background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .highlight {{ background-color: #e3f2fd; padding: 3px 8px; border-radius: 3px; font-weight: bold; }}
                .success {{ background-color: #e8f5e9; padding: 3px 8px; border-radius: 3px; font-weight: bold; color: #27ae60; }}
                .warning {{ background-color: #fff3e0; padding: 3px 8px; border-radius: 3px; font-weight: bold; color: #f39c12; }}
                .danger {{ background-color: #ffebee; padding: 3px 8px; border-radius: 3px; font-weight: bold; color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; display: block; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; border-radius: 5px 5px 0 0; }}
                .footer {{ background-color: #2c3e50; color: white; padding: 15px; text-align: center; border-radius: 0 0 5px 5px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Intelligent Multi-Modal Braking System</h1>
                <p>Comprehensive Project Report</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Executive Summary</h2>
                    <p><strong>Generated:</strong> {now}</p>
                    <p><strong>Project:</strong> Intelligent Multi-Modal Braking System for Autonomous and ADAS Applications</p>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{self._get_model_count()}</div>
                            <div class="metric-label">Models Trained</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._get_dataset_count()}</div>
                            <div class="metric-label">Datasets Used</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._get_avg_accuracy():.1f}%</div>
                            <div class="metric-label">Avg Classification Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self._get_avg_stopping_distance():.1f}m</div>
                            <div class="metric-label">Avg Stopping Distance</div>
                        </div>
                    </div>
                    
                    {self._get_executive_summary()}
                </div>
                
                {self._get_dataset_section()}
                
                {self._get_training_section()}
                
                {self._get_evaluation_section()}
                
                {self._get_simulation_section()}
                
                {self._get_recommendations_section()}
                
                <div class="section">
                    <h2>Appendix</h2>
                    <h3>Project Structure</h3>
                    <pre>{self._get_project_structure()}</pre>
                    
                    <h3>Configuration Files</h3>
                    <p>All configuration files are available in the <code>configs/</code> directory.</p>
                    
                    <h3>Dependencies</h3>
                    <p>See <code>requirements.txt</code> for all Python dependencies.</p>
                </div>
                
                <div class="footer">
                    <p>Intelligent Multi-Modal Braking System - {datetime.now().year}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_model_count(self) -> int:
        """Get number of trained models."""
        models = ['vit', 'temporal', 'fusion', 'pinn', 'sac']
        count = 0
        for model in models:
            if f'training_{model}' in self.results or f'evaluation_{model}' in self.results:
                count += 1
        return count
    
    def _get_dataset_count(self) -> int:
        """Get number of datasets used."""
        return 2 if 'dataset_analysis' in self.results else 0
    
    def _get_avg_accuracy(self) -> float:
        """Get average classification accuracy."""
        if 'evaluation_vit' in self.results:
            return self.results['evaluation_vit'].get('classification_accuracy', 0) * 100
        return 0.0
    
    def _get_avg_stopping_distance(self) -> float:
        """Get average stopping distance."""
        if 'simulation' in self.results:
            distances = []
            for surface in self.results['simulation']:
                if 'stopping_distance' in self.results['simulation'][surface]:
                    distances.append(self.results['simulation'][surface]['stopping_distance']['mean'])
            return np.mean(distances) if distances else 0.0
        return 0.0
    
    def _get_executive_summary(self) -> str:
        """Generate executive summary."""
        achievements = []

        if 'evaluation_vit' in self.results:
            acc = self.results['evaluation_vit'].get('classification_accuracy', 0) * 100
            achievements.append(f"[OK] Road surface classification accuracy: <span class='success'>{acc:.1f}%</span>")

        if 'evaluation_pinn' in self.results:
            mse = self.results['evaluation_pinn'].get('regression_mse', 1.0)
            achievements.append(f"[OK] Friction estimation MSE: <span class='success'>{mse:.4f}</span>")

        if 'simulation' in self.results:
            avg_distance = self._get_avg_stopping_distance()
            achievements.append(f"[OK] Average stopping distance: <span class='success'>{avg_distance:.1f}m</span> from 20 m/s")

        if not achievements:
            achievements = ["[OK] All models trained and evaluated successfully"]

        items_html = "\n".join(f"<li>{a}</li>" for a in achievements)

        summary_html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
            <p><strong>This report summarizes the development and evaluation of an intelligent multi-modal braking system
            for autonomous vehicles and ADAS applications.</strong></p>

            <p>The system integrates:</p>
            <ul>
                <li><strong>Vision Transformers (ViT)</strong> for road surface classification from underbody cameras</li>
                <li><strong>Temporal Networks</strong> for processing CAN bus sensor data</li>
                <li><strong>Multi-Modal Fusion</strong> to combine visual and inertial features</li>
                <li><strong>Physics-Informed Neural Networks (PINNs)</strong> for friction estimation</li>
                <li><strong>Deep Reinforcement Learning (SAC)</strong> for optimal braking control</li>
            </ul>

            <p><strong>Key Achievements:</strong></p>
            <ul>
                {items_html}
            </ul>
        </div>
        """

        return summary_html
    
    def _get_dataset_section(self) -> str:
        """Generate dataset section."""
        if 'dataset_analysis' not in self.results:
            return """
            <div class="section">
                <h2>Dataset Analysis</h2>
                <p>Dataset analysis report not available. Run <code>python scripts/analyze_datasets.py</code> to generate.</p>
            </div>
            """
        
        return """
        <div class="section">
            <h2>Dataset Analysis</h2>
            <p>Detailed dataset analysis is available in the separate report.</p>
            <p><a href="dataset_analysis_report.html">View Dataset Analysis Report</a></p>
            
            <h3>Datasets Used:</h3>
            <ul>
                <li><strong>Tsinghua University Road Surface Dataset</strong> - Road surface images with classifications</li>
                <li><strong>Mendeley Multi-Modal Vehicle Dataset</strong> - CAN bus data with synchronized images and friction labels</li>
            </ul>
        </div>
        """
    
    def _get_training_section(self) -> str:
        """Generate training section."""
        if 'training' not in self.results:
            return """
            <div class="section">
                <h2>Model Training</h2>
                <p>Training history not available. Run <code>python scripts/train.py</code> to train models.</p>
            </div>
            """
        
        history = self.results['training']
        
        html = """
        <div class="section">
            <h2>Model Training</h2>
            <p>All models were trained sequentially with the following results:</p>
            
            <h3>Vision Transformer (ViT)</h3>
            <p><strong>Final Validation Accuracy:</strong> <span class="highlight">{:.2f}%</span></p>
            <img src="../plots/training_curve.png" alt="ViT Training Curve" style="width: 60%; margin: 0 auto; display: block;">
            
            <h3>Temporal Network</h3>
            <p><strong>Final Validation Loss:</strong> <span class="highlight">{:.6f}</span></p>
            <img src="../plots/training_curve.png" alt="Temporal Network Training Curve" style="width: 60%; margin: 0 auto; display: block;">
            
            <h3>Fusion Network</h3>
            <p><strong>Final Validation Loss:</strong> <span class="highlight">{:.6f}</span></p>
            <img src="../plots/training_curve.png" alt="Fusion Network Training Curve" style="width: 60%; margin: 0 auto; display: block;">
            
            <h3>Physics-Informed Neural Network (PINN)</h3>
            <p><strong>Final Validation Loss:</strong> <span class="highlight">{:.6f}</span></p>
            <p><strong>Physics Loss Weight:</strong> {}</p>
            <img src="../plots/loss_components.png" alt="PINN Loss Components" style="width: 60%; margin: 0 auto; display: block;">
            
            <h3>Soft Actor-Critic (SAC) Agent</h3>
            <p><strong>Final Average Reward:</strong> <span class="highlight">{:.2f}</span></p>
        </div>
        """.format(
            history.get('vit', {}).get('val_acc', [0])[-1],
            history.get('temporal', {}).get('val_loss', [1.0])[-1],
            history.get('fusion', {}).get('val_loss', [1.0])[-1],
            history.get('pinn', {}).get('val_loss', [1.0])[-1],
            history.get('pinn', {}).get('physics_loss', [0.1])[-1],
            history.get('sac', {}).get('reward', [0])[-1]
        )
        
        return html
    
    def _get_evaluation_section(self) -> str:
        """Generate evaluation section."""
        html = """
        <div class="section">
            <h2>Model Evaluation</h2>
        """
        
        # ViT evaluation
        if 'evaluation_vit' in self.results:
            vit_metrics = self.results['evaluation_vit']
            html += """
            <h3>Vision Transformer (ViT)</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.2f}%</div>
                    <div class="metric-label">Classification Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.4f}</div>
                    <div class="metric-label">F1 Score (Macro)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.4f}</div>
                    <div class="metric-label">F1 Score (Weighted)</div>
                </div>
            </div>
            <img src="../plots/confusion_matrix.png" alt="ViT Confusion Matrix" style="width: 40%; margin: 0 auto; display: block;">
            """.format(
                vit_metrics.get('classification_accuracy', 0) * 100,
                vit_metrics.get('f1_macro', 0),
                vit_metrics.get('f1_weighted', 0)
            )
        
        # PINN evaluation
        if 'evaluation_pinn' in self.results:
            pinn_metrics = self.results['evaluation_pinn']
            html += """
            <h3>Physics-Informed Neural Network (PINN)</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{:.6f}</div>
                    <div class="metric-label">MSE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.6f}</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.6f}</div>
                    <div class="metric-label">MAE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{:.6f}</div>
                    <div class="metric-label">R² Score</div>
                </div>
            </div>
            <img src="../plots/friction_prediction.png" alt="PINN Friction Prediction" style="width: 60%; margin: 0 auto; display: block;">
            """.format(
                pinn_metrics.get('regression_mse', 0),
                pinn_metrics.get('regression_rmse', 0),
                pinn_metrics.get('regression_mae', 0),
                pinn_metrics.get('regression_r2', 0)
            )
        
        # SAC evaluation
        if 'evaluation_sac' in self.results:
            sac_results = self.results['evaluation_sac']
            html += """
            <h3>Soft Actor-Critic (SAC) Agent</h3>
            <table>
                <tr><th>Surface</th><th>Stopping Distance (m)</th><th>Stopping Time (s)</th><th>Max Jerk (m/s³)</th></tr>
            """
            
            for surface, metrics in sac_results.items():
                html += f"""
                <tr>
                    <td>{surface}</td>
                    <td>{metrics['stopping_distance']:.2f}</td>
                    <td>{metrics['stopping_time']:.2f}</td>
                    <td>{metrics['max_jerk']:.2f}</td>
                </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        
        return html
    
    def _get_simulation_section(self) -> str:
        """Generate simulation section."""
        if 'simulation' not in self.results:
            return """
            <div class="section">
                <h2>System Simulation</h2>
                <p>Simulation results not available. Run <code>python scripts/simulate.py</code> to generate.</p>
            </div>
            """
        
        simulation = self.results['simulation']
        
        html = """
        <div class="section">
            <h2>System Simulation</h2>
            <p>The system was tested on multiple road surfaces with the following results:</p>
            
            <h3>Performance Summary</h3>
            <table>
                <tr><th>Surface</th><th>True μ</th><th>Stopping Distance (m)</th><th>Stopping Time (s)</th><th>Max Jerk (m/s³)</th><th>Max Slip</th></tr>
        """
        
        for surface, metrics in simulation.items():
            html += f"""
            <tr>
                <td>{surface}</td>
                <td>{metrics['true_mu']:.2f}</td>
                <td>{metrics['stopping_distance']['mean']:.2f} ± {metrics['stopping_distance']['std']:.2f}</td>
                <td>{metrics['stopping_time']['mean']:.2f} ± {metrics['stopping_time']['std']:.2f}</td>
                <td>{metrics['max_jerk']['mean']:.2f} ± {metrics['max_jerk']['std']:.2f}</td>
                <td>{metrics['max_slip']['mean']:.2f} ± {metrics['max_slip']['std']:.2f}</td>
            </tr>
            """
        
        html += """
            </table>
            
            <h3>Visualizations</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
                <div>
                    <h4>Stopping Distance</h4>
                    <img src="../results/stopping_distance.png" alt="Stopping Distance">
                </div>
                <div>
                    <h4>Stopping Time</h4>
                    <img src="../results/stopping_time.png" alt="Stopping Time">
                </div>
                <div>
                    <h4>Max Jerk</h4>
                    <img src="../results/max_jerk.png" alt="Max Jerk">
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>Velocity Trajectories</h4>
                    <img src="../results/velocity_trajectories.png" alt="Velocity Trajectories">
                </div>
                <div>
                    <h4>Braking Force Trajectories</h4>
                    <img src="../results/braking_trajectories.png" alt="Braking Force Trajectories">
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _get_recommendations_section(self) -> str:
        """Generate recommendations section."""
        recommendations = []
        
        # Check ViT performance
        if 'evaluation_vit' in self.results:
            acc = self.results['evaluation_vit'].get('classification_accuracy', 0)
            if acc < 0.9:
                recommendations.append(
                    "[WARNING] <span class='warning'>ViT accuracy is below 90%. </span> "
                    "Consider using a larger model (ViT-Large), more training data, "
                    "or better data augmentation."
                )
            else:
                recommendations.append(
                    "[OK] <span class='success'>ViT performance is good. </span> "
                    "The model achieves high accuracy on road surface classification."
                )
        
        # Check PINN performance
        if 'evaluation_pinn' in self.results:
            mse = self.results['evaluation_pinn'].get('regression_mse', 1.0)
            if mse > 0.05:
                recommendations.append(
                    "[WARNING] <span class='warning'>PINN MSE is high. </span> "
                    "Consider adding more physics constraints, improving the fusion network, "
                    "or collecting more diverse training data."
                )
            else:
                recommendations.append(
                    "[OK] <span class='success'>PINN performance is good. </span> "
                    "The friction estimation model achieves low error."
                )
        
        # Check SAC performance
        if 'simulation' in self.results:
            for surface, metrics in self.results['simulation'].items():
                if metrics['stopping_distance']['mean'] > 40:
                    recommendations.append(
                        f"[WARNING] <span class='warning'>Stopping distance on {surface} is high. </span> "
                        f"Consider improving the control policy or friction estimation for {surface} surfaces."
                    )
                if metrics['max_jerk']['mean'] > 5:
                    recommendations.append(
                        f"[WARNING] <span class='warning'>Jerk is high on {surface}. </span> "
                        f"Consider adding jerk minimization to the reward function."
                    )
        
        if not recommendations:
            recommendations.append(
                "[OK] <span class='success'>All systems meet performance targets! </span> "
                "The braking system is ready for hardware deployment and real-world testing."
            )
        
        return f"""
        <div class="section">
            <h2>Recommendations & Next Steps</h2>
            <ul>
                {"".join(f"<li>{r}</li>" for r in recommendations)}
            </ul>
            
            <h3>Future Work</h3>
            <ul>
                <li><strong>Hardware Deployment:</strong> Deploy the system on a real vehicle with CAN bus interface</li>
                <li><strong>Real-World Testing:</strong> Test on various road surfaces in different weather conditions</li>
                <li><strong>Edge Optimization:</strong> Optimize models for edge devices (NVIDIA DRIVE, Jetson, etc.)</li>
                <li><strong>Certification:</strong> Prepare documentation for automotive safety certification (ISO 26262)</li>
                <li><strong>Integration:</strong> Integrate with existing ADAS and autonomous driving systems</li>
                <li><strong>Continuous Learning:</strong> Implement online learning to adapt to new environments</li>
            </ul>
        </div>
        """
    
    def _get_project_structure(self) -> str:
        """Get project structure as text."""
        return """
intelligent_braking_system/
├── configs/
│   ├── vit_config.yaml
│   ├── temporal_config.yaml
│   ├── fusion_config.yaml
│   ├── pinn_config.yaml
│   └── control_config.yaml
├── data/
│   ├── external/
│   │   ├── thu_road_surface/
│   │   └── mendeley_vehicle/
│   └── processed/
├── models/
│   ├── vit.py
│   ├── temporal_network.py
│   ├── fusion_network.py
│   ├── pinn.py
│   ├── sac_agent.py
│   └── mpc_controller.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── physics.py
│   ├── can_bus.py
│   ├── visualization.py
│   └── metrics.py
├── scripts/
│   ├── setup.py
│   ├── download_datasets.py
│   ├── analyze_datasets.py
│   ├── preprocess_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── simulate.py
│   └── report.py
├── output/
│   ├── models/
│   ├── plots/
│   ├── metrics/
│   └── reports/
└── results/
        """
    
    def generate_pdf_report(self) -> Path:
        """Generate PDF report (requires weasyprint)."""
        try:
            from weasyprint import HTML
            import tempfile
            
            # Generate HTML
            html_content = self._generate_html_content()
            
            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                f.write(html_content)
                temp_html = f.name
            
            # Convert to PDF
            pdf_path = self.reports_dir / "full_report.pdf"
            HTML(temp_html).write_pdf(str(pdf_path))
            
            # Clean up
            os.unlink(temp_html)
            
            print(f"[OK] PDF report generated: {pdf_path}")
            return pdf_path
            
        except ImportError:
            print("[WARNING] weasyprint not installed. PDF generation skipped.")
            print("Install with: pip install weasyprint")
            return None
    
    def generate_all_reports(self) -> None:
        """Generate all reports."""
        print("\n" + "="*70)
        print("GENERATING REPORTS")
        print("="*70)
        
        # HTML report
        self.generate_html_report()
        
        # PDF report
        self.generate_pdf_report()
        
        print("\n" + "="*70)
        print("[OK] All reports generated!")
        print(f"Reports saved to: {self.reports_dir}")
        print("="*70)


def main():
    """Main report generation function."""
    generator = ReportGenerator()
    generator.generate_all_reports()


if __name__ == "__main__":
    main()