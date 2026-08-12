"""
Simulation script for the intelligent braking system.
Runs comprehensive simulations on different surfaces and generates performance reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vit import RoadSurfaceViT
from models.temporal_network import TemporalNetwork
from models.fusion_network import CrossModalAttentionFusion
from models.pinn import PhysicsInformedNetwork
from models.sac_agent import SACAgent
from models.mpc_controller import MPCController
from utils.visualization import PredictionVisualizer, SystemAnalyzer
from utils.metrics import ControlMetrics


class VehicleSimulator:
    """High-fidelity vehicle dynamics simulator."""
    
    def __init__(self):
        # Vehicle parameters
        self.m = 1500.0      # Mass [kg]
        self.I_w = 1.5       # Wheel inertia [kg·m^2]
        self.R_w = 0.3       # Wheel radius [m]
        self.C = 10.0        # Tire stiffness coefficient
        self.h_cg = 0.5      # Center of gravity height [m]
        self.track_width = 1.6  # Distance between wheels [m]
        self.wheelbase = 2.8    # Distance between axles [m]
        
        # Initial state
        self.v_x = 20.0      # Longitudinal velocity [m/s] (~72 km/h)
        self.v_y = 0.0       # Lateral velocity [m/s]
        self.omega = np.array([20.0, 20.0, 20.0, 20.0])  # Wheel speeds [rad/s]
        self.a_x = 0.0       # Longitudinal acceleration [m/s^2]
        self.a_y = 0.0       # Lateral acceleration [m/s^2]
        self.psi = 0.0       # Yaw angle [rad]
        self.psi_dot = 0.0   # Yaw rate [rad/s]
        self.steering = 0.0  # Steering angle [rad]
        
        # Environment
        self.mu = 0.8        # Friction coefficient
        self.surface = "dry" # Road surface type
        
        # Time
        self.dt = 0.01      # Time step [s]
        self.time = 0.0     # Current time [s]
        
        # History
        self.history = {
            'time': [],
            'v_x': [],
            'v_y': [],
            'omega': [],
            'a_x': [],
            'a_y': [],
            'psi_dot': [],
            'braking_force': [],
            'mu': []
        }
    
    def set_surface(self, surface: str, mu: float) -> None:
        """Set road surface type and friction coefficient."""
        self.surface = surface
        self.mu = mu
    
    def step(self, braking_force: float) -> Dict:
        """
        Simulate one time step.
        
        Args:
            braking_force: Braking force command (0-1)
        
        Returns:
            Dictionary with current state
        """
        # Convert braking force to torque (simplified)
        # Assume max braking torque is 1000 N·m per wheel
        max_torque = 1000.0
        T_brake = braking_force * max_torque
        
        # Calculate normal forces (simplified)
        F_z = self.m * 9.81 / 4  # Even distribution
        
        # Calculate slip ratios
        s = np.zeros(4)
        for i in range(4):
            s[i] = (self.R_w * self.omega[i] - self.v_x) / max(self.R_w * self.omega[i], self.v_x, 1e-6)
        
        # Calculate tire forces (Brush Model)
        F_x = self.mu * F_z * (1 - np.exp(-self.C * np.abs(s)))
        F_x = F_x * np.sign(s)  # Direction
        F_x_total = np.sum(F_x)
        
        # Rolling resistance (simplified)
        C_rr = 0.01  # Rolling resistance coefficient
        F_rolling = C_rr * self.m * 9.81
        
        # Aerodynamic drag
        rho = 1.225  # Air density [kg/m^3]
        C_d = 0.3    # Drag coefficient
        A_f = 2.0    # Frontal area [m^2]
        F_aero = 0.5 * rho * C_d * A_f * (self.v_x ** 2)
        F_aero = -np.sign(self.v_x) * F_aero
        
        # Longitudinal dynamics
        F_net = F_x_total - F_rolling - F_aero
        self.a_x = F_net / self.m
        
        # Update velocity
        self.v_x += self.a_x * self.dt
        self.v_x = max(self.v_x, 0)  # Can't go negative
        
        # Update wheel speeds
        for i in range(4):
            # Net torque on wheel
            T_net = T_brake - F_x[i] * self.R_w
            alpha = T_net / self.I_w  # Angular acceleration
            self.omega[i] += alpha * self.dt
            self.omega[i] = max(self.omega[i], 0)
        
        # Lateral dynamics (simplified)
        if self.steering != 0:
            # Simple lateral force based on steering
            F_y = -self.steering * self.m * 9.81 * 0.5
            self.a_y = F_y / self.m
            self.v_y += self.a_y * self.dt
            
            # Yaw dynamics
            M_z = self.steering * F_x_total * self.wheelbase * 0.5
            self.psi_dot = M_z / (self.m * self.wheelbase)
        
        # Update yaw angle
        self.psi += self.psi_dot * self.dt
        
        # Update time
        self.time += self.dt
        
        # Record history
        self.history['time'].append(self.time)
        self.history['v_x'].append(self.v_x)
        self.history['v_y'].append(self.v_y)
        self.history['omega'].append(self.omega.copy())
        self.history['a_x'].append(self.a_x)
        self.history['a_y'].append(self.a_y)
        self.history['psi_dot'].append(self.psi_dot)
        self.history['braking_force'].append(braking_force)
        self.history['mu'].append(self.mu)
        
        return {
            'v_x': self.v_x,
            'v_y': self.v_y,
            'omega': self.omega.copy(),
            'a_x': self.a_x,
            'a_y': self.a_y,
            'psi': self.psi,
            'psi_dot': self.psi_dot,
            'steering': self.steering,
            'mu': self.mu,
            'time': self.time
        }
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.v_x = 20.0
        self.v_y = 0.0
        self.omega = np.array([20.0, 20.0, 20.0, 20.0])
        self.a_x = 0.0
        self.a_y = 0.0
        self.psi = 0.0
        self.psi_dot = 0.0
        self.steering = 0.0
        self.time = 0.0
        
        # Clear history
        for key in self.history:
            self.history[key] = []


class BrakingSystem:
    """Integrated braking system with all components."""
    
    def __init__(self, models_dir: str = "output/models", device: str = None):
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = Path(models_dir)
        if (self.models_dir / "latest").exists():
            self.models_dir = self.models_dir / "latest"
        self._warned_missing_friction_models = False
        self._warned_missing_sac = False
        
        # Load models
        self.models = self._load_models()
        
        # Visualizer
        self.visualizer = PredictionVisualizer("output/plots")

    def _find_latest_model_path(self, model_name: str) -> Path:
        """Find the newest checkpoint for a model across run directories."""
        direct_path = self.models_dir / f"{model_name}_final.pth"
        candidates = []
        if direct_path.exists():
            candidates.append(direct_path)
        candidates.extend(self.models_dir.glob(f"**/{model_name}_final.pth"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    def _load_models(self) -> Dict:
        """Load all trained models."""
        models = {}
        
        # ViT
        vit_path = self._find_latest_model_path("vit")
        if vit_path is not None:
            vit = RoadSurfaceViT(num_classes=27, pretrained=False)
            vit.load_state_dict(torch.load(vit_path, map_location=self.device))
            vit.to(self.device).eval()
            models['vit'] = vit
            print("Loaded ViT model")
        
        # Temporal Network
        temporal_path = self._find_latest_model_path("temporal")
        if temporal_path is not None:
            temporal = TemporalNetwork(input_dim=17, hidden_dim=256, num_layers=2)
            temporal.load_state_dict(torch.load(temporal_path, map_location=self.device))
            temporal.to(self.device).eval()
            models['temporal'] = temporal
            print("Loaded Temporal Network")
        
        # Fusion Network
        fusion_path = self._find_latest_model_path("fusion")
        if fusion_path is not None:
            fusion = CrossModalAttentionFusion(vit_dim=512, temporal_dim=256, fusion_dim=512)
            fusion.load_state_dict(torch.load(fusion_path, map_location=self.device))
            fusion.to(self.device).eval()
            models['fusion'] = fusion
            print("Loaded Fusion Network")
        
        # PINN
        pinn_path = self._find_latest_model_path("pinn")
        if pinn_path is not None:
            pinn = PhysicsInformedNetwork(input_dim=512, hidden_dims=[256, 128, 64])
            pinn.load_state_dict(torch.load(pinn_path, map_location=self.device))
            pinn.to(self.device).eval()
            models['pinn'] = pinn
            print("Loaded PINN")
        
        # SAC Agent
        sac_path = self._find_latest_model_path("sac")
        if sac_path is not None:
            sac = SACAgent(state_dim=10, action_dim=1, config={"device": str(self.device)})
            sac.load(sac_path)
            models['sac'] = sac
            print("Loaded SAC Agent")
        
        # MPC Controller (if trained)
        if (self.models_dir / "mpc_final.pth").exists():
            # MPC would need special loading
            print("MPC Controller loading not implemented")
        
        return models

    @staticmethod
    def _extract_temporal_features(temporal_output: torch.Tensor) -> torch.Tensor:
        """Normalize temporal model outputs to feature tensor."""
        if isinstance(temporal_output, (tuple, list)):
            return temporal_output[0]
        return temporal_output

    def _align_can_sequence(self, can_sequence: torch.Tensor) -> torch.Tensor:
        """Pad or truncate CAN sequence to temporal model input size."""
        if can_sequence.dim() != 2:
            raise ValueError(f"Expected CAN sequence shape (seq_len, features), got {tuple(can_sequence.shape)}")

        temporal_model = self.models.get('temporal')
        expected_dim = int(getattr(temporal_model, 'input_dim', can_sequence.size(-1))) if temporal_model is not None else can_sequence.size(-1)
        current_dim = can_sequence.size(-1)

        if current_dim == expected_dim:
            return can_sequence

        if current_dim < expected_dim:
            pad = torch.zeros(can_sequence.size(0), expected_dim - current_dim, dtype=can_sequence.dtype, device=can_sequence.device)
            return torch.cat([can_sequence, pad], dim=-1)

        return can_sequence[:, :expected_dim]
    
    def estimate_friction(self, image: torch.Tensor, can_sequence: torch.Tensor) -> float:
        """
        Estimate friction coefficient from image and CAN data.
        
        Args:
            image: Preprocessed image tensor (3, 224, 224)
            can_sequence: CAN sequence tensor (seq_len, features)
        
        Returns:
            Estimated friction coefficient mu
        """
        if not all(m in self.models for m in ['vit', 'temporal', 'fusion', 'pinn']):
            if not self._warned_missing_friction_models:
                print("Missing models for friction estimation")
                self._warned_missing_friction_models = True
            return 0.5  # Default value
        
        with torch.no_grad():
            # Get ViT features
            vit_out = self.models['vit'](image.unsqueeze(0).to(self.device))
            vit_features = vit_out['features']
            
            # Get temporal features
            can_sequence = self._align_can_sequence(can_sequence)
            temporal_out = self.models['temporal'](can_sequence.unsqueeze(0).to(self.device))
            temporal_features = self._extract_temporal_features(temporal_out)
            
            # Fusion
            fusion_features = self.models['fusion'](vit_features, temporal_features)
            
            # PINN prediction
            mu = self.models['pinn'](fusion_features)
            
            return float(mu.squeeze().cpu().numpy())
    
    def get_braking_command(self, state: np.ndarray) -> float:
        """
        Get braking command from SAC agent.
        
        Args:
            state: State vector (10,)
        
        Returns:
            Braking force command (0-1)
        """
        if 'sac' not in self.models:
            if not self._warned_missing_sac:
                print("SAC agent not loaded")
                self._warned_missing_sac = True
            return 0.5  # Default: moderate braking
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.models['sac'].get_action(state_tensor, deterministic=True)
        
        return float(np.clip(action[0], 0, 1))


class SimulationRunner:
    """Run comprehensive simulations."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize braking system
        self.braking_system = BrakingSystem()
        
        # Control metrics
        self.control_metrics = ControlMetrics()
    
    def run_single_simulation(self, surface: str, true_mu: float, 
                            initial_velocity: float = 20.0,
                            use_estimated_mu: bool = False) -> Dict:
        """
        Run a single simulation on a specific surface.
        
        Args:
            surface: Road surface type ('dry', 'wet', 'icy', 'rough')
            true_mu: True friction coefficient
            initial_velocity: Initial velocity [m/s]
            use_estimated_mu: Whether to use estimated mu in state (or true mu)
        
        Returns:
            Dictionary with simulation results
        """
        # Initialize simulator
        simulator = VehicleSimulator()
        simulator.v_x = initial_velocity
        simulator.omega = np.array([initial_velocity / simulator.R_w] * 4)
        simulator.set_surface(surface, true_mu)
        
        # Run simulation
        states = []
        actions = []
        times = []
        velocities = []
        friction_estimates = []
        
        # Create dummy image and CAN sequence for friction estimation
        # In practice, you'd have real sensor data
        dummy_image = torch.randn(3, 224, 224)
        temporal_model = self.braking_system.models.get('temporal')
        temporal_input_dim = int(getattr(temporal_model, 'input_dim', 17))
        dummy_can_seq = torch.randn(50, temporal_input_dim) * 0.5 + 0.5  # Normalized
        
        # Get initial friction estimate
        mu_estimate = self.braking_system.estimate_friction(dummy_image, dummy_can_seq)
        
        obs = simulator.step(0)
        
        for t in range(500):
            # Get state
            if use_estimated_mu:
                state_mu = mu_estimate
            else:
                state_mu = true_mu  # Use true mu for control
            
            state = np.array([
                state_mu,
                obs["v_x"],
                obs["omega"][0],
                obs["omega"][1],
                obs["omega"][2],
                obs["omega"][3],
                obs["a_x"],
                obs["psi_dot"],
                obs["steering"],
                0.0  # Brake pressure (placeholder)
            ])
            
            # Get braking command
            braking_force = self.braking_system.get_braking_command(state)
            
            # Apply braking
            obs = simulator.step(braking_force)
            
            # Update friction estimate (with new dummy data)
            mu_estimate = self.braking_system.estimate_friction(dummy_image, dummy_can_seq)
            
            # Record
            states.append(state.copy())
            actions.append(braking_force)
            times.append(simulator.time)
            velocities.append(obs["v_x"])
            friction_estimates.append(mu_estimate)
            
            # Stop if vehicle has stopped
            if obs["v_x"] < 0.1:
                break
        
        # Calculate metrics
        stopping_distance = np.trapezoid(velocities, times)
        stopping_time = times[-1] if times else 0
        
        # Calculate jerk
        accelerations = [s[6] for s in states]  # a_x is at index 6
        jerks = []
        for i in range(1, len(accelerations)):
            if times[i] - times[i-1] > 0:
                jerk = (accelerations[i] - accelerations[i-1]) / (times[i] - times[i-1])
                jerks.append(abs(jerk))
        max_jerk = np.max(jerks) if jerks else 0
        mean_jerk = np.mean(jerks) if jerks else 0
        
        # Calculate wheel slip
        wheel_slips = []
        for s in states:
            R_omega = simulator.R_w * s[2:6]  # wheel speeds at indices 2-5
            v_x = s[1]
            for i in range(4):
                slip = (R_omega[i] - v_x) / max(R_omega[i], v_x, 1e-6)
                wheel_slips.append(abs(slip))
        max_slip = np.max(wheel_slips) if wheel_slips else 0
        mean_slip = np.mean(wheel_slips) if wheel_slips else 0
        
        return {
            'surface': surface,
            'true_mu': true_mu,
            'initial_velocity': initial_velocity,
            'stopping_distance': stopping_distance,
            'stopping_time': stopping_time,
            'max_jerk': max_jerk,
            'mean_jerk': mean_jerk,
            'max_slip': max_slip,
            'mean_slip': mean_slip,
            'final_mu_estimate': mu_estimate,
            'times': times,
            'velocities': velocities,
            'actions': actions,
            'friction_estimates': friction_estimates
        }
    
    def run_comprehensive_simulation(self, num_runs: int = 5) -> Dict:
        """Run comprehensive simulations on all surfaces."""
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE SIMULATIONS")
        print("="*70)
        
        # Define test scenarios
        scenarios = [
            ('dry', 0.8, 20.0),
            ('wet', 0.4, 20.0),
            ('icy', 0.1, 20.0),
            ('rough', 0.6, 20.0)
        ]
        
        results = {}
        
        for surface, true_mu, initial_velocity in scenarios:
            print(f"\nSimulating on {surface} surface (mu = {true_mu})...")
            
            surface_results = []
            
            for run in range(num_runs):
                result = self.run_single_simulation(
                    surface, true_mu, initial_velocity,
                    use_estimated_mu=True
                )
                surface_results.append(result)
                
                print(f"  Run {run+1}: Stopping distance = {result['stopping_distance']:.2f} m, "
                      f"Stopping time = {result['stopping_time']:.2f} s")
            
            # Aggregate results
            results[surface] = self._aggregate_results(surface_results)
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        # Print summary
        self._print_summary(results)
        
        print("\n" + "="*70)
        print("Comprehensive simulations complete!")
        print(f"Results saved to: {self.results_dir}")
        print("="*70)
        
        return results
    
    def _aggregate_results(self, surface_results: List[Dict]) -> Dict:
        """Aggregate results from multiple runs."""
        aggregated = {}
        
        # Collect all values for each metric
        metrics = ['stopping_distance', 'stopping_time', 'max_jerk', 'mean_jerk', 
                   'max_slip', 'mean_slip', 'final_mu_estimate']
        
        for metric in metrics:
            values = [r[metric] for r in surface_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Add surface info
        aggregated['surface'] = surface_results[0]['surface']
        aggregated['true_mu'] = surface_results[0]['true_mu']
        aggregated['initial_velocity'] = surface_results[0]['initial_velocity']
        aggregated['num_runs'] = len(surface_results)
        
        # Add sample trajectories
        aggregated['sample_times'] = surface_results[0]['times']
        aggregated['sample_velocities'] = surface_results[0]['velocities']
        aggregated['sample_actions'] = surface_results[0]['actions']
        
        return aggregated
    
    def _save_results(self, results: Dict) -> None:
        """Save simulation results to JSON."""
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for surface, metrics in results.items():
            serializable_results[surface] = {}
            for metric, values in metrics.items():
                if isinstance(values, dict):
                    serializable_results[surface][metric] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in values.items()
                    }
                elif isinstance(values, (np.ndarray, list)):
                    serializable_results[surface][metric] = [float(v) for v in values]
                else:
                    serializable_results[surface][metric] = float(values) if isinstance(values, (np.floating, np.integer)) else values
        
        # Save to JSON
        results_path = self.results_dir / "simulation_results.json"
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    def _generate_visualizations(self, results: Dict) -> None:
        """Generate visualization plots."""
        # Plot stopping distances
        surfaces = list(results.keys())
        stopping_distances = [results[s]['stopping_distance']['mean'] for s in surfaces]
        stopping_distances_std = [results[s]['stopping_distance']['std'] for s in surfaces]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(surfaces, stopping_distances, yerr=stopping_distances_std, 
               capsize=5, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Road Surface")
        ax.set_ylabel("Stopping Distance [m]")
        ax.set_title("Stopping Distance by Surface Type")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "stopping_distance.png")
        plt.close()
        
        # Plot stopping times
        stopping_times = [results[s]['stopping_time']['mean'] for s in surfaces]
        stopping_times_std = [results[s]['stopping_time']['std'] for s in surfaces]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(surfaces, stopping_times, yerr=stopping_times_std, 
               capsize=5, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Road Surface")
        ax.set_ylabel("Stopping Time [s]")
        ax.set_title("Stopping Time by Surface Type")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "stopping_time.png")
        plt.close()
        
        # Plot max jerk
        max_jerks = [results[s]['max_jerk']['mean'] for s in surfaces]
        max_jerks_std = [results[s]['max_jerk']['std'] for s in surfaces]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(surfaces, max_jerks, yerr=max_jerks_std, 
               capsize=5, color='salmon', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Road Surface")
        ax.set_ylabel("Max Jerk [m/s^3]")
        ax.set_title("Max Jerk by Surface Type")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "max_jerk.png")
        plt.close()
        
        # Plot sample trajectories
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for i, surface in enumerate(surfaces):
            if i >= 4:
                break
            
            times = results[surface]['sample_times']
            velocities = results[surface]['sample_velocities']
            
            axes[i].plot(times, velocities, 'b-', linewidth=2)
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel("Velocity [m/s]")
            axes[i].set_title(f"{surface.capitalize()} Surface")
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle("Sample Velocity Trajectories")
        plt.tight_layout()
        plt.savefig(self.results_dir / "velocity_trajectories.png")
        plt.close()
        
        # Plot braking force trajectories
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for i, surface in enumerate(surfaces):
            if i >= 4:
                break
            
            times = results[surface]['sample_times']
            actions = results[surface]['sample_actions']
            
            axes[i].plot(times, actions, 'r-', linewidth=2)
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel("Braking Force")
            axes[i].set_title(f"{surface.capitalize()} Surface")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([0, 1])
        
        plt.suptitle("Sample Braking Force Trajectories")
        plt.tight_layout()
        plt.savefig(self.results_dir / "braking_trajectories.png")
        plt.close()
        
        print(f"Visualizations saved to: {self.results_dir}")
    
    def _print_summary(self, results: Dict) -> None:
        """Print simulation summary."""
        print("\n" + "="*70)
        print("SIMULATION SUMMARY")
        print("="*70)
        
        for surface in results:
            print(f"\n{surface.upper()}:")
            print(f"  True mu: {results[surface]['true_mu']:.2f}")
            print(f"  Stopping Distance: {results[surface]['stopping_distance']['mean']:.2f} +/- {results[surface]['stopping_distance']['std']:.2f} m")
            print(f"  Stopping Time: {results[surface]['stopping_time']['mean']:.2f} +/- {results[surface]['stopping_time']['std']:.2f} s")
            print(f"  Max Jerk: {results[surface]['max_jerk']['mean']:.2f} +/- {results[surface]['max_jerk']['std']:.2f} m/s^3")
            print(f"  Max Slip: {results[surface]['max_slip']['mean']:.2f} +/- {results[surface]['max_slip']['std']:.2f}")
            print(f"  Final mu Estimate: {results[surface]['final_mu_estimate']['mean']:.2f} +/- {results[surface]['final_mu_estimate']['std']:.2f}")


def main():
    """Main simulation function."""
    runner = SimulationRunner()
    
    # Run comprehensive simulations
    results = runner.run_comprehensive_simulation(num_runs=5)
    
    # Summary
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {runner.results_dir}")
    print("="*70)


if __name__ == "__main__":
    main()