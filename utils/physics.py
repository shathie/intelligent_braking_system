"""
Physics models for vehicle dynamics.
Used in Physics-Informed Neural Networks (PINNs) for friction estimation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehicleParameters:
    """Physical parameters of the vehicle."""
    # Mass and dimensions
    m: float = 1500.0          # Vehicle mass [kg]
    I_z: float = 2500.0       # Yaw inertia [kg·m²]
    I_w: float = 1.5          # Wheel inertia [kg·m²]
    
    # Wheel geometry
    R_w: float = 0.3          # Wheel radius [m]
    track_width: float = 1.6  # Distance between wheels [m]
    wheelbase: float = 2.8    # Distance between axles [m]
    
    # Tire parameters
    C_alpha: float = 100000   # Cornering stiffness [N/rad]
    C_s: float = 10.0         # Longitudinal slip stiffness
    
    # Aerodynamics
    C_d: float = 0.3          # Drag coefficient
    A_f: float = 2.0          # Frontal area [m²]
    rho: float = 1.225        # Air density [kg/m³]
    
    # Gravity
    g: float = 9.81           # Gravitational acceleration [m/s²]


@dataclass
class TireParameters:
    """Tire model parameters."""
    # Pacejka tire model coefficients (simplified)
    B: float = 10.0          # Stiffness factor
    C: float = 1.5           # Shape factor
    D: float = 1.0           # Peak factor
    E: float = -1.0          # Curvature factor
    
    # Brush model parameters
    stiffness: float = 10.0  # Tire stiffness coefficient


class VehicleDynamics:
    """
    Vehicle dynamics model for physics-informed learning.
    Implements various tire models and vehicle motion equations.
    """
    
    def __init__(self, vehicle_params: VehicleParameters = None,
                 tire_params: TireParameters = None):
        self.vehicle = vehicle_params or VehicleParameters()
        self.tire = tire_params or TireParameters()
    
    def longitudinal_force(self, mu: torch.Tensor, F_z: torch.Tensor, 
                          slip: torch.Tensor) -> torch.Tensor:
        """
        Calculate longitudinal tire force using Brush Model.
        
        Args:
            mu: Friction coefficient (B,)
            F_z: Normal force [N] (B,)
            slip: Longitudinal slip ratio (B,)
        
        Returns:
            Longitudinal force [N] (B,)
        """
        # Brush Model: F_x = μ * F_z * (1 - exp(-C * |s| / μ))
        # Where C is the tire stiffness coefficient
        C = self.tire.stiffness
        
        # Clamp mu to avoid division by zero
        mu = torch.clamp(mu, min=1e-6)
        
        # Calculate force
        F_x = mu * F_z * (1 - torch.exp(-C * slip.abs() / mu))
        
        # Apply sign based on slip direction
        F_x = F_x * torch.sign(slip)
        
        return F_x
    
    def lateral_force(self, mu: torch.Tensor, F_z: torch.Tensor,
                     alpha: torch.Tensor) -> torch.Tensor:
        """
        Calculate lateral tire force using simplified Pacejka model.
        
        Args:
            mu: Friction coefficient (B,)
            F_z: Normal force [N] (B,)
            alpha: Slip angle [rad] (B,)
        
        Returns:
            Lateral force [N] (B,)
        """
        # Simplified Pacejka model
        B = self.tire.B
        C = self.tire.C
        D = self.tire.D * mu * F_z
        E = self.tire.E
        
        # Calculate force
        F_y = D * torch.sin(C * torch.arctan(B * alpha - E * (B * alpha - torch.arctan(B * alpha))))
        
        return F_y
    
    def slip_ratio(self, omega: torch.Tensor, v_x: torch.Tensor) -> torch.Tensor:
        """
        Calculate longitudinal slip ratio.
        
        Args:
            omega: Wheel angular velocity [rad/s] (B, 4)
            v_x: Vehicle longitudinal velocity [m/s] (B,)
        
        Returns:
            Slip ratio (B, 4)
        """
        # s = (R * ω - v_x) / max(R * ω, v_x)
        R_omega = self.vehicle.R_w * omega
        
        # Avoid division by zero
        denominator = torch.maximum(R_omega, v_x.unsqueeze(1))
        denominator = torch.clamp(denominator, min=1e-6)
        
        slip = (R_omega - v_x.unsqueeze(1)) / denominator
        
        return slip
    
    def slip_angle(self, v_x: torch.Tensor, v_y: torch.Tensor,
                  omega: torch.Tensor, steering: torch.Tensor) -> torch.Tensor:
        """
        Calculate slip angle for each wheel.
        
        Args:
            v_x: Longitudinal velocity [m/s] (B,)
            v_y: Lateral velocity [m/s] (B,)
            omega: Wheel angular velocities [rad/s] (B, 4)
            steering: Steering angle [rad] (B,)
        
        Returns:
            Slip angles [rad] (B, 4)
        """
        # Simplified calculation (assuming small angles)
        # For front wheels, slip angle depends on steering
        # For rear wheels, slip angle depends on vehicle slip
        
        # Vehicle slip angle at CG
        beta = torch.arctan2(v_y, v_x)
        
        # Slip angles for each wheel
        # Front left
        alpha_fl = beta + steering - torch.arctan2(v_y, v_x)
        # Front right
        alpha_fr = beta - steering - torch.arctan2(v_y, v_x)
        # Rear left and right (no steering)
        alpha_rl = beta - torch.arctan2(v_y, v_x)
        alpha_rr = beta - torch.arctan2(v_y, v_x)
        
        alpha = torch.stack([alpha_fl, alpha_fr, alpha_rl, alpha_rr], dim=1)
        
        return alpha
    
    def normal_forces(self, a_x: torch.Tensor, a_y: torch.Tensor,
                     pitch: torch.Tensor = None, roll: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate normal forces on each wheel.
        
        Args:
            a_x: Longitudinal acceleration [m/s²] (B,)
            a_y: Lateral acceleration [m/s²] (B,)
            pitch: Pitch angle [rad] (B,), optional
            roll: Roll angle [rad] (B,), optional
        
        Returns:
            Normal forces [N] (B, 4) - [FL, FR, RL, RR]
        """
        m = self.vehicle.m
        g = self.vehicle.g
        track_width = self.vehicle.track_width
        wheelbase = self.vehicle.wheelbase
        
        # Static load distribution (50/50 front/rear, 50/50 left/right)
        F_z_static = (m * g) / 4 * torch.ones_like(a_x.unsqueeze(1))
        
        # Dynamic load transfer due to longitudinal acceleration
        # Front wheels lose load, rear wheels gain load
        delta_F_z_x = (m * a_x * self.vehicle.h_cg) / wheelbase
        F_z_front = F_z_static[:, :2] - delta_F_z_x.unsqueeze(1) / 2
        F_z_rear = F_z_static[:, 2:] + delta_F_z_x.unsqueeze(1) / 2
        
        # Dynamic load transfer due to lateral acceleration
        delta_F_z_y = (m * a_y * self.vehicle.h_cg) / track_width
        F_z_left = torch.cat([F_z_front[:, 0:1], F_z_rear[:, 0:1]], dim=1) + delta_F_z_y.unsqueeze(1) / 2
        F_z_right = torch.cat([F_z_front[:, 1:2], F_z_rear[:, 1:2]], dim=1) - delta_F_z_y.unsqueeze(1) / 2
        
        # Combine
        F_z = torch.cat([F_z_left, F_z_right], dim=1)
        
        # Clamp to positive values
        F_z = torch.clamp(F_z, min=1e-3)
        
        return F_z
    
    def longitudinal_dynamics(self, F_x_total: torch.Tensor, 
                             F_rolling: torch.Tensor, 
                             F_aero: torch.Tensor, 
                             F_grade: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate longitudinal acceleration from forces.
        
        Args:
            F_x_total: Total longitudinal tire force [N] (B,)
            F_rolling: Rolling resistance [N] (B,)
            F_aero: Aerodynamic drag [N] (B,)
            F_grade: Grade force [N] (B,), optional
        
        Returns:
            Longitudinal acceleration [m/s²] (B,)
        """
        # Sum of opposing forces
        F_opposing = F_rolling + F_aero
        if F_grade is not None:
            F_opposing = F_opposing + F_grade
        
        # Net force
        F_net = F_x_total - F_opposing
        
        # Acceleration: a = F / m
        a_x = F_net / self.vehicle.m
        
        return a_x
    
    def rolling_resistance(self, v_x: torch.Tensor) -> torch.Tensor:
        """
        Calculate rolling resistance force.
        
        Args:
            v_x: Longitudinal velocity [m/s] (B,)
        
        Returns:
            Rolling resistance [N] (B,)
        """
        # Simplified model: F_rolling = C_rr * m * g
        C_rr = 0.01  # Rolling resistance coefficient
        F_rolling = C_rr * self.vehicle.m * self.vehicle.g * torch.ones_like(v_x)
        
        return F_rolling
    
    def aerodynamic_drag(self, v_x: torch.Tensor) -> torch.Tensor:
        """
        Calculate aerodynamic drag force.
        
        Args:
            v_x: Longitudinal velocity [m/s] (B,)
        
        Returns:
            Aerodynamic drag [N] (B,)
        """
        # F_drag = 0.5 * rho * C_d * A_f * v_x^2
        F_drag = 0.5 * self.vehicle.rho * self.vehicle.C_d * self.vehicle.A_f * (v_x ** 2)
        
        # Drag opposes motion
        F_drag = -torch.sign(v_x) * F_drag
        
        return F_drag
    
    def wheel_dynamics(self, T_brake: torch.Tensor, F_x: torch.Tensor,
                       omega: torch.Tensor) -> torch.Tensor:
        """
        Calculate wheel angular acceleration.
        
        Args:
            T_brake: Braking torque [N·m] (B, 4)
            F_x: Longitudinal tire force [N] (B, 4)
            omega: Wheel angular velocity [rad/s] (B, 4)
        
        Returns:
            Wheel angular acceleration [rad/s²] (B, 4)
        """
        # Net torque on wheel: T_net = T_brake - F_x * R_w
        T_net = T_brake - F_x * self.vehicle.R_w
        
        # Angular acceleration: alpha = T_net / I_w
        alpha = T_net / self.vehicle.I_w
        
        return alpha
    
    def physics_loss(self, state: Dict[str, torch.Tensor], 
                    pred_mu: torch.Tensor, 
                    vehicle_params: VehicleParameters = None) -> torch.Tensor:
        """
        Calculate physics-informed loss for PINN training.
        
        Args:
            state: Dictionary containing:
                - v_x: Longitudinal velocity [m/s] (B,)
                - a_x: Longitudinal acceleration [m/s²] (B,)
                - omega: Wheel speeds [rad/s] (B, 4)
                - F_brake: Braking force [N] (B,) or torque [N·m] (B, 4)
            pred_mu: Predicted friction coefficient (B,)
            vehicle_params: Optional vehicle parameters
        
        Returns:
            Physics loss (scalar)
        """
        params = vehicle_params or self.vehicle
        
        # Calculate normal forces
        F_z = self.normal_forces(state["a_x"], torch.zeros_like(state["a_x"]))
        
        # Calculate slip ratios
        slip = self.slip_ratio(state["omega"], state["v_x"])
        
        # Calculate tire forces
        F_x = self.longitudinal_force(pred_mu, F_z, slip)
        F_x_total = F_x.sum(dim=1)
        
        # Calculate predicted acceleration from forces
        F_rolling = self.rolling_resistance(state["v_x"])
        F_aero = self.aerodynamic_drag(state["v_x"])
        
        # Assume F_brake is total braking force (not per-wheel torque)
        # If it's torque, we'd need to convert
        if "F_brake" in state:
            F_brake = state["F_brake"]
        elif "T_brake" in state:
            # Convert torque to force: F = T / R_w
            F_brake = state["T_brake"].mean(dim=1) / params.R_w
        else:
            F_brake = torch.zeros_like(state["v_x"])
        
        pred_a_x = self.longitudinal_dynamics(F_x_total, F_rolling, F_aero) - F_brake / params.m
        
        # Dynamics residual: predicted acceleration vs actual acceleration
        dynamics_residual = pred_a_x - state["a_x"]
        dynamics_loss = torch.mean(dynamics_residual ** 2)
        
        # Wheel dynamics residual
        if "T_brake" in state:
            T_brake = state["T_brake"]
            alpha_pred = self.wheel_dynamics(T_brake, F_x, state["omega"])
            
            # We don't have actual wheel acceleration, so skip this for now
            # In practice, you'd need sequential data to compute this
            wheel_loss = torch.tensor(0.0)
        else:
            wheel_loss = torch.tensor(0.0)
        
        # Total physics loss
        total_loss = dynamics_loss + wheel_loss
        
        return total_loss


# Singleton instance for convenience
vehicle_dynamics = VehicleDynamics()


# Utility functions for PINN integration
def get_physics_constraints() -> Dict[str, callable]:
    """Get dictionary of physics constraint functions."""
    return {
        "longitudinal_dynamics": vehicle_dynamics.longitudinal_dynamics,
        "wheel_dynamics": vehicle_dynamics.wheel_dynamics,
        "tire_force": vehicle_dynamics.longitudinal_force,
        "slip_ratio": vehicle_dynamics.slip_ratio,
        "normal_forces": vehicle_dynamics.normal_forces
    }


def compute_physics_loss(state_dict: Dict, pred_mu: torch.Tensor) -> torch.Tensor:
    """
    Convenience function to compute physics loss.
    
    Args:
        state_dict: Dictionary of state variables
        pred_mu: Predicted friction coefficient
    
    Returns:
        Physics loss
    """
    return vehicle_dynamics.physics_loss(state_dict, pred_mu)


# Example usage
if __name__ == "__main__":
    # Create sample data
    batch_size = 32
    
    # State variables
    v_x = torch.rand(batch_size) * 30  # 0-30 m/s
    a_x = torch.rand(batch_size) * 10 - 5  # -5 to 5 m/s²
    omega = torch.rand(batch_size, 4) * 100  # 0-100 rad/s
    F_brake = torch.rand(batch_size) * 5000  # 0-5000 N
    
    state = {
        "v_x": v_x,
        "a_x": a_x,
        "omega": omega,
        "F_brake": F_brake
    }
    
    # Predicted friction
    pred_mu = torch.rand(batch_size) * 0.8 + 0.1  # 0.1-0.9
    
    # Compute physics loss
    loss = vehicle_dynamics.physics_loss(state, pred_mu)
    print(f"Physics loss: {loss.item():.6f}")